"""The score models: DARTS's conditional one (blind-spot context encoder plus noise-conditional
head) and the context-free one (the head alone, DART), with their checkpoint files.

The model works in `Standardize` coordinates z of the whitened spectra x. Training noises the centre
with the variance-preserving kernel, noised = a z + s eps with a = (1 + sigma^2)^(-1/2) and
s = sigma a, and the head predicts eps from the noised centre, the clean context and sigma:

    eps_hat = a out + s noised

(the v-parametrization). The minimizer is the conditional Tweedie predictor, so
-eps_hat(a z, c, sigma) / (sigma std) is the score at x of p0(x | c) smoothed by
N(0, sigma^2 diag(std^2)), in whitened units (`reads.rao`).
"""

import os
from pathlib import Path

import torch

from dartsdet.encoder import BlindContext
from dartsdet.network import ScoreMLP


def save_atomically(payload: dict, path: Path) -> None:
    """torch.save to a temporary file first, so that a crash leaves no partial file at `path`."""
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def checked(saved: dict, expected: dict, path: Path) -> dict:
    """`saved`, once each key of `expected` holds the same value in it; refused otherwise."""
    if wrong := sorted(k for k, v in expected.items() if saved.get(k) != v):
        raise ValueError(f"{path} was saved with other {', '.join(wrong)}")
    return saved


def vp_coefficients(sigma: float | torch.Tensor) -> tuple[float | torch.Tensor,
                                                         float | torch.Tensor]:
    """(a, s) of the variance-preserving kernel at noise-to-signal ratio sigma: a^2 + s^2 = 1,
    s / a = sigma."""
    a = (1 + sigma**2) ** -0.5
    return a, sigma * a


class Standardize(torch.nn.Module):
    """Per-band z-score (x - mean) / std of spectra (n, B)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardize":
        """Mean and divide-by-n standard deviation of spectra x (n, B)."""
        std = x.std(dim=0, correction=0)
        if (std == 0).any():
            raise ValueError(f"constant bands cannot be standardized: {std.eq(0).nonzero()}")
        return cls(x.mean(dim=0), std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ScoreModel(torch.nn.Module):
    """eps_hat of a pixel from its noised standardized spectrum, its context and sigma."""

    def __init__(self, coordinates: Standardize, spec: dict) -> None:
        """spec: the recipe (configs/darts.json): `sigma_min`, `sigma_max`, `width`, `depth`,
        `network` (the head's) and `encoder` (`BlindContext`'s)."""
        super().__init__()
        self.spec, self.coordinates = spec, coordinates
        self.sigma_min, self.sigma_max = spec["sigma_min"], spec["sigma_max"]
        n_bands = coordinates.mean.shape[0]
        self.encoder = BlindContext(n_bands, spec["encoder"])
        self.network = ScoreMLP(n_bands, spec["encoder"]["context"], spec["width"], spec["depth"],
                                spec["network"])
        self.history, self.validation, self.best_step, self.record = [], [], 0, {}

    def image(self, fx: torch.Tensor, usable: torch.Tensor,
              grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """(standardized image (1, B, W, H), mask (1, 1, W, H)) of whitened spectra fx (P, B)
        in pixel order, and the pixels contexts may use, usable (P,)."""
        W, H = grid
        z = self.coordinates(fx).reshape(W, H, -1).permute(2, 0, 1)[None]
        return z, usable.reshape(W, H)[None, None].to(z)

    def epsilon(self, noised: torch.Tensor, context: torch.Tensor,
                sigma: torch.Tensor | float) -> torch.Tensor:
        """eps_hat (n, B) from noised standardized centres (n, B), contexts (n, C) and sigma, (n,)
        or a scalar."""
        sigma = torch.as_tensor(sigma, dtype=noised.dtype, device=noised.device).expand(len(noised))
        a, s = vp_coefficients(sigma[:, None])
        return a * self.network(torch.cat([noised, context], dim=1), sigma.log()) + s * noised

    def save(self, path: Path, **record: object) -> None:
        """The recipe, the weights and the training record, plus `record` (kept as `self.record`
        on loading); written to a temporary file first, so a crash leaves no partial file."""
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        save_atomically({"spec": self.spec, "n_bands": self.coordinates.mean.shape[0],
                         "state": state, "history": self.history, "validation": self.validation,
                         "best_step": self.best_step} | record, path)

    @classmethod
    def load(cls, path: Path) -> "ScoreModel":
        """A saved model, on the CPU."""
        saved = torch.load(path, weights_only=True, map_location="cpu")
        n = saved.pop("n_bands")
        model = cls(Standardize(torch.zeros(n), torch.ones(n)), saved.pop("spec"))
        model.load_state_dict(saved.pop("state"))
        model.history, model.validation = saved.pop("history"), saved.pop("validation")
        model.best_step, model.record = saved.pop("best_step"), saved
        return model


class ContextFreeScore(torch.nn.Module):
    """eps_hat of a standardized spectrum alone: the score head without a context (DART).

    It takes `ScoreModel`'s place in `reads` with an empty (n, 0) context.
    """

    def __init__(self, coordinates: Standardize, width: int, depth: int, head: dict) -> None:
        """head: the `network` spec of `ScoreMLP`."""
        super().__init__()
        self.coordinates = coordinates
        self.network = ScoreMLP(coordinates.mean.shape[0], 0, width, depth, head)

    def epsilon(self, noised: torch.Tensor, context: torch.Tensor,  # noqa: ARG002
                sigma: torch.Tensor | float) -> torch.Tensor:
        """eps_hat (n, B) of noised standardized spectra (n, B) at sigma, (n,) or a scalar."""
        sigma = torch.as_tensor(sigma, dtype=noised.dtype, device=noised.device).expand(len(noised))
        a, s = vp_coefficients(sigma[:, None])
        return a * self.network(noised, sigma.log()) + s * noised
