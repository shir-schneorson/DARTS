"""The per-scene steps the experiments share: fold models, the noise-level rule's reads, and the
Rao statistic at the chosen noise level.

The rule reads, from each fold, the log ratios of up to `SAMPLE` held-out and `SAMPLE` screened
pixels, each moved by one draw of smoothing noise per pixel (shared by every noise level), and the
curvatures of every clean held-out pixel (`noise_level`). P is the number of pixels, S the number
of noise levels and L = len(noise_level.LR_THETAS).
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from dartsdet import noise_level
from dartsdet.model import ScoreModel
from dartsdet.preprocessing import Inputs
from dartsdet.reads import contexts, log_ratio, rao, rao_and_curvature
from dartsdet.training import fit, held_out

SAMPLE = 20000                                  # noised pixels per fold and pool
NOISE_SEEDS = {"held": 1000, "screened": 1500}  # plus the fold index


def pixels_digest(inputs: Inputs) -> str:
    """sha256 of the usable-pixel mask: which pixels a fold model trained and held out on."""
    return hashlib.sha256(inputs.usable.numpy().tobytes()).hexdigest()


def fold_model(inputs: Inputs, spec: dict, fold: int, device: str, path: Path) -> ScoreModel:
    """Fold `fold` of the recipe, in evaluation mode on `device`: loaded from `path` if it is
    there, else trained and saved there. A saved model is refused unless it was trained with this
    recipe, as this fold, on these pixels."""
    if path.exists():
        model = ScoreModel.load(path)
        trained = (model.spec, model.record.get("fold"), model.record.get("pixels"))
        if trained != (spec, fold, pixels_digest(inputs)):
            raise ValueError(f"{path} was not trained as fold {fold} of this recipe on these "
                             "pixels")
    else:
        model = fit(inputs.fx.float(), inputs.usable, inputs.grid, fold, spec, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path, fold=fold, pixels=pixels_digest(inputs))
    return model.to(device).eval().requires_grad_(False)


def unit_signature(inputs: Inputs, device: str) -> torch.Tensor:
    """t / |t| (B,) in float32 on `device`."""
    return (inputs.t / inputs.t.norm()).float().to(device)


def sample(pixels: np.ndarray, fold: int) -> torch.Tensor:
    """Up to `SAMPLE` of the pixel indices `pixels`, drawn with the fold's seed, in pixel order."""
    rng = np.random.default_rng(fold)
    return torch.from_numpy(np.sort(rng.choice(pixels, min(len(pixels), SAMPLE), replace=False)))


def noised_log_ratios(model: ScoreModel, zc: torch.Tensor, context: torch.Tensor,
                      t_hat: torch.Tensor, seed: int, sigmas: list[float]) -> np.ndarray:
    """l (n, S, L) of standardized centres zc (n, B) moved to zc + sigma eps for each sigma, one
    standard normal eps per centre drawn with `seed` and shared by every sigma."""
    eps = torch.randn(len(zc), zc.shape[1], generator=torch.Generator().manual_seed(seed))
    eps = eps.to(zc.device)
    return np.stack([log_ratio(model, zc + s * eps, context, t_hat, s,
                               noise_level.LR_THETAS).double().numpy() for s in sigmas], 1)


def rule_reads(model: ScoreModel, inputs: Inputs, exists: np.ndarray, fold: int,
               sigmas: list[float], device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold `fold`'s reads on the noise levels `sigmas`: log ratios of noised held-out pixels
    (n_h, S, L) and of noised screened pixels (n_s, S, L), and curvatures of the clean held-out
    pixels (n_h', S); exists: (P,) bool."""
    zc, ctx = contexts(model, inputs.fx.float().to(device), inputs.usable.to(device), inputs.grid)
    t_hat = unit_signature(inputs, device)
    held = np.zeros(len(exists), dtype=bool)
    held[held_out(inputs.usable, fold, model.spec["train"]["folds"])[0].numpy()] = True
    pools = {"held": held, "screened": exists & ~inputs.usable.numpy()}
    ell = {}
    for pool, seed in NOISE_SEEDS.items():
        rows = sample(np.flatnonzero(pools[pool]), fold).to(device)
        ell[pool] = noised_log_ratios(model, zc[rows], ctx[rows], t_hat, seed + fold, sigmas)
    rows = torch.from_numpy(np.flatnonzero(held)).to(device)
    _, C = rao_and_curvature(model, zc[rows], ctx[rows], t_hat, sigmas)
    return ell["held"], ell["screened"], C.double().numpy()


def sigma_hat(reads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
              sigmas: list[float]) -> float:
    """The rule's sigma from the `rule_reads` of one or more folds on `sigmas`, pooled."""
    held, screened, C = (np.concatenate(parts) for parts in zip(*reads, strict=True))
    return noise_level.noise_level(held, screened, C, np.array(sigmas))


def statistic(model: ScoreModel, inputs: Inputs, sigma: float, device: str) -> np.ndarray:
    """The fold's Rao statistic (P,) at sigma."""
    zc, ctx = contexts(model, inputs.fx.float().to(device), inputs.usable.to(device), inputs.grid)
    with torch.no_grad():
        return rao(model, zc, ctx, unit_signature(inputs, device), sigma).cpu().double().numpy()
