"""DARTS's conditional score model, eq. (cond-eps) (Sections V-B and V-C).

The model reads standardized whitened spectra z = (x - mean) / std, band by band, as images
(N, B, W, H) whose pixels are in `data`'s Fortran order: pixel p of an image is (w, h) = (p // H,
p % H). At noise level sigma, with a = (1 + sigma^2)^(-1/2) and s = sigma a, it receives the
variance-preserving input u = a y of a noised centre y = z + sigma eps and predicts the noise

    eps_hat = a d psi / d u + s u,                                               (cond-eps)

the paper's grad_y psi(a y) + sigma a^2 y, from a scalar potential psi(u, c, sigma) of the centre,
its context c and sigma. The score of the smoothed density is -eps_hat / sigma, in standardized
units; with psi = 0 it is that of a standard Gaussian.

Encoder (`BlindContext`): a pointwise stem of the clean image a z, modulated by sigma (FiLM, from
the head's embedding of log sigma); screened pixels zeroed and marked by an indicator channel; a
half-plane network (`HalfPlane`) of residual 2 x 3 convolutions, padded on top and shifted down by
`shift` rows, applied to the four rotations of the image; the rotated-back outputs combined
pointwise into c. Normalization is per pixel over channels, so no statistic leaks a pixel into its
own context. With shift 1 and dilations summing to 11, a context reads the 25 x 25 window around
its pixel minus the pixel and the window's four corners.

Head (`PotentialHead`): a residual MLP of [u, c], each block FiLM-conditioned on a Fourier
embedding e of log sigma. Its output layer starts at zero, so an untrained model has psi = 0.

With `context` 0 the model has no encoder: psi(u, sigma) alone, the smoothed score of the pooled
density p0. This is DART, which differs from DARTS by the context alone. With `exact_layer_norm`
the head's layer norms are computed by elementary operations (`ExactLayerNorm`), as the networks
trained with LRao's objective were.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F


def vp(sigma: float | torch.Tensor) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    """(a, s) of the variance-preserving kernel at noise level sigma: a^2 + s^2 = 1, s = sigma a."""
    a = (1 + sigma**2) ** -0.5
    return a, sigma * a


def pixels(image: torch.Tensor) -> torch.Tensor:
    """Images (N, C, W, H) as rows (N W H, C), in pixel order within each image."""
    n, c, w, h = image.shape
    return image.permute(0, 2, 3, 1).reshape(n * w * h, c)


class FourierFeatures(torch.nn.Module):
    """[sin 2 pi f u, cos 2 pi f u] of a scalar u per row, (n,) -> (n, n_features), f fixed
    N(0, scale^2) draws."""

    def __init__(self, n_features: int, scale: float) -> None:
        super().__init__()
        frequencies = torch.randn(n_features // 2, dtype=torch.float64) * scale
        self.register_buffer("frequencies", frequencies.float())

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        angle = 2 * math.pi * u[:, None] * self.frequencies
        return torch.cat([angle.sin(), angle.cos()], dim=1)


class PixelNorm(torch.nn.Module):
    """Layer normalization over the channels at each pixel, (N, C, W, H) -> (N, C, W, H)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        centred = h - h.mean(1, keepdim=True)
        scale = torch.rsqrt(centred.pow(2).mean(1, keepdim=True) + 1e-5)
        return centred * scale * self.weight + self.bias


class HalfPlane(torch.nn.Module):
    """Features at each pixel from the rows at least `shift` above it, (N, C, W, H) -> same: each
    residual (2 x 3) convolution, padded on top, reads its own row and the row `dilation` above."""

    def __init__(self, channels: int, dilations: list[int], shift: int,
                 act: type[torch.nn.Module]) -> None:
        super().__init__()
        self.dilations, self.shift = list(dilations), shift
        self.norms = torch.nn.ModuleList(PixelNorm(channels) for _ in dilations)
        self.convs = torch.nn.ModuleList(torch.nn.Conv2d(channels, channels, (2, 3), dilation=d)
                                         for d in dilations)
        self.act = act()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for norm, conv, d in zip(self.norms, self.convs, self.dilations, strict=True):
            h = h + conv(F.pad(self.act(norm(h)), (d, d, d, 0)))
        return F.pad(h, (0, 0, self.shift, 0))[:, :, : h.shape[2]]


class BlindContext(torch.nn.Module):
    """(image (N, B, W, H), mask (N, 1, W, H), FiLM gamma and beta (N, channels - 1, 1, 1)) ->
    contexts (N, context, W, H), blind to each pixel and to the pixels outside the mask."""

    def __init__(self, n_bands: int, spec: dict) -> None:
        """spec: the recipe's `model` block (`channels`, `dilations`, `shift`, `context`,
        `activation`)."""
        super().__init__()
        channels, act = spec["channels"], getattr(torch.nn, spec["activation"])
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(n_bands, channels, 1), act(),
                                        torch.nn.Conv2d(channels, channels - 1, 1))
        self.branch = HalfPlane(channels, spec["dilations"], spec["shift"], act)
        self.norm = PixelNorm(4 * channels)
        self.out = torch.nn.Sequential(torch.nn.Conv2d(4 * channels, channels, 1), act(),
                                       torch.nn.Conv2d(channels, spec["context"], 1))

    def forward(self, image: torch.Tensor, mask: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
        h = torch.cat([(self.stem(image) * (1 + gamma) + beta) * mask, mask], dim=1)
        views = [None] * 4
        # Rotations of one shape run as one batch: all four of a square image, else (0, 2), (1, 3).
        for turns in [(0, 1, 2, 3)] if h.shape[2] == h.shape[3] else [(0, 2), (1, 3)]:
            out = self.branch(torch.cat([torch.rot90(h, k, (2, 3)) for k in turns]))
            for k, view in zip(turns, out.chunk(len(turns)), strict=True):
                views[k] = torch.rot90(view, -k, (2, 3))
        return self.out(self.norm(torch.cat(views, dim=1)))


class ExactLayerNorm(torch.nn.LayerNorm):
    """torch's LayerNorm computed by elementary operations: the same function and parameters, with
    exact third derivatives, which the fused kernel gets wrong and LRao's objective needs."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = h.mean(dims, keepdim=True)
        out = (h - mean) / torch.sqrt(((h - mean) ** 2).mean(dims, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out + self.bias if self.bias is not None else out


class Block(torch.nn.Module):
    """h + W_down act(W_up (LN(h) (1 + gamma(e)) + beta(e))), (n, width) -> (n, width), W_up
    widening `expand` times."""

    def __init__(self, width: int, expand: int, embed_dim: int, act: type[torch.nn.Module],
                 norm: type[torch.nn.LayerNorm]) -> None:
        super().__init__()
        self.norm = norm(width, elementwise_affine=False)
        self.film = torch.nn.Linear(embed_dim, 2 * width)
        self.up = torch.nn.Linear(width, expand * width)
        self.down = torch.nn.Linear(expand * width, width)
        self.act = act()

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(e).chunk(2, dim=1)
        return h + self.down(self.act(self.up(self.norm(h) * (1 + gamma) + beta)))


class PotentialHead(torch.nn.Module):
    """psi (n,) of centres u (n, B), contexts c (n, C) and log sigma (n,)."""

    def __init__(self, n_bands: int, spec: dict) -> None:
        """spec: the recipe's `model` block (`context`, `width`, `depth`, `expand`, `embed_dim`,
        `fourier_scale`, `activation`)."""
        super().__init__()
        act = getattr(torch.nn, spec["activation"])
        embed_dim, width = spec["embed_dim"], spec["width"]
        self.embed = torch.nn.Sequential(FourierFeatures(embed_dim, spec["fourier_scale"]),
                                         torch.nn.Linear(embed_dim, embed_dim), act(),
                                         torch.nn.Linear(embed_dim, embed_dim), act())
        self.input = torch.nn.Linear(n_bands + spec["context"], width)
        norm = ExactLayerNorm if spec.get("exact_layer_norm") else torch.nn.LayerNorm
        self.blocks = torch.nn.ModuleList(Block(width, spec["expand"], embed_dim, act, norm)
                                          for _ in range(spec["depth"]))
        self.norm = norm(width)
        self.output = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, u: torch.Tensor, c: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        e = self.embed(log_sigma)
        h = self.input(torch.cat([u, c], dim=1))
        for block in self.blocks:
            h = block(h, e)
        return self.output(self.norm(h))[:, 0]


class DARTS(torch.nn.Module):
    """The conditional score model: encoder, potential head, the encoder's sigma FiLM, and the
    standardization `mean`, `std` (B,) of the whitened spectra; no encoder or FiLM with
    `context` 0 (DART)."""

    def __init__(self, n_bands: int, spec: dict) -> None:
        """spec: the recipe's `model` block (configs/darts.json)."""
        super().__init__()
        self.spec = spec
        self.encoder = BlindContext(n_bands, spec) if spec["context"] else None
        self.head = PotentialHead(n_bands, spec)
        if spec["context"]:
            self.film = torch.nn.Linear(spec["embed_dim"], 2 * (spec["channels"] - 1))
            torch.nn.init.zeros_(self.film.weight)
            torch.nn.init.zeros_(self.film.bias)
        self.register_buffer("mean", torch.zeros(n_bands))
        self.register_buffer("std", torch.ones(n_bands))

    def image(self, fx: torch.Tensor, usable: torch.Tensor,
              grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """(standardized image z (1, B, W, H), mask (1, 1, W, H)) of whitened spectra fx (P, B)
        in pixel order and the pixels contexts may read, usable (P,); on the model's device."""
        W, H = grid
        z = (fx.float().to(self.mean.device) - self.mean) / self.std
        mask = usable.to(z.device).reshape(W, H)[None, None].float()
        return z.reshape(W, H, -1).permute(2, 0, 1)[None], mask

    def contexts(self, image: torch.Tensor, mask: torch.Tensor,
                 log_sigma: torch.Tensor) -> torch.Tensor:
        """c (N, C, W, H) of the clean images a z (N, B, W, H) the contexts read, mask (N, 1, W, H)
        and log sigma (N,); (N, 0, W, H) without an encoder."""
        if self.encoder is None:
            return image.new_zeros(image.shape[0], 0, *image.shape[2:])
        gamma, beta = self.film(self.head.embed(log_sigma))[:, :, None, None].chunk(2, dim=1)
        return self.encoder(image, mask, gamma, beta)

    def epsilon(self, u: torch.Tensor, c: torch.Tensor, sigma: float | torch.Tensor,
                create_graph: bool = False) -> torch.Tensor:
        """eps_hat (n, B) of eq. (cond-eps) at the inputs u = a y (n, B) with contexts c (n, C),
        each context held fixed. sigma: one level for every row (a float, whose log and a, s are
        taken in double precision) or one per row, (n,). create_graph: keep the graph, for a
        loss on eps_hat."""
        if isinstance(sigma, float):
            log_sigma, (a, s) = torch.full((len(u),), math.log(sigma), device=u.device), vp(sigma)
        else:
            log_sigma, (a, s) = sigma.log(), vp(sigma[:, None])
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.head(u, c, log_sigma).sum(), u,
                                          create_graph=create_graph)
        return a * grad + s * u.detach()

    def save(self, path: Path, **record: object) -> None:
        """The recipe's `model` block, the weights on the CPU, and `record` (e.g. the step)."""
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save({"spec": self.spec, "n_bands": len(self.mean), "state": state} | record, path)

    @classmethod
    def load(cls, path: Path) -> "DARTS":
        """A saved model, on the CPU, in evaluation mode; what else was saved with it (the training
        record) in `record`."""
        saved = torch.load(path, weights_only=True, map_location="cpu")
        model = cls(saved["n_bands"], saved["spec"])
        model.load_state_dict(saved["state"])
        model.record = {k: v for k, v in saved.items() if k not in ("spec", "n_bands", "state")}
        return model.eval()
