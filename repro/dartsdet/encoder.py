"""The blind-spot context encoder: features of each pixel's surroundings, never of the pixel.

The input is an image of standardized spectra and a mask of the pixels a context may use; masked
pixels and the outside of the image are zeros. A pointwise stem maps each pixel to features (times
the mask, with the mask appended). One half-plane network sees, at each pixel, only rows at least
`shift` above it: residual (2 x 3)-kernel convolutions padded on top, so that each output row reads
rows at or above itself, then the output moved down by `shift` rows. Applied to the four rotations
of the stem output and rotated back, the four receptive fields cover the plane minus the
(2 shift - 1)^2 square around the pixel. Normalization is per pixel over channels, never over space,
so no statistic leaks the pixel into its own context.
"""

import torch
import torch.nn.functional as F


class PixelNorm(torch.nn.Module):
    """Layer normalization over the channels at each pixel, (N, C, H, W) -> (N, C, H, W).

    Written out rather than `F.layer_norm`, whose arithmetic and parameter shapes differ from the
    trained checkpoints'.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        centred = h - h.mean(1, keepdim=True)
        scale = torch.rsqrt(centred.pow(2).mean(1, keepdim=True) + 1e-5)
        return centred * scale * self.weight + self.bias


class HalfPlane(torch.nn.Module):
    """Features at each pixel from the rows at least `shift` above it, (N, C, H, W) -> same."""

    def __init__(self, channels: int, dilations: list[int], shift: int, activation: str) -> None:
        super().__init__()
        self.dilations, self.shift = list(dilations), shift
        self.norms = torch.nn.ModuleList(PixelNorm(channels) for _ in dilations)
        self.convs = torch.nn.ModuleList(torch.nn.Conv2d(channels, channels, (2, 3), dilation=d)
                                         for d in dilations)
        self.act = getattr(torch.nn, activation)()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for norm, conv, d in zip(self.norms, self.convs, self.dilations, strict=True):
            h = h + conv(F.pad(self.act(norm(h)), (d, d, d, 0)))
        return F.pad(h, (0, 0, self.shift, 0))[:, :, : h.shape[2]]


class BlindContext(torch.nn.Module):
    """(image (N, B, H, W), mask (N, 1, H, W)) -> context (N, C, H, W), blind to each pixel's
    (2 shift - 1)^2 square."""

    def __init__(self, n_bands: int, spec: dict) -> None:
        """spec: `channels`, `dilations`, `shift`, `context` (C) and `activation`."""
        super().__init__()
        channels, act = spec["channels"], getattr(torch.nn, spec["activation"])
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(n_bands, channels, 1), act(),
                                        torch.nn.Conv2d(channels, channels - 1, 1))
        self.branch = HalfPlane(channels, spec["dilations"], spec["shift"], spec["activation"])
        self.norm = PixelNorm(4 * channels)
        self.out = torch.nn.Sequential(torch.nn.Conv2d(4 * channels, channels, 1), act(),
                                       torch.nn.Conv2d(channels, spec["context"], 1))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.stem(z) * mask, mask], dim=1)
        if h.shape[2] == h.shape[3]:  # a square image's rotations share a shape: one batched pass
            views = self.branch(torch.cat([torch.rot90(h, k, (2, 3)) for k in range(4)])).chunk(4)
            views = [torch.rot90(v, -k, (2, 3)) for k, v in enumerate(views)]
        else:
            views = [torch.rot90(self.branch(torch.rot90(h, k, (2, 3))), -k, (2, 3))
                     for k in range(4)]
        return self.out(self.norm(torch.cat(views, dim=1)))

    def margin(self) -> int:
        """Pixels of context a crop needs around it: the shift plus the dilated kernels' reach."""
        return self.branch.shift + sum(self.branch.dilations)

