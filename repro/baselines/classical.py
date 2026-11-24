"""The classical detectors of Table II: AMF, ACE, CEM, and the local AMF and ACE.

The global AMF and ACE use the maximum-likelihood Gaussian (m, S) of the scene's existing
whitened pixels, along the whitened signature t; ACE is the signed root of the AMF over the
pixel's Mahalanobis norm. CEM uses the raw spectra and the raw target spectrum d, with the
uncentred correlation R of every existing pixel: x' R^-1 d / (d' R^-1 d).

The local detectors replace (m, S) by each pixel's window: the square around it minus a
`guard` x `guard` square, truncated at the image edge, holding only the allowed (unscreened)
pixels. Its side is the smallest odd one whose window holds at least `need` allowed pixels, so a
screened patch widens the windows around it. The window's n spectra give a mean m_i and a
Ledoit-Wolf covariance S_i = (1 - rho) S + rho (tr S / B) I. Every quantity comes from the
window's n x n Gram matrix by the Woodbury identity, so no B x B matrix is formed per pixel.
With w = x_i - m_i: AMF_i = t' S_i^-1 w / sqrt(t' S_i^-1 t), ACE_i = AMF_i / sqrt(w' S_i^-1 w).
"""

from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from dartsdet.whitening import amf

SLOTS = 2**25  # entries of a batch's Gram matrices; memory only


def cem(spectra: np.ndarray, d: np.ndarray) -> np.ndarray:
    """CEM (n,) of raw spectra (n, B) for the target spectrum d (B,)."""
    w = np.linalg.solve(spectra.T @ spectra / len(spectra), d)
    return spectra @ w / (d @ w)


def ace(spectra: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One-sided ACE (n,) of spectra (n, B) along t (B,): the AMF over sqrt(y' S^-1 y),
    y = x - m."""
    root = np.linalg.cholesky(np.cov(spectra, rowvar=False, ddof=0))
    y = np.linalg.solve(root, (spectra - spectra.mean(0)).T)
    return amf(spectra, t) / np.linalg.norm(y, axis=0)


def window_indices(grid: tuple[int, int], side: int, guard: int,
                   pixels: torch.Tensor) -> torch.Tensor:
    """(n, side^2 - guard^2) indices of the windows of `pixels` (n,), -1 outside the image; in
    Fortran pixel order, index = column H + row of the (W, H) grid."""
    W, H = grid
    offset = torch.arange(side) - side // 2
    dc, dr = torch.meshgrid(offset, offset, indexing="ij")
    half = (guard - 1) // 2
    ring = (dc.abs() > half) | (dr.abs() > half)
    c, r = pixels[:, None] // H + dc[ring], pixels[:, None] % H + dr[ring]
    return torch.where((c >= 0) & (c < W) & (r >= 0) & (r < H), c * H + r, -1)


def window_batches(allowed: torch.Tensor, centres: torch.Tensor, grid: tuple[int, int],
                   guard: int, need: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """(pixels (n,), window indices (n, N)) batches covering the `centres` (P,) bool pixels, each
    window the smallest odd side holding at least `need` `allowed` (P,) pixels; one side per
    batch."""
    W, H = grid
    image = allowed.cpu().double().reshape(1, 1, W, H)
    inner = (F.avg_pool2d(image, guard, 1, guard // 2) * guard**2).round().flatten()
    sides, side = torch.zeros(W * H, dtype=torch.long), guard
    while (sides == 0).any():
        side += 2
        if side > 2 * max(W, H) + 1:
            raise ValueError(f"fewer than {need} allowed pixels around a pixel")
        count = (F.avg_pool2d(image, side, 1, side // 2) * side**2).round().flatten()
        sides[(sides == 0) & (count - inner >= need)] = side
    pixels = centres.cpu().nonzero().squeeze(1)
    for side in sides[pixels].unique().tolist():
        slots = side**2 - guard**2
        for batch in pixels[sides[pixels] == side].split(max(1, SLOTS // slots**2)):
            yield batch, window_indices(grid, side, guard, batch)


def ledoit_wolf_terms(gram: torch.Tensor, n: torch.Tensor,
                      bands: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(a, b) of the Ledoit-Wolf covariance a S + b I, (C,) each, from the centred windows' Gram
    matrices (C, N, N) of n (C,) spectra each (other rows zero)."""
    trace = gram.diagonal(dim1=1, dim2=2).sum(1) / n
    mu = trace / bands
    delta = (gram**2).sum((1, 2)) / n**2 - trace**2 / bands
    beta = ((gram.diagonal(dim1=1, dim2=2) ** 2).sum(1) - (gram**2).sum((1, 2)) / n) / n**2
    rho = torch.minimum(beta, delta) / delta
    return 1 - rho, rho * mu


def local_amf_ace(fx: torch.Tensor, allowed: torch.Tensor, t: torch.Tensor,
                  batches: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> tuple[np.ndarray, ...]:
    """(AMF, ACE) (P,) float64 of the pixels of `batches` (`window_batches`), zero elsewhere, from
    whitened spectra fx (P, B) and signature t (B,); windows hold the `allowed` (P,) pixels."""
    fx, t, allowed = fx.double().cpu(), t.double().cpu(), allowed.cpu()
    amf_map, ace_map = np.zeros(len(fx)), np.zeros(len(fx))
    for batch, index in batches:
        keep = (index >= 0) & allowed[index.clamp(min=0)]
        n = keep.sum(1).double()
        window = fx[index.clamp(min=0)] * keep[..., None]
        mean = window.sum(1) / n[:, None]
        y = (window - mean[:, None]) * keep[..., None]
        gram = y @ y.transpose(1, 2)
        a, b = ledoit_wolf_terms(gram, n, fx.shape[1])
        w = fx[batch] - mean
        yt, yw = y @ t, (y @ w[..., None]).squeeze(-1)
        eye = torch.eye(y.shape[1], dtype=y.dtype)
        system = (b * n)[:, None, None] * eye + a[:, None, None] * gram
        solved = torch.linalg.solve(system, torch.stack([yt, yw], -1))
        te, st, sw = t.expand_as(w), solved[..., 0], solved[..., 1]  # Woodbury: u' S_i^-1 v
        tt = ((te * te).sum(-1) - a * (yt * st).sum(-1)) / b
        tw = ((te * w).sum(-1) - a * (yt * sw).sum(-1)) / b
        ww = ((w * w).sum(-1) - a * (yw * sw).sum(-1)) / b
        amf_map[batch.numpy()] = (tw / tt.sqrt()).numpy()
        ace_map[batch.numpy()] = (tw / tt.sqrt() / ww.sqrt()).numpy()
    return amf_map, ace_map
