"""From a scene to the model's inputs: whitened spectra, the signature, and the screen (Sections
V-C, V-D and VI-B).

Whitening is symmetric (ZCA) with the Ledoit-Wolf covariance C of every existing pixel, its
eigenvalues floored at `FLOOR` times the largest: x -> W (x - mu), W = C^(-1/2). An additive target
keeps its form, with signature W d; a real target, which replaces the background, moves a pixel
along W (d - mu) on average.

The screen leaves out of training the existing pixels with the highest one-sided adaptive matched
filter (AMF) statistic t' S^-1 (x - m), a given share of them, under the maximum-likelihood
Gaussian (m, S) of every existing pixel: the pixels most likely to be targets.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf

from dartsdet.data import Scene

FLOOR = 1e-10


@dataclass(frozen=True)
class Inputs:
    """fx: (P, B) float64 whitened spectra in pixel order, zeros where a pixel does not exist.
    usable: (P,) bool, the existing pixels the screen keeps. t: (B,) float64, the signature in
    whitened units. grid: (W, H); the Fortran-ordered pixel vectors reshape to the (W, H) image
    the encoder reads."""

    fx: torch.Tensor
    usable: torch.Tensor
    t: torch.Tensor
    grid: tuple[int, int]


def whitening(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean (B,), symmetric W (B, B)) of spectra x (n, B) float64."""
    shrunk, _ = ledoit_wolf(x.numpy())
    values, vectors = torch.linalg.eigh(torch.from_numpy(shrunk))
    floor = FLOOR * values.max()
    return x.mean(dim=0), vectors @ torch.diag(values.clamp_min(floor).rsqrt()) @ vectors.T


def amf(spectra: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """One-sided AMF t' S^-1 (x - m) / sqrt(t' S^-1 t) (n,) of spectra (n, B) along t = direction
    (B,), under the maximum-likelihood Gaussian (m, S) of the spectra themselves."""
    w = np.linalg.solve(np.cov(spectra, rowvar=False, ddof=0), direction)
    return (spectra - spectra.mean(axis=0)) @ w / np.sqrt(direction @ w)


def screened(spectra: np.ndarray, direction: np.ndarray, share: float) -> np.ndarray:
    """(n,) bool: the round(share n) spectra (n, B) with the highest `amf` along `direction`, the
    raw-units direction a target moves a pixel."""
    out = np.zeros(len(spectra), dtype=bool)
    out[np.argsort(-amf(spectra, direction), kind="stable")[:round(share * len(spectra))]] = True
    return out


def prepare(scene: Scene, screen: float,
            fitted: tuple[torch.Tensor, torch.Tensor] | None = None) -> Inputs:
    """The scene's whitened spectra, the signature, and the pixels left after screening the share
    `screen`; whitened by `fitted`, a (mean, W) pair of `whitening`, instead of the scene's own
    when given."""
    spectra = scene.spectra()[scene.exists]
    direction = scene.target if scene.additive else scene.target - spectra.mean(axis=0)
    x = torch.from_numpy(spectra)
    mean, matrix = whitening(x) if fitted is None else fitted
    t = torch.from_numpy(direction if scene.additive else scene.target - mean.numpy()) @ matrix
    exists = torch.from_numpy(scene.exists)
    fx = torch.zeros(len(scene.exists), spectra.shape[1], dtype=torch.float64)
    fx[exists] = (x - mean) @ matrix
    usable = torch.zeros(len(scene.exists), dtype=torch.bool)
    usable[exists] = torch.from_numpy(~screened(spectra, direction, screen))
    H, W = scene.cube.shape[:2]
    return Inputs(fx, usable, t, (W, H))
