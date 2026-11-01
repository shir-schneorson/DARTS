"""Experiment 5's background, whose smoothed score is known exactly.

B = 10 independent standardized sources, 5 bimodal (density proportional to
exp(-(w^2 - 1)^2 / 0.5)) and 5 Gumbel, rotated by a random orthogonal Q, so that x = Q z is white,
as after whitening. Isotropic smoothing commutes with Q, so the smoothed score is
s_sigma(x) = Q psi_sigma(Q' x), with psi_sigma = (log f_i * phi_sigma)' per source, computed on a
fine grid. Along a unit signature t_hat, with q = Q' t_hat:

    A = -q' psi_sigma(z),   C = -sum_i psi_sigma'(z_i) q_i^2,
    l(theta) = sum_i [log g_i(z_i - theta q_i) - log g_i(z_i)],   g_i = f_i * phi_sigma.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

B = 10
GRID = np.linspace(-14, 14, 14001)
STEP = GRID[1] - GRID[0]


def sources() -> list[np.ndarray]:
    """The B source densities on `GRID`, each with zero mean and unit variance to the grid's
    accuracy (about 1e-4 after the regridding)."""
    raw = [np.exp(-((GRID**2 - 1) ** 2) / 0.5)] * 5 + [np.exp(-(GRID + np.exp(-GRID)))] * 5
    out = []
    for f in raw:
        f = f / (f.sum() * STEP)
        mean = (GRID * f).sum() * STEP
        sd = math.sqrt(((GRID - mean) ** 2 * f).sum() * STEP)
        g = np.interp(GRID, (GRID - mean) / sd, f * sd)
        out.append(g / (g.sum() * STEP))
    return out


def smoothed_log(f: np.ndarray, sigma: float) -> np.ndarray:
    """log(f * phi_sigma) on `GRID`, floored at 1e-300 before the log; f itself at sigma = 0."""
    g = gaussian_filter1d(f, sigma / STEP, mode="constant") if sigma > 0 else f
    return np.log(np.maximum(g, 1e-300))


def scores(fs: list[np.ndarray], sigma: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per source, psi_sigma and psi_sigma' on `GRID`."""
    psi = [np.gradient(smoothed_log(f, sigma), STEP) for f in fs]
    return psi, [np.gradient(p, STEP) for p in psi]


def sample(fs: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """n draws of the sources (n, B), by inverting each one's distribution function on the grid."""
    columns = []
    for f in fs:
        cdf = np.cumsum(f) * STEP
        columns.append(np.interp(rng.random(n), cdf / cdf[-1], GRID))
    return np.stack(columns, 1)


def exact_reads(z: np.ndarray, q: np.ndarray, psi: list,
                dpsi: list) -> tuple[np.ndarray, np.ndarray]:
    """(A (n,), C (n,)) at source coordinates z (n, B); q = Q' t_hat (B,)."""
    s = np.stack([np.interp(z[:, i], GRID, psi[i]) for i in range(B)], 1)
    d = np.stack([np.interp(z[:, i], GRID, dpsi[i]) for i in range(B)], 1)
    return -(s @ q), -(d @ q**2)


def exact_log_ratio(y: np.ndarray, q: np.ndarray, fs: list[np.ndarray], sigma: float,
                    thetas: tuple[float, ...]) -> np.ndarray:
    """l(theta) (n, len(thetas)) at source coordinates y (n, B)."""
    logs = [smoothed_log(f, sigma) for f in fs]
    base = sum(np.interp(y[:, i], GRID, logs[i]) for i in range(B))
    return np.stack([sum(np.interp(y[:, i] - theta * q[i], GRID, logs[i]) for i in range(B)) - base
                     for theta in thetas], 1)
