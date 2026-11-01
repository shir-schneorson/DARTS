"""The noise-level rule: one sigma per scene, chosen without labels from three reads of the folds.

sigma_hat = max(sigma_LR, sigma_cvx, sigma_pl):

- sigma_LR, the likelihood-ratio bound (`lr_sigma`). A pixel y drawn from the smoothed background
  given its context has E[exp l(theta)] = 1 under the exact model, so by Markov's inequality the
  mixture likelihood ratio mean_theta exp l(theta) reaches 1/alpha with probability at most alpha.
  sigma_LR is where noised held-out pixels come to obey this bound.
- sigma_cvx, convexity (`convex_sigma`): from where -log p_sigma is convex along t at all but a
  share PF of the clean held-out pixels (curvature C > 0).
- sigma_pl, the screened plateau (`plateau_end`): where the likelihood-ratio rate of the noised
  screened pixels, the region past the screen cut on which the test decides, leaves its plateau.

Every read is on the noise-level grid `grid(sigma_max)`; S is its length.
"""

import numpy as np
from scipy.special import logsumexp

from dartsdet.evaluation import PF

LR_THETAS = (0.25, 0.5, 1.0, 2.0, 4.0)   # the likelihood-ratio amplitudes, whitened units
LR_ALPHA = 1e-2                          # and its level


def grid(sigma_max: float) -> list[float]:
    """The noise levels the rule reads: 0.005 times 200^(k/12), up to sigma_max."""
    levels = [0.005 * 200 ** (k / 12) for k in range(40)]
    return [s for s in levels if s <= sigma_max * (1 + 1e-9)]


def first_crossing(m: np.ndarray, sigmas: np.ndarray) -> float:
    """Where a falling curve m (S,) first reaches 1: log m interpolated linearly in log sigma
    between the last level above 1 and the first at or below it. The first level if m starts at
    or below 1, the last if it never gets there; a drop to m = 0 gives the level before it."""
    with np.errstate(divide="ignore"):
        lm = np.log(m)
    ls = np.log(sigmas)
    below = np.flatnonzero(lm <= 0)
    if len(below) == 0:
        return float(sigmas[-1])
    j = below[0]
    if j == 0:
        return float(sigmas[0])
    w = lm[j - 1] / (lm[j - 1] - lm[j])
    return float(np.exp(ls[j - 1] + w * (ls[j] - ls[j - 1])))


def lr_rate(ell: np.ndarray) -> np.ndarray:
    """r(sigma) / alpha (S,): the share of pixels whose mixture likelihood ratio
    mean_theta exp l(theta) reaches 1 / LR_ALPHA, over LR_ALPHA. ell: (n, S, len(LR_THETAS))."""
    mixture = logsumexp(ell, axis=2) - np.log(ell.shape[2])
    return np.mean(mixture >= np.log(1 / LR_ALPHA), 0) / LR_ALPHA


def lr_sigma(ell: np.ndarray, sigmas: np.ndarray) -> float:
    """sigma_LR: the `first_crossing` of 1 by the `lr_rate` of noised held-out pixels."""
    return first_crossing(lr_rate(ell), sigmas)


def convex_sigma(C: np.ndarray, sigmas: np.ndarray, beta: float = PF) -> float:
    """sigma_cvx: the grid level after the last one at which more than a share beta of the clean
    held-out curvatures C (n, S) are at most 0; the first level if there is none, and the last
    level if the last level itself violates."""
    bad = np.flatnonzero(np.mean(C <= 0, 0) > beta)
    return float(sigmas[0] if len(bad) == 0 else sigmas[min(bad[-1] + 1, len(sigmas) - 1)])


def plateau_end(rate: np.ndarray, sigmas: np.ndarray) -> float:
    """sigma_pl: the breakpoint sigma_0 of the least-squares fit
    r / max r = P - a max(0, log sigma - log sigma_0)^2, a > 0, to a falling rate curve (S,) from
    the grid's second level (the first is the edge of the training range) through its steepest
    drop per log sigma, sigma_0 searched on 400 points in log sigma. A curve that is zero from the
    second level on has no plateau to protect: the first level. If no breakpoint gives a > 0
    (a curve that never falls), the second level."""
    if rate[1:].max() == 0:
        return float(sigmas[0])
    x, y = np.log(sigmas[1:]), rate[1:] / rate[1:].max()
    end = int(np.argmin(np.diff(y) / np.diff(x))) + 2
    x, y = x[:end], y[:end]
    best = (np.inf, x[0])
    for x0 in np.linspace(x[0], x[-1], 400):
        design = np.c_[np.ones_like(x), -np.maximum(0, x - x0) ** 2]
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = float(np.sum((design @ coef - y) ** 2))
        if coef[1] > 0 and sse < best[0]:
            best = (sse, x0)
    return float(np.exp(best[1]))


def noise_level(ell_held: np.ndarray, ell_screened: np.ndarray, C_held: np.ndarray,
                sigmas: np.ndarray) -> float:
    """sigma_hat from the folds' pooled reads on the grid `sigmas` (S,): log ratios of noised
    held-out and noised screened pixels, (n, S, len(LR_THETAS)) each, and curvatures of the clean
    held-out pixels, (n, S)."""
    return max(lr_sigma(ell_held, sigmas), convex_sigma(C_held, sigmas),
               plateau_end(lr_rate(ell_screened), sigmas))
