"""Held-out tail calibration of a fitted background density along 1-D projections.

If a background model is right, its 1-D marginal along any direction is right, so on pixels it was
not fitted to, the share of pixels in each tail of probability p is p. Directions are taken in the
training half's whitened frame z = (x - m) U', so that each weighs the background's spread
equally: the whitened target direction (the one the AMF projects on) and uniform random unit
directions. A whitened direction v is the projection (U'v)'(x - m), returned as U'v.
"""

import numpy as np
import torch
from scipy import stats


def projections(t: torch.Tensor, unmix: torch.Tensor, n_random: int, seed: int) -> torch.Tensor:
    """Projection vectors U'v (1 + n_random, B): row 0 the whitened target direction, then
    `n_random` uniform random whitened directions drawn with `seed`; unmix: U (B, B)."""
    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(n_random, len(t), generator=generator, dtype=unmix.dtype)
    rows = torch.cat([(unmix @ t.to(unmix.dtype))[None], random])
    return (rows / rows.norm(dim=1, keepdim=True)) @ unmix


def tail_counts(lower: torch.Tensor, upper: torch.Tensor,
                tails: list[float]) -> dict[tuple[str, float], np.ndarray]:
    """Per direction, the number of held-out pixels in each tail of probability p, from the
    tail probabilities P(a'X <= a'x) and P(a'X > a'x) (n, D)."""
    return {(side, p): (prob < p).sum(0).numpy()
            for p in tails for side, prob in (("lower", lower), ("upper", upper))}


def binomial_rejected(counts: np.ndarray, n: int, p: float, level: float) -> np.ndarray:
    """Exact two-sided binomial test of each count against Binomial(n, p) at `level`, the p-value
    twice the smaller one-sided tail; True where calibration is rejected."""
    below, above = stats.binom.cdf(counts, n, p), stats.binom.sf(counts - 1, n, p)
    return np.minimum(1.0, 2 * np.minimum(below, above)) < level
