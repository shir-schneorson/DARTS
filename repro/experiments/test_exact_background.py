"""Experiment 5's exact background: standardized sources, and reads that agree with each other."""

import numpy as np

from experiments import exact_background as eb


def test_sources_are_standardized_densities() -> None:
    for f in eb.sources():
        mean = (eb.GRID * f).sum() * eb.STEP
        assert abs(f.sum() * eb.STEP - 1) < 1e-9 and abs(mean) < 1e-4
        assert abs(((eb.GRID - mean) ** 2 * f).sum() * eb.STEP - 1) < 1e-3


def test_reads_agree_with_each_other() -> None:
    """A is the slope of l at 0, and C = -dA/du, each against a central difference over ten grid
    steps."""
    fs, rng = eb.sources(), np.random.default_rng(0)
    z, q = eb.sample(fs, 50, rng), rng.standard_normal(eb.B)
    q /= np.linalg.norm(q)
    psi, dpsi = eb.scores(fs, 0.3)
    A, C = eb.exact_reads(z, q, psi, dpsi)
    h = 10 * eb.STEP
    ell = eb.exact_log_ratio(z, q, fs, 0.3, (-h, h))
    np.testing.assert_allclose((ell[:, 1] - ell[:, 0]) / (2 * h), A, rtol=1e-3, atol=1e-3)
    ahead, behind = (eb.exact_reads(z - u * q, q, psi, dpsi)[0] for u in (h, -h))
    np.testing.assert_allclose(-(ahead - behind) / (2 * h), C, rtol=1e-3, atol=1e-3)
