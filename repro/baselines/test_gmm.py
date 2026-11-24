"""The GMM detectors in the one-component case, where they are the Gaussian ones."""

import numpy as np
import torch

from baselines import gmm


def one_component() -> tuple:
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2000, 4)) @ rng.standard_normal((4, 4)) + 3)
    t = torch.from_numpy(rng.standard_normal(4))
    model = gmm.fit(x, 1, 0, 1e-6)
    return model, x, t, gmm.line(model, x, t)


def test_one_component_fit_is_the_gaussian_of_the_data() -> None:
    model, x, _, _ = one_component()
    assert model.converged
    z = (x - model.shift) @ model.unmix.T
    torch.testing.assert_close(model.means[0], z.mean(0), atol=1e-12, rtol=0)
    covariance = torch.cov(z.T, correction=0) + 1e-6 * torch.eye(4, dtype=z.dtype)
    torch.testing.assert_close(model.covariances[0], covariance)


def test_one_component_rao_and_glrt_are_the_amf_and_its_square() -> None:
    """Rao = t' S^-1 (x - m); GLRT = (t' S^-1 (x - m))^2 / (t' S^-1 t) where the maximizing
    amplitude lies inside the grid, 0 where it is negative."""
    model, x, t, terms = one_component()
    weights = model.weights.double().log()
    b, c = terms.b[:, 0], terms.c[0]
    torch.testing.assert_close(gmm.rao(weights, terms), b)
    statistic = gmm.glrt(weights, terms)
    theta = b / c
    inside = (theta > 0.1) & (theta < 99)
    torch.testing.assert_close(statistic[inside], (b**2 / c)[inside], rtol=1e-9, atol=1e-9)
    assert (statistic[theta < 0] == 0).all() and inside.sum() > 100
