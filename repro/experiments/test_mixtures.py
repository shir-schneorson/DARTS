"""The mixtures, the calibration test and the synthetic scene's oracles, on closed-form cases."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from experiments import calibration
from experiments import spatial_scene as sp
from experiments.gmm import GMM
from experiments.student_mixture import StudentMixture


def gaussian_data(n: int = 4000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mixing = torch.randn(4, 4, generator=g, dtype=torch.float64)
    return torch.randn(n, 4, generator=g, dtype=torch.float64) @ mixing.T + 3, mixing


def test_gaussian_fit_is_the_sample_gaussian() -> None:
    x, _ = gaussian_data()
    model = GMM.fit(x, 1, 0, 0.0)
    cov = torch.cov(x.T, correction=0)
    exact = torch.distributions.MultivariateNormal(x.mean(0), covariance_matrix=cov)
    assert torch.allclose(model.log_prob(x), exact.log_prob(x), atol=1e-8)
    a = torch.randn(3, 4, dtype=torch.float64)
    lower, upper = model.marginal_cdf(x[:5], a)
    sd = (a @ cov @ a.T).diagonal().sqrt()
    expected = stats.norm.cdf(((x[:5] - x.mean(0)) @ a.T / sd).numpy())
    np.testing.assert_allclose(lower.numpy(), expected, rtol=1e-9)
    np.testing.assert_allclose((lower + upper).numpy(), 1.0, rtol=1e-12)


def test_gmm_separates_two_clusters() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.cat([torch.randn(3000, 2, generator=g, dtype=torch.float64) - 6,
                   torch.randn(1000, 2, generator=g, dtype=torch.float64) + 6])
    model = GMM.fit(x, 2, 0, 1e-6)
    assert sorted(model.weights.tolist()) == pytest.approx([0.25, 0.75], abs=1e-3)


def test_student_log_density_and_marginal() -> None:
    mean, scale = torch.tensor([[1.0, -2.0]], dtype=torch.float64), torch.tensor(
        [[[2.0, 0.3], [0.3, 0.5]]], dtype=torch.float64)
    model = StudentMixture.from_covariances(torch.ones(1, dtype=torch.float64), mean, scale, 5.0)
    x = torch.tensor([[0.0, 0.0], [3.0, -1.0]], dtype=torch.float64)
    expected = stats.multivariate_t(mean[0].numpy(), scale[0].numpy(), df=5.0).logpdf(x.numpy())
    np.testing.assert_allclose(model.log_prob(x).numpy(), expected, rtol=1e-12)
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    lower, _ = model.marginal_cdf(x, a)
    spread = math.sqrt(float(a @ scale[0] @ a.T))
    np.testing.assert_allclose(lower[:, 0].numpy(),
                               stats.t.cdf(((x - mean) @ a.T)[:, 0].numpy() / spread, 5.0))


def test_fits_increase_the_likelihood_and_report_convergence() -> None:
    x, _ = gaussian_data(2000, 3)
    gmm = GMM.fit(x, 3, 0, 1e-6)
    t = StudentMixture.fit(x, 2, 0, 1e-6, torch.tensor([3.0, 10.0, 100.0], dtype=torch.float64))
    for model in (gmm, t):
        assert np.all(np.diff(model.history) > -1e-9) and model.converged


def test_two_fold_record_on_gaussian_data() -> None:
    from experiments import experiment1
    x, _ = gaussian_data(2000, 4)
    config = {"split_seed": 0, "direction_seed": 0, "n_directions": 20, "tails": [0.01],
              "level": 0.01, "fit_seed": 0, "reg_covar": 1e-6}
    result, _ = experiment1.two_fold(x, torch.ones(4, dtype=torch.float64),
                                     experiment1.fitter("gmm", 1, config), config)
    line = experiment1.record(result, len(x), config)
    assert line["converged"] == "yes/yes" and line["dof"] == ""
    assert 0.5 < line["upper_t_0.01"] < 2 and line["upper_rejected_0.01"] < 0.2


def test_binomial_test() -> None:
    assert not calibration.binomial_rejected(np.array([10]), 1000, 0.01, 0.01)[0]
    assert calibration.binomial_rejected(np.array([40, 0]), 1000, 0.01, 0.01).all()


def test_target_projection_is_the_whitened_direction() -> None:
    x, _ = gaussian_data()
    t = torch.tensor([1.0, 0.5, -1.0, 2.0], dtype=torch.float64)
    unmix = torch.linalg.inv(torch.linalg.cholesky(torch.cov(x.T, correction=0)))
    a = calibration.projections(t, unmix, 5, 0)
    z_direction = unmix @ t / (unmix @ t).norm()
    assert torch.allclose(a[0], z_direction @ unmix) and a.shape == (6, 4)


def test_oracles_are_the_gaussian_tests_for_large_dof() -> None:
    """With nu large the scene density is Gaussian: the GLRT is max(0, u)^2 of the AMF u, and
    the context-given Rao statistic with w = 1 is u itself."""
    mean = torch.zeros(1, 3, dtype=torch.float64)
    cov = torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64))[None]
    oracle = StudentMixture.from_covariances(torch.ones(1, dtype=torch.float64), mean, cov, 1e7)
    x = torch.randn(400, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    t = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    u = x @ torch.linalg.solve(cov[0], t) / torch.sqrt(t @ torch.linalg.solve(cov[0], t))
    scene = {"oracle": oracle, "component": torch.zeros(400, dtype=torch.long),
             "scale": torch.ones(400, dtype=torch.float64)}
    assert torch.allclose(sp.oracle_context(scene, x, t), u, atol=1e-9)
    rao = sp.oracle_rao(oracle, x, t)
    assert torch.allclose(rao / rao.square().mean().sqrt(), u / u.square().mean().sqrt(),
                          atol=1e-4)
    glrt = sp.oracle_glrt(oracle, x, t, 200.0, 20001)
    np.testing.assert_allclose(glrt.numpy(), torch.clamp(u, min=0).square().numpy(), atol=1e-4)
