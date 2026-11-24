"""The statistics against closed forms: with psi = 0 the model is the affine score of the
whitened Gaussian, so its Rao statistic is the Ledoit-Wolf AMF and its GLRT the squared one-sided
AMF, both times a^2."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf

from dartsdet import whitening
from dartsdet.data import Scene
from dartsdet.model import DARTS, pixels, vp
from dartsdet.statistics import contexts, glrt, log_ratio, rao, rao_gradient, refined_maximum

SPEC = json.loads((Path(__file__).parents[1] / "configs" / "darts.json").read_text())["model"]
SIGMA = 0.4


def gaussian_scene() -> tuple[Scene, np.ndarray]:
    """A correlated Gaussian scene with an additive signature, and its Ledoit-Wolf AMF (P,)."""
    rng = np.random.default_rng(0)
    H, W, B = 30, 40, 6
    spectra = rng.standard_normal((H * W, B)) @ rng.standard_normal((B, B)) + 5
    d = rng.standard_normal(B)
    covariance = ledoit_wolf(spectra)[0]
    w = np.linalg.solve(covariance, d)
    lw_amf = (spectra - spectra.mean(0)) @ w / np.sqrt(d @ w)
    cube = spectra.reshape(H, W, B, order="F")
    return Scene(cube, np.zeros(H * W), np.ones(H * W, dtype=bool), d, additive=True), lw_amf


def untrained_inputs() -> tuple:
    scene, lw_amf = gaussian_scene()
    inputs = whitening.prepare(scene, 0.1)
    model = DARTS(len(scene.target), SPEC)  # psi = 0, mean 0, std 1
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    return model, pixels(z), contexts(model, z, mask, SIGMA), inputs.t, lw_amf


def test_rao_of_the_affine_score_is_the_ledoit_wolf_amf() -> None:
    model, y, c, t, lw_amf = untrained_inputs()
    a, _ = vp(SIGMA)
    statistic = rao(model, y, c, t, SIGMA)
    np.testing.assert_allclose(statistic.numpy(), a**2 * lw_amf, rtol=1e-4, atol=1e-5)


def test_glrt_of_the_affine_score_is_the_squared_one_sided_amf() -> None:
    """2 max_theta a^2 (theta t'x - theta^2 |t|^2 / 2) = a^2 max(0, t_hat'x)^2, at theta* =
    t'x / |t|^2; checked where theta* lies inside the amplitude grid, whose spacing bounds the
    error by about 0.4%."""
    model, y, c, t, lw_amf = untrained_inputs()
    a, _ = vp(SIGMA)
    statistic = glrt(model, y, c, t, SIGMA, torch.logspace(-3, 1, 81))
    expected = a**2 * np.maximum(lw_amf, 0) ** 2
    theta_star = lw_amf / float(t.norm())
    inside = (theta_star > 2e-3) & (theta_star < 9)
    np.testing.assert_allclose(statistic.numpy()[inside], expected[inside], rtol=5e-3)
    assert (statistic.numpy()[lw_amf <= 0] == 0).all() and inside.sum() > 400


def test_log_ratio_of_the_affine_score_is_quadratic() -> None:
    """l(d) = a^2 (d t_hat'x - d^2 / 2) at whitened distance d along -t_hat."""
    model, y, c, t, lw_amf = untrained_inputs()
    a, _ = vp(SIGMA)
    distances = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    ell = log_ratio(model, y, c, t, SIGMA, distances).numpy()
    d = np.r_[0.0, distances.numpy()]
    expected = a**2 * (lw_amf[:, None] * d - d**2 / 2)
    np.testing.assert_allclose(ell, expected, rtol=1e-4, atol=1e-4)


def test_refined_maximum() -> None:
    grid = torch.arange(10, dtype=torch.float64)
    parabola = -((grid - 4.3) ** 2)
    rising, falling = grid.clone(), -grid
    out = refined_maximum(torch.stack([parabola, rising, falling]))
    torch.testing.assert_close(out, torch.tensor([0.0, 9.0, 0.0], dtype=torch.float64))


def test_rao_gradient_of_the_affine_score_is_constant() -> None:
    """With psi = 0, eps_hat(a y) = s a y, so the gradient of eps_hat' v / sigma is
    s a v / sigma."""
    model, y, c, t, _ = untrained_inputs()
    a, s = vp(SIGMA)
    v = torch.randn(len(t), generator=torch.Generator().manual_seed(0))
    g = rao_gradient(model, y[:300], c[:300], v, SIGMA)
    torch.testing.assert_close(g, (s * a * v / SIGMA).expand(300, -1), rtol=1e-5, atol=1e-5)


def test_rao_gradient_matches_finite_differences() -> None:
    """A model with a nonzero potential: the directional derivative of `rao` along a step equals
    the gradient's projection on it."""
    model, y, c, t, _ = untrained_inputs()
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.normal_(0, 0.3)
    y, c = y[:200].double(), c[:200].double()
    model = model.double()
    v = (t / t.norm()) / model.std
    step = 1e-6 * torch.randn(y.shape, generator=torch.Generator().manual_seed(1), dtype=y.dtype)
    change = rao(model, y + step, c, t, SIGMA) - rao(model, y - step, c, t, SIGMA)
    g = rao_gradient(model, y, c, v, SIGMA)
    torch.testing.assert_close(change / 2, (g * step).sum(1), rtol=1e-4, atol=1e-9)
