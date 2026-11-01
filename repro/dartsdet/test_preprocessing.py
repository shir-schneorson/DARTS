"""Whitening, the signature and the screen on synthetic scenes."""

import numpy as np
import pytest
import torch
from sklearn.covariance import ledoit_wolf

from dartsdet import preprocessing, scenes
from dartsdet.scenes import IGNORED


def synthetic_scene(additive: bool) -> scenes.Scene:
    rng = np.random.default_rng(0)
    H, W, B = 30, 40, 6
    mixing = rng.standard_normal((B, B))
    cube = rng.standard_normal((H, W, B)) @ mixing.T + 10
    d = rng.standard_normal(B) * 3
    labels = np.zeros(H * W)
    labels[rng.choice(H * W, 12, replace=False)] = 1
    flat = cube.reshape(-1, B, order="F")
    flat[labels == 1] += d if additive else 0
    cube = flat.reshape(H, W, B, order="F")
    exists = np.ones(H * W, dtype=bool)
    exists[5] = False
    labels[5] = IGNORED
    return scenes.Scene(cube, labels, exists, d, additive)


def test_whitening_inverts_the_shrunk_covariance() -> None:
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((400, 5)) @ np.diag([1, 2, 3, 4,
                                                                                      50.0]))
    mean, matrix = preprocessing.whitening(x)
    shrunk = torch.from_numpy(ledoit_wolf(x.numpy())[0])
    assert torch.allclose(matrix, matrix.T)
    assert torch.allclose(matrix @ shrunk @ matrix, torch.eye(5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(mean, x.mean(0))


def test_screen_is_the_top_amf() -> None:
    rng = np.random.default_rng(2)
    spectra = rng.standard_normal((1000, 4))
    direction = np.array([1.0, 0, 0, 0])
    spectra[:20] += 6 * direction
    out = preprocessing.screened(spectra, direction)
    assert out.sum() == 100 and out[:20].all()


def test_prepare_additive_signature_and_layout() -> None:
    scene = synthetic_scene(additive=True)
    inputs = preprocessing.prepare(scene)
    mean, matrix = preprocessing.whitening(torch.from_numpy(scene.spectra()[scene.exists]))
    assert torch.allclose(inputs.t, torch.from_numpy(scene.target) @ matrix)
    assert inputs.grid == (40, 30) and inputs.fx.shape == (1200, 6)
    assert not inputs.fx[5].any() and not inputs.usable[5]
    assert int(inputs.usable.sum()) == 1199 - round(0.1 * 1199)


def test_prepare_real_target_signature() -> None:
    scene = synthetic_scene(additive=False)
    inputs = preprocessing.prepare(scene)
    mean, matrix = preprocessing.whitening(torch.from_numpy(scene.spectra()[scene.exists]))
    assert torch.allclose(inputs.t, (torch.from_numpy(scene.target) - mean) @ matrix)


def test_amf_is_standardized_under_its_own_gaussian() -> None:
    rng = np.random.default_rng(4)
    spectra = rng.standard_normal((2000, 5)) @ rng.standard_normal((5, 5)) + 3
    direction = rng.standard_normal(5)
    u = preprocessing.amf(spectra, direction)
    assert abs(u.mean()) < 1e-12 and u.var() == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(preprocessing.amf(spectra, 7 * direction), u, rtol=1e-12)


def test_prepare_with_a_fitted_whitening() -> None:
    scene = synthetic_scene(additive=True)
    other = torch.from_numpy(np.random.default_rng(5).standard_normal((300, 6)) * 2)
    mean, matrix = preprocessing.whitening(other)
    inputs = preprocessing.prepare(scene, (mean, matrix))
    x = torch.from_numpy(scene.spectra()[scene.exists])
    assert torch.allclose(inputs.fx[torch.from_numpy(scene.exists)], (x - mean) @ matrix)
    assert torch.allclose(inputs.t, torch.from_numpy(scene.target) @ matrix)
