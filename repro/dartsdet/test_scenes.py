"""The implant protocol and the pixel order."""

import numpy as np

from dartsdet import scenes
from dartsdet.scenes import IGNORED


def test_implant_count() -> None:
    assert scenes.implant_count(10_100, 0, 100) == 100
    assert scenes.implant_count(10_100, 101, 100) == 99


def test_implant_sites_keep_their_distance() -> None:
    allowed = np.ones((40, 40), dtype=bool)
    allowed[:10] = False
    sites = scenes.implant_sites(allowed, 60, 3, np.random.default_rng(0))
    assert len(sites) == 60 and allowed[sites[:, 0], sites[:, 1]].all()
    gaps = np.abs(sites[:, None] - sites[None]).max(axis=2) + 1000 * np.eye(60, dtype=int)
    assert gaps.min() >= 3


def test_implanted_adds_alpha_d_at_the_labelled_sites_only() -> None:
    rng = np.random.default_rng(0)
    cube = rng.integers(0, 100, (30, 20, 4)).astype(np.uint16)
    real, gone = np.zeros((30, 20), dtype=bool), np.zeros((30, 20), dtype=bool)
    real[:5, :5], gone[-1, -1] = True, True
    d = np.array([1.0, 2.0, 3.0, 4.0])
    scene = scenes.implanted(cube, d, real, gone, 0.5)
    implant = scene.labels == 1
    assert implant.sum() == round((600 - 26) / 101)
    difference = scene.spectra() - cube.reshape(-1, 4, order="F")
    np.testing.assert_allclose(difference[implant], np.tile(0.5 * d, (implant.sum(), 1)))
    assert not difference[~implant].any()
    assert (scene.labels[scenes.fortran(real | gone)] == IGNORED).all()
    assert not scene.exists[scenes.fortran(gone)].any() and scene.additive


def test_fortran_order() -> None:
    grid = np.arange(6).reshape(2, 3)          # H = 2, W = 3
    p = scenes.fortran(grid)
    H = 2
    assert all(p[i] == grid[i % H, i // H] for i in range(6))


def test_without_targets_keeps_everything_but_the_implants() -> None:
    cube = np.zeros((30, 20, 2))
    real, gone = np.zeros((30, 20), dtype=bool), np.zeros((30, 20), dtype=bool)
    real[:4, :4] = True
    scene = scenes.implanted(cube, np.ones(2), real, gone, 0.0)
    clean = scenes.without_targets(scene)
    assert not (clean.labels == 1).any() and (clean.labels[scene.labels == 1] == 0).all()
    assert (clean.labels[scene.labels == IGNORED] == IGNORED).all()
    assert clean.cube is scene.cube and clean.target is scene.target
