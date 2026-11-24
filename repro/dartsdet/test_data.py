"""The implant protocol, the MUUFL borders, and the pixel order."""

import numpy as np
import pytest

from dartsdet import data
from dartsdet.data import IGNORED


def test_implant_sites_keep_their_distance() -> None:
    allowed = np.ones((40, 40), dtype=bool)
    allowed[:10] = False
    sites = data.implant_sites(allowed, 60, 3, np.random.default_rng(0))
    assert len(sites) == 60 and allowed[sites[:, 0], sites[:, 1]].all()
    gaps = np.abs(sites[:, None] - sites[None]).max(axis=2) + 1000 * np.eye(60, dtype=int)
    assert gaps.min() >= 3


def test_implanted_adds_alpha_d_at_the_labelled_sites_only() -> None:
    rng = np.random.default_rng(0)
    cube = rng.integers(0, 100, (30, 20, 4)).astype(np.uint16)
    real, gone = np.zeros((30, 20), dtype=bool), np.zeros((30, 20), dtype=bool)
    real[:5, :5], gone[-1, -1] = True, True
    d = np.array([1.0, 2.0, 3.0, 4.0])
    scene = data.implanted(cube, d, real, gone, 0.5)
    implant = scene.labels == 1
    assert implant.sum() == round((600 - 26) / 101)  # one per 100 scored background pixels
    difference = scene.spectra() - cube.reshape(-1, 4, order="F")
    np.testing.assert_allclose(difference[implant], np.tile(0.5 * d, (implant.sum(), 1)))
    assert not difference[~implant].any()
    assert (scene.labels[(real | gone).ravel(order="F")] == IGNORED).all()
    assert not scene.exists[gone.ravel(order="F")].any() and scene.additive


def test_spectra_are_in_fortran_order() -> None:
    H, W = 2, 3
    cube = np.arange(H * W, dtype=np.float64).reshape(H, W, 1)
    scene = data.Scene(cube, np.zeros(6), np.ones(6, dtype=bool), np.ones(1), additive=True)
    p = np.arange(6)
    np.testing.assert_array_equal(scene.spectra()[:, 0], cube[p % H, p // H, 0])


def test_without_targets_keeps_everything_but_the_implants() -> None:
    real, gone = np.zeros((30, 20), dtype=bool), np.zeros((30, 20), dtype=bool)
    real[:4, :4] = True
    scene = data.implanted(np.zeros((30, 20, 2)), np.ones(2), real, gone, 0.0)
    clean = data.without_targets(scene)
    assert not (clean.labels == 1).any() and (clean.labels[scene.labels == 1] == 0).all()
    assert (clean.labels[scene.labels == IGNORED] == IGNORED).all()
    assert clean.cube is scene.cube and clean.target is scene.target


def test_halo_mask_sizes_and_clipping() -> None:
    mask = data.halo_mask((20, 20), np.array([0, 10]), np.array([0, 10]), np.array([1.0, 3.0]))
    assert mask[:3, :3].all() and mask.sum() == 9 + 49  # 5 x 5 clipped to 3 x 3, and 7 x 7
    tall = data.halo_mask((40, 40), np.array([20]), np.array([20]), np.array([6.0]))
    assert tall.sum() == 17 * 15 and tall.any(axis=1).sum() == 17


def test_unknown_cloth_size_is_refused() -> None:
    with pytest.raises(ValueError, match="no border"):
        data.halo_mask((10, 10), np.array([5]), np.array([5]), np.array([2.0]))
