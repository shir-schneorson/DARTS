"""The rule's stages: its grids, loudness against its closed form, and the mixed reference."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from dartsdet.metrics import PF
from dartsdet.model import DARTS, vp
from dartsdet.noise_level import (
    Rule,
    band_average,
    held_out_glrt,
    held_out_reads,
    loud,
    mix_response,
    reference_pool,
    select,
)
from dartsdet.test_statistics import SPEC, gaussian_scene
from dartsdet.whitening import prepare

RECIPE = json.loads((Path(__file__).parents[1] / "configs" / "darts.json").read_text())
RULE = Rule.from_config(RECIPE, (1 / 64, 4.0))


def test_grids() -> None:
    assert len(RULE.sigmas) == 16 and RULE.sigmas[0] == 0.005
    assert abs(RULE.sigmas[-1] - 3.7606) < 1e-4
    ratios = np.diff(np.log2(RULE.thetas))
    assert len(RULE.thetas) == 27 and np.allclose(ratios, 1 / 3)
    # every third octave of 1/64 lies midway, in log, between two candidates
    assert math.isclose(RULE.thetas[1], 2 ** (1 / 6) / 64)


def test_loud_matches_the_closed_form_of_the_affine_score() -> None:
    """With psi = 0 and unit standardization, l(d; y) = a^2 (d t_hat'y - d^2 / 2), so a pixel is
    loud iff log mean_d exp l(d) >= log(1 / alpha)."""
    torch.manual_seed(0)
    model = DARTS(4, RECIPE["model"])
    t = torch.tensor([3.0, 0.0, 4.0, 0.0], dtype=torch.float64)
    y, eps = torch.randn(500, 4) * 2, torch.randn(500, 4)
    c = torch.zeros(500, RECIPE["model"]["context"])
    sigma = 0.3
    a, _ = vp(sigma)
    projection = ((y + sigma * eps) @ (t / t.norm()).float()).double().numpy()
    d = RULE.distances.double().numpy()
    ell = a**2 * (projection[:, None] * d - d**2 / 2)
    expected = np.log(np.exp(ell).mean(1)) >= np.log(1 / RULE.alpha)
    assert 0 < expected.sum() < 500
    np.testing.assert_array_equal(loud(model, y, c, t, sigma, eps, RULE), expected)


def test_mix_response_counts_the_screened_pool() -> None:
    """Ten held-out pixels, a screen of 2 (share 0.2), two levels. Pixel 9 tops the projections
    and is loud at level 0, pixel 8 at level 1. Adding theta |t| = 10 to implant 3 (loud at
    both levels) pushes it into the screen beside pixel 9."""
    rule = Rule((0.1, 1.0), (1.0,), (0.1,), torch.tensor([1.0]), 0.01, 2, 0.2)
    reads = {"projection": np.arange(10.0), "t_norm": 10.0, "top": np.array([8, 9]),
             "pick": np.array([3, 5]), "clean": np.array([[False, True], [True, False]]),
             "flags": np.array([[[True, True], [False, False]]])}
    np.testing.assert_allclose(mix_response(reads, 0, 0.1, np.array([0, 1]), rule), [1.0, 0.5])
    np.testing.assert_allclose(mix_response(reads, 0, 0.0, np.array([0, 1]), rule), [0.5, 0.5])


def test_select_of_the_affine_score() -> None:
    """With psi = 0 (mean 0, std 1) the Rao statistic is s a y'v / sigma, a positive multiple of
    the projection on t_hat at every level. So the held-out PD of theta t is the share of held-out
    projections that theta |t| lifts above the (floor(PF n) + 1)-th largest one, at every level;
    the averaged PD is flat, sigma-hat is the smallest level; and the pool of an unscreened model
    is its held-out set."""
    scene, _ = gaussian_scene()
    inputs = prepare(scene, 0.1)
    model = DARTS(len(scene.target), SPEC)
    rule = Rule.from_config(RECIPE, (0.25, 1.0))
    held = torch.arange(0, len(scene.labels), 3)
    every = torch.ones(len(scene.labels), dtype=torch.bool)
    assert torch.equal(reference_pool(held, every, every), held)
    reads = held_out_reads(model, inputs, held, rule)
    projection = (inputs.fx[held] @ (inputs.t / inputs.t.norm())).numpy()
    threshold = np.sort(projection)[::-1][int(np.floor(PF * len(held)))]
    for j, theta in enumerate(rule.thetas):
        lifted = np.mean(projection + theta * float(inputs.t.norm()) > threshold)
        np.testing.assert_allclose(reads["pd"][j], lifted, atol=1e-12)
    screened = torch.from_numpy(np.flatnonzero(~inputs.usable.numpy()))
    chosen = select(model, reads, inputs, screened, rule)
    assert np.ptp(chosen["band_pd"]) == 0 and chosen["sigma_hat"] == rule.sigmas[0]


def test_held_out_glrt_of_the_affine_score() -> None:
    """With psi = 0 the GLRT is a^2 max(0, y't_hat)^2, increasing in the projection where it is
    positive, as the threshold is here; so its PD is the share of the implanted picks lifted above
    the (floor(PF n) + 1)-th largest projection, the same at every level, and sigma-hat_G is the
    smallest level."""
    scene, _ = gaussian_scene()
    inputs = prepare(scene, 0.1)
    model = DARTS(len(scene.target), SPEC)
    rule = Rule.from_config(RECIPE, (0.25, 1.0))
    held = torch.arange(0, len(scene.labels), 3)
    pd = held_out_glrt(model, inputs, held, rule, torch.logspace(-3, 1, 81))
    projection = (inputs.fx[held] @ (inputs.t / inputs.t.norm())).numpy()
    threshold = np.sort(projection)[::-1][int(np.floor(PF * len(held)))]
    assert threshold > 0
    pick = np.sort(np.random.default_rng(0).choice(len(held), min(rule.implants, len(held)),
                                                   replace=False))
    for j, theta in enumerate(rule.thetas):
        lifted = np.mean(projection[pick] + theta * float(inputs.t.norm()) > threshold)
        np.testing.assert_allclose(pd[j], lifted, atol=1e-12)
    assert np.ptp(band_average(pd, len(rule.thetas) - 1, rule)) == 0
