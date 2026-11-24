"""Experiment 4's estimators in closed form: omega and the threshold with psi = 0, where the
Rao statistic is a positive multiple of the projection on t_hat; and the level choices on toy
curves."""

import dataclasses
import math

import numpy as np

from dartsdet.metrics import PF
from dartsdet.model import DARTS, vp
from dartsdet.test_statistics import SPEC, gaussian_scene
from dartsdet.whitening import prepare
from experiments.experiment4 import levels, matched, omega, with_fixed


def test_omega_of_the_affine_score_is_one() -> None:
    """With psi = 0 the gradient of the Rao statistic is constant along t / std, so J t is too."""
    scene, _ = gaussian_scene()
    inputs = prepare(scene, 0.1)
    model = DARTS(len(scene.target), SPEC)
    spec = {"pixels": 500, "noise_seed": 4000}
    assert math.isclose(omega(model, scene, inputs, 0.005, np.random.default_rng(0), spec), 1.0,
                        rel_tol=1e-9)


def test_threshold_and_matched_amplitude_of_the_affine_score() -> None:
    """Rao = s a (y't_hat) / sigma with std 1, so k is that multiple of the (floor(PF n) + 1)-th
    largest projection, and theta_m = 2 k sigma^2 / |t|."""
    scene, _ = gaussian_scene()
    scene = dataclasses.replace(scene, labels=np.where(np.arange(len(scene.labels)) % 50, 0, 1))
    inputs = prepare(scene, 0.1)
    model = DARTS(len(scene.target), SPEC)
    projection = (inputs.fx @ (inputs.t / inputs.t.norm())).numpy()
    projection = np.sort(projection[scene.labels == 0])[::-1]
    for row in levels(model, scene, inputs, (0.05, 0.4)):
        a, s = vp(row["sigma"])
        k = s * a * projection[math.floor(PF * len(projection))] / row["sigma"]
        assert math.isclose(row["k"], k, rel_tol=1e-5)
        assert math.isclose(row["theta_m"], 2 * row["k"] * row["sigma"] ** 2
                            / float(inputs.t.norm()), rel_tol=1e-6)


def test_matched_level_on_a_toy_curve() -> None:
    """theta_m = sigma on a log grid with a dip: the dip is left out of the increasing part, the
    amplitude 0.3 maps to the nearest level 0.25, and an unknown amplitude takes the part's best
    PD."""
    grid = [0.0625, 0.125, 0.25, 0.5, 1.0]
    theta = [0.0625, 0.125, 0.25, 0.2, 1.0]
    pd = [0.1, 0.9, 0.5, 0.95, 0.3]
    curve = [{"sigma": g, "theta_m": t, "pd": p} for g, t, p in zip(grid, theta, pd, strict=True)]
    assert matched(curve, 1.0, 0.3) == 0.25
    assert matched(curve, 0.5, 0.25) == 0.125
    assert matched(curve, 1.0, None) == 0.125


def test_fixed_is_the_best_mean_level_of_a_signature() -> None:
    base = {"set": "s", "scene": "x", "target": 1, "omega": 1.0, "theta_hat": 1.0}
    choices = {f"sigma_{n}": 0.1 for n in ("sigma_min", "uniform", "matched", "matched_omega1",
                                          "rule", "best")}
    rows = [base | {"alpha": 1.0, "curve": {0.1: 0.2, 0.2: 0.9}} | choices,
            base | {"alpha": 2.0, "curve": {0.1: 0.8, 0.2: 0.6}} | choices]
    out = with_fixed(rows)
    assert [r["sigma_fixed"] for r in out] == ["0.2000", "0.2000"]
    assert [r["pd_fixed"] for r in out] == ["0.9000", "0.6000"]
