"""The noise-level rule's parts on curves whose answers are known."""

import numpy as np
import pytest

from dartsdet import noise_level as nl

SIGMAS = np.array([1.0, 2.0, 4.0])


def test_first_crossing_interpolates_in_log_sigma() -> None:
    # log m goes from log 2 to -log 2 between sigma = 2 and 4: halfway in log sigma is 2 sqrt 2
    assert nl.first_crossing(np.array([4.0, 2.0, 0.5]), SIGMAS) == pytest.approx(2 * np.sqrt(2))


def test_first_crossing_ends() -> None:
    assert nl.first_crossing(np.array([0.5, 0.2, 0.1]), SIGMAS) == 1.0
    assert nl.first_crossing(np.array([4.0, 3.0, 2.0]), SIGMAS) == 4.0
    assert nl.first_crossing(np.array([4.0, 0.0, 0.0]), SIGMAS) == 1.0  # log 0: at the last above


def test_lr_rate_obeys_markov_for_a_correct_model() -> None:
    # Whitened Gaussian, C = 1: l(theta) = theta A - theta^2 / 2 with A ~ N(0, 1) under the null,
    # so E exp l(theta) = 1 and the mixture reaches 1/alpha with probability at most alpha.
    a = np.random.default_rng(0).standard_normal(200_000)
    thetas = np.array(nl.LR_THETAS)
    ell = (a[:, None] * thetas - thetas**2 / 2)[:, None, :]
    assert nl.lr_rate(ell)[0] < 1


def test_lr_rate_flags_an_overconfident_model() -> None:
    # A model whose score is twice the truth reads l = 2 theta A - theta^2 / 2 ... and exceeds.
    a = np.random.default_rng(0).standard_normal(200_000)
    thetas = np.array(nl.LR_THETAS)
    ell = (2 * a[:, None] * thetas - thetas**2 / 2)[:, None, :]
    assert nl.lr_rate(ell)[0] > 1


def test_convex_sigma_is_the_level_after_the_last_violation() -> None:
    C = np.ones((1000, 4))
    C[:5, 1] = -1                # 0.5% non-convex at level 1 > PF
    C[:1, 2] = -1                # 0.1% at level 2 = PF, allowed
    assert nl.convex_sigma(C, np.array([1.0, 2.0, 3.0, 4.0])) == 3.0
    assert nl.convex_sigma(np.ones((10, 4)), np.array([1.0, 2.0, 3.0, 4.0])) == 1.0


def test_plateau_end_recovers_the_breakpoint() -> None:
    sigmas = np.exp(np.linspace(np.log(0.01), np.log(1.0), 30))
    knee = 0.1
    rate = 5 * (1 - 0.3 * np.maximum(0, np.log(sigmas) - np.log(knee)) ** 2)
    rate = np.maximum(rate, 0)
    assert nl.plateau_end(rate, sigmas) == pytest.approx(knee, rel=0.02)


def test_plateau_end_without_exceedance_is_the_first_level() -> None:
    assert nl.plateau_end(np.zeros(5), np.arange(1.0, 6.0)) == 1.0


def test_noise_level_is_the_largest_part() -> None:
    sigmas = np.array([0.01, 0.02, 0.04, 0.08])
    ell_held = np.full((100, 4, 5), -10.0)    # never exceeds: sigma_LR = first level
    C = np.ones((100, 4))
    C[:50, :3] = -1                           # non-convex through level 2: sigma_cvx = 0.08
    ell_screened = np.full((100, 4, 5), -10.0)
    assert nl.noise_level(ell_held, ell_screened, C, sigmas) == 0.08


def test_grid() -> None:
    levels = nl.grid(4.0)
    assert levels[0] == 0.005 and levels[-1] <= 4.0 < levels[-1] * 200 ** (1 / 12)
    assert len(levels) == 16
