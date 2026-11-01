"""Folds, schedule and a short, reproducible training run."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dartsdet import training

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "darts.json"


def test_folds_hold_out_every_usable_pixel_once() -> None:
    usable = torch.rand(500, generator=torch.Generator().manual_seed(0)) > 0.2
    held = [training.held_out(usable, j, 5) for j in range(5)]
    everyone = torch.cat([h for h, _ in held])
    assert len(everyone) == int(usable.sum()) == len(set(everyone.tolist()))
    assert usable[everyone].all()
    for h, kept in held:
        assert len(h) + len(kept) == int(usable.sum())
        assert not set(h.tolist()) & set(kept.tolist())


def test_lr_factor() -> None:
    assert training.lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert training.lr_factor(9, 100, 10) == pytest.approx(0.5 * (1 + torch.cos(torch.tensor(
        9 * torch.pi / 100)).item()))
    assert training.lr_factor(100, 100, 10) == pytest.approx(0.0)


def test_log_uniform_range() -> None:
    sigma = training.log_uniform(10_000, 0.005, 4.0, torch.Generator().manual_seed(0))
    assert sigma.min() >= 0.005 and sigma.max() <= 4.0
    assert torch.log(sigma).mean() == pytest.approx((torch.log(torch.tensor(0.005 * 4.0)) / 2)
                                                    .item(), abs=0.05)


def short_run(fold: int) -> training.ScoreModel:
    spec = json.loads(RECIPE.read_text())
    spec |= {"width": 16, "depth": 1}
    spec["encoder"] |= {"channels": 8, "context": 4}
    spec["train"] |= {"max_steps": 6, "eval_every": 2, "crop": 12, "centres": 64}
    spec["train"]["training"] = spec["train"]["training"] | {"ema": 0.0, "warmup": 1}
    g = torch.Generator().manual_seed(1)
    fx = torch.randn(20 * 16, 3, generator=g, dtype=torch.float64)
    usable = torch.rand(20 * 16, generator=g) > 0.1
    return training.fit(fx, usable, (20, 16), fold, spec, "cpu")


def test_training_is_reproducible() -> None:
    first, second = short_run(1), short_run(1)
    assert first.validation == second.validation
    for key, value in first.state_dict().items():
        assert torch.equal(second.state_dict()[key], value)


def test_training_draws_are_pinned() -> None:
    # Recorded with torch 2.14 on the CPU; a reordered or added random draw changes them.
    model = short_run(1)
    steps, losses = zip(*model.validation, strict=True)
    assert steps == (2, 4, 6) and model.best_step == 2
    np.testing.assert_allclose(losses, [1.0315656661987305, 1.0318325757980347,
                                        1.0316987037658691], rtol=1e-5)
    weight = float(model.state_dict()["network.output.weight"].abs().sum())
    assert weight == pytest.approx(0.06297445297241211, rel=1e-4)


@pytest.mark.parametrize("k", range(8))
def test_dihedral_inverse(k: int) -> None:
    x = torch.arange(12.0).reshape(1, 1, 3, 4)
    assert torch.equal(training.dihedral(training.dihedral(x, k), k, inverse=True), x)
