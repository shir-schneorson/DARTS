"""The training mechanics: the held-out split, the crops, and the stopping and checkpoint rule."""

import json
from pathlib import Path

import numpy as np
import torch

from dartsdet import whitening
from dartsdet.data import Scene
from dartsdet.training import crops, held_out, train

RECIPE = json.loads((Path(__file__).parents[1] / "configs" / "darts.json").read_text())


def test_folds_partition_the_usable_pixels() -> None:
    usable = torch.rand(1000, generator=torch.Generator().manual_seed(0)) > 0.3
    folds = [held_out(usable, j, 5) for j in range(5)]
    held = torch.cat([h for h, _ in folds]).sort().values
    assert torch.equal(held, torch.nonzero(usable)[:, 0])
    for h, kept in folds:
        assert abs(len(h) - usable.sum() / 5) <= 1
        assert torch.equal(torch.cat([h, kept]).sort().values, held)


def test_crops_cut_every_image_alike() -> None:
    image = torch.arange(2 * 30 * 20, dtype=torch.float32).reshape(1, 2, 30, 20)
    other = image[:, :1] * 10
    a, b = crops([image, other], 16, 5, torch.Generator().manual_seed(0))
    assert a.shape == (5, 2, 16, 16) and torch.equal(b, a[:, :1] * 10)
    narrow, = crops([image], 25, 3, torch.Generator().manual_seed(1))
    assert narrow.shape == (3, 2, 25, 20)


def test_kept_model_is_the_checkpoint_before_the_lowest_held_out_loss() -> None:
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((12, 14, 3))
    scene = Scene(cube, np.zeros(168), np.ones(168, dtype=bool), np.ones(3), additive=True)
    recipe = json.loads(json.dumps(RECIPE))
    recipe["training"] |= {"steps": 60, "crop": 8, "warmup": 2, "ema": 0.5, "validate_every": 2,
                           "checkpoint_every": 5, "patience": 6}
    _, record = train(whitening.prepare(scene, 0.1), recipe, 1e-2, 0, "cpu")
    assert record["step"] % 5 == 0
    assert record["best_step"] - 5 < record["step"] <= record["best_step"]
    assert record["stopped"] == 60 or record["stopped"] - record["best_step"] >= 6


def test_training_repeats_bit_for_bit() -> None:
    """Every draw of training is seeded: two runs on the CPU give the same weights and record."""
    rng = np.random.default_rng(1)
    cube = rng.standard_normal((12, 14, 3))
    scene = Scene(cube, np.zeros(168), np.ones(168, dtype=bool), np.ones(3), additive=True)
    recipe = json.loads(json.dumps(RECIPE))
    recipe["training"] |= {"steps": 20, "crop": 8, "warmup": 2, "ema": 0.5, "validate_every": 2,
                           "checkpoint_every": 4, "patience": 10}
    runs = [train(whitening.prepare(scene, 0.1), recipe, 1e-2, 0, "cpu") for _ in range(2)]
    (first, record), (second, again) = runs
    assert record == again
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_without_patience_training_runs_every_step_and_keeps_the_last() -> None:
    rng = np.random.default_rng(2)
    cube = rng.standard_normal((12, 14, 3))
    scene = Scene(cube, np.zeros(168), np.ones(168, dtype=bool), np.ones(3), additive=True)
    recipe = json.loads(json.dumps(RECIPE))
    recipe["training"] |= {"steps": 30, "crop": 8, "warmup": 2, "ema": 0.5, "validate_every": 2,
                           "checkpoint_every": 4, "patience": None}
    _, record = train(whitening.prepare(scene, 0.1), recipe, 1e-2, 0, "cpu")
    assert record["stopped"] == 30 and record["step"] == 30
