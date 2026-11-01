"""The per-scene steps end to end, on a tiny scene with a tiny recipe."""

import json
from pathlib import Path

import numpy as np
import pytest

from dartsdet import evaluation, noise_level, pipeline, preprocessing, scenes

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "darts.json"


def tiny_recipe() -> dict:
    spec = json.loads(RECIPE.read_text())
    spec |= {"width": 16, "depth": 1}
    spec["encoder"] |= {"channels": 8, "context": 4}
    spec["train"] |= {"max_steps": 4, "eval_every": 2, "crop": 12, "centres": 64}
    return spec


def tiny_scene() -> scenes.Scene:
    rng = np.random.default_rng(3)
    H, W, B = 16, 20, 3
    cube = rng.standard_normal((H, W, B))
    d = np.array([2.0, -1.0, 0.5])
    labels = np.zeros(H * W)
    labels[rng.choice(H * W, 4, replace=False)] = 1
    flat = cube.reshape(-1, B, order="F")
    flat[labels == 1] += d
    return scenes.Scene(flat.reshape(H, W, B, order="F"), labels, np.ones(H * W, dtype=bool), d,
                        additive=True)


def test_fold_model_trains_saves_reloads_and_guards(tmp_path: Path) -> None:
    inputs, spec = preprocessing.prepare(tiny_scene()), tiny_recipe()
    trained = pipeline.fold_model(inputs, spec, 1, "cpu", tmp_path / "fold1.pt")
    loaded = pipeline.fold_model(inputs, spec, 1, "cpu", tmp_path / "fold1.pt")
    np.testing.assert_array_equal(pipeline.statistic(trained, inputs, 0.1, "cpu"),
                                  pipeline.statistic(loaded, inputs, 0.1, "cpu"))
    with pytest.raises(ValueError, match="not trained as fold 2"):
        pipeline.fold_model(inputs, spec, 2, "cpu", tmp_path / "fold1.pt")
    with pytest.raises(ValueError, match="not trained"):
        pipeline.fold_model(inputs, spec | {"width": 32}, 1, "cpu", tmp_path / "fold1.pt")


def test_rule_reads_to_metrics(tmp_path: Path) -> None:
    scene, spec = tiny_scene(), tiny_recipe()
    inputs = preprocessing.prepare(scene)
    sigmas = noise_level.grid(spec["sigma_max"])
    models = [pipeline.fold_model(inputs, spec, j, "cpu", tmp_path / f"fold{j}.pt")
              for j in range(2)]
    reads = [pipeline.rule_reads(m, inputs, scene.exists, j, sigmas, "cpu")
             for j, m in enumerate(models)]
    held, screened, C = reads[0]
    n_screened = int((~inputs.usable).sum())
    assert held.shape[1:] == (len(sigmas), len(noise_level.LR_THETAS))
    assert screened.shape == (n_screened, len(sigmas), len(noise_level.LR_THETAS))
    assert C.shape == (len(held), len(sigmas))
    sigma = pipeline.sigma_hat(reads, sigmas)
    assert sigmas[0] <= sigma <= sigmas[-1]
    score = evaluation.ensemble([pipeline.statistic(m, inputs, sigma, "cpu") for m in models],
                                scene.exists)
    metrics = evaluation.detection_metrics(scene.labels, score)
    assert np.isfinite(score).all() and all(0 <= v <= 1 for v in metrics.values())
