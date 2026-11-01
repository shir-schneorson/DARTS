"""DART's noise law, early-stopping arithmetic and checkpoint guard."""

import json
from pathlib import Path

import pytest
import torch

from experiments import dart

SPEC = {"sampling": {"median": 0.25, "log_std": 1.2}, "sigma_min": 0.005, "sigma_max": 2.0}


def test_truncated_log_normal() -> None:
    sigma = dart.truncated_log_normal(100_000, torch.Generator().manual_seed(0), SPEC)
    assert sigma.dtype == torch.float64
    assert sigma.min() >= 0.005 and sigma.max() <= 2.0
    # the truncation is nearly symmetric about the median in log sigma here
    assert abs(float(sigma.median()) / 0.25 - 1) < 0.1


def test_first_minimum_prefers_the_earlier_of_equal_losses() -> None:
    assert dart.first_minimum([3.0, 1.0, 2.0, 1.0]) == 1


def test_a_dart_checkpoint_from_other_scene_settings_is_refused(tmp_path: Path) -> None:
    from experiments import experiment3
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" /
                         "experiment3.json").read_text())
    scene = {k: config[k] for k in ("scene", "background", "target", "screen_fraction")}
    torch.save({"state": {}, "seed": 0, "spec": config["dart"], "scene": scene, "n": 7},
               tmp_path / "dart.pt")
    config["target"] = config["target"] | {"distance": 4.0}
    with pytest.raises(ValueError, match="other scene"):
        experiment3.dart_model(torch.zeros(7, 10), 0, config, "cpu", tmp_path / "dart.pt")
