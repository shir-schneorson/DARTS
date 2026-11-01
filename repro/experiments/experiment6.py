"""Experiment 6 (Tables I and II), DARTS rows: five fold models per scene, the noise-level rule
and the ensemble Rao test.

Per scene: train fold j = 0..4 (or load it from <checkpoints>/experiment6/<scene>/fold<j>.pt),
pool the five folds' rule reads into sigma_hat (`pipeline.sigma_hat`), read each fold's Rao
statistic at sigma_hat and combine them by `evaluation.ensemble`. Writes <out>/<scene>.json:
`sigma`, and under `labels` the AUC, PD at 1e-3 and partial AUC; for MUUFL also under `masked`,
the same with every emplaced cloth's halo left out, the numbers Table II reports.

Usage: python -m experiments.experiment6 [--device DEVICE] [--data DIR] [--checkpoints DIR]
                                         [--out DIR] [scene ...]
"""

import argparse
import json

from dartsdet import evaluation, noise_level, pipeline, preprocessing, scenes
from experiments.cli import REPRO, arguments

SCENES = ("sandiego", "abu-beach-4", "pavia-u-a0.75", "muufl-campus1-220")


def run(name: str, args: argparse.Namespace) -> dict:
    """sigma_hat and the metrics of scene `name`."""
    scene = scenes.load(name, REPRO / "configs" / "scenes.json", args.data)
    inputs = preprocessing.prepare(scene)
    spec = json.loads((REPRO / "configs" / "darts.json").read_text())
    sigmas = noise_level.grid(spec["sigma_max"])
    folder = args.checkpoints / "experiment6" / name
    models = [pipeline.fold_model(inputs, spec, j, args.device, folder / f"fold{j}.pt")
              for j in range(spec["train"]["folds"])]
    sigma = pipeline.sigma_hat([pipeline.rule_reads(m, inputs, scene.exists, j, sigmas,
                                                    args.device)
                                for j, m in enumerate(models)], sigmas)
    score = evaluation.ensemble([pipeline.statistic(m, inputs, sigma, args.device)
                                 for m in models], scene.exists)
    out = {"sigma": sigma, "labels": evaluation.detection_metrics(scene.labels, score)}
    if scene.masked is not None:
        out["masked"] = evaluation.detection_metrics(scene.masked, score)
    return out


def main() -> None:
    parser = arguments(__doc__, "experiment6", device=True, data=True, checkpoints=True)
    parser.add_argument("scenes", nargs="*", default=list(SCENES), help="scenes to run")
    args = parser.parse_args()
    if unknown := sorted(set(args.scenes) - set(SCENES)):
        parser.error(f"unknown scenes {unknown}; choose from {list(SCENES)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.scenes:
        result = run(name, args)
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
