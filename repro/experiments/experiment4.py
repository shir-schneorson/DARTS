"""Experiment 4: the Rao test against the GLRT on real backgrounds, over targets and amplitudes.

One fold model per background (fold 0), trained on the scene before its implants: Experiment 6's
Pavia University and MUUFL scenes (configs/scenes.json) at alpha = 0. Their screen, and so the
model's training pixels and noise level, follow that registry entry's target, Pavia class 9 and
MUUFL signature 0; the noise level comes from the rule on this fold's reads, before any implant is
scored. For each target type (Pavia: the 9 class means; MUUFL: the 4 lab signatures) and
amplitude, the implanted scene is whitened with the pre-implant scene's fit, so that t = W d
exactly, and its contexts leave out its own screen. At sigma_hat: the Rao statistic, the one-sided
GLRT and, for reference, the one-sided AMF of the whitened pixels; scored on the implant labels
(MUUFL: every emplaced cloth's halo left out). Writes <out>/<background>.csv, row by row
to a .partial file that takes the final name once complete.

Usage: python -m experiments.experiment4 [--device DEVICE] [--data DIR] [--checkpoints DIR]
                                         [--out DIR] pavia|muufl
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from dartsdet import evaluation, noise_level, pipeline, preprocessing, scenes
from dartsdet.model import ScoreModel
from dartsdet.reads import contexts, glrt, rao
from experiments.cli import REPRO, arguments

PLAN = {"pavia": {"scene": "pavia-u-a0.75", "types": range(1, 10),
                  "alphas": (0.125, 0.25, 0.5, 1.0, 2.0)},
        "muufl": {"scene": "muufl-campus1-220", "types": range(4),
                  "alphas": (0.025, 0.05, 0.1, 0.2, 0.4)}}
FIELDS = ("scene", "type", "alpha", "detector", "sigma", "auc", "pd", "pauc", "targets")
GLRT_GRID = (10.0, 250)  # theta_max (multiples of t) and grid points


def background(entry: dict, args: argparse.Namespace, spec: dict,
               path: Path) -> tuple[ScoreModel, tuple[torch.Tensor, torch.Tensor], float]:
    """(fold-0 model, the pre-implant whitening, sigma_hat) of a background."""
    pre = scenes.without_targets(scenes.from_entry(entry | {"alpha": 0.0}, args.data))
    fitted = preprocessing.whitening(torch.from_numpy(pre.spectra()[pre.exists]))
    inputs = preprocessing.prepare(pre, fitted)
    sigmas = noise_level.grid(spec["sigma_max"])
    model = pipeline.fold_model(inputs, spec, 0, args.device, path)
    reads = pipeline.rule_reads(model, inputs, pre.exists, 0, sigmas, args.device)
    return model, fitted, pipeline.sigma_hat([reads], sigmas)


def statistics(model: ScoreModel, scene: scenes.Scene, inputs: preprocessing.Inputs, sigma: float,
               device: str) -> dict[str, np.ndarray]:
    """Rao, GLRT and AMF maps (P,) of an implanted scene."""
    zc, ctx = contexts(model, inputs.fx.float().to(device), inputs.usable.to(device), inputs.grid)
    t_hat = pipeline.unit_signature(inputs, device)
    with torch.no_grad():
        rao_map = rao(model, zc, ctx, t_hat, sigma).cpu().double().numpy()
    amf = np.zeros(len(scene.exists))
    amf[scene.exists] = preprocessing.amf(inputs.fx[scene.exists].numpy(), inputs.t.numpy())
    return {"Rao": rao_map,
            "GLRT": glrt(model, zc, ctx, t_hat, float(inputs.t.norm()), sigma,
                         GLRT_GRID).double().numpy(),
            "AMF": amf}


def run(which: str, args: argparse.Namespace, writer: csv.DictWriter) -> None:
    """One row per (type, amplitude, detector) of background `which`, written as produced."""
    plan = PLAN[which]
    entry = json.loads((REPRO / "configs" / "scenes.json").read_text())[plan["scene"]]
    spec = json.loads((REPRO / "configs" / "darts.json").read_text())
    path = args.checkpoints / "experiment4" / which / "fold0.pt"
    model, fitted, sigma = background(entry, args, spec, path)
    for alpha in plan["alphas"]:
        for target in plan["types"]:
            scene = scenes.from_entry(entry | {"target": target, "alpha": alpha}, args.data)
            labels = scene.labels if scene.masked is None else scene.masked
            maps = statistics(model, scene, preprocessing.prepare(scene, fitted), sigma,
                              args.device)
            rows = [{"scene": plan["scene"], "type": target, "alpha": alpha, "detector": detector,
                     "sigma": sigma, **evaluation.detection_metrics(labels, stat),
                     "targets": int((labels == 1).sum())} for detector, stat in maps.items()]
            writer.writerows(rows)
            print(which, target, alpha, {r["detector"]: round(r["pd"], 3) for r in rows},
                  flush=True)


def main() -> None:
    parser = arguments(__doc__, "experiment4", device=True, data=True, checkpoints=True)
    parser.add_argument("background", choices=sorted(PLAN))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.background}.csv"
    with out.with_suffix(".partial").open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        run(args.background, args, writer)
    os.replace(out.with_suffix(".partial"), out)


if __name__ == "__main__":
    main()
