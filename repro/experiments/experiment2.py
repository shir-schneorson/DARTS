"""Experiment 2 (Fig. 2): spatial conditioning on a synthetic scene with known densities.

Per seed (configs/experiment2.json), the scene of `experiments.synthetic` is drawn. DARTS and
DART (the recipe without its context encoder), both with the experiment's smaller network and
step count, are trained on its implant-free version, every pixel usable, holding out fold `seed`;
both are whitened by that scene's own fit. Each implanted scene, at every amplitude, is whitened by
the same fit and screened on its own, and scored by:
- the oracles: the Rao test and the GLRT of the exact scene density, the region Rao oracle, and
  the region Rao oracle smoothed at the matched level;
- DART and DARTS, each at its own rule's sigma-hat on that scene, Rao test and plug-in GLRT, the
  contexts leaving out the scene's screened pixels, as in Experiment 1.
The statistics are computed on the CPU. Models come from checkpoints/experiment2/seed<s>/, or are
trained anew on --device with --train and saved in <out>/checkpoints/seed<s>/.

Writes <out>/runs.csv (seed, alpha, detector, sigma, auc, pd, pauc; sigma is sigma-hat, or sigma_m
for the matched-level oracle) and <out>/rule.csv (seed, alpha, model, sigma_hat, theta_hat,
rho_hat). Usage, from repro/:
    python -m experiments.experiment2 [--train] [--device DEVICE] [--out DIR]
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from dartsdet import noise_level, statistics, whitening
from dartsdet.metrics import detection_metrics
from dartsdet.model import DARTS, pixels
from dartsdet.training import held_out, train
from experiments import default_device, synthetic, write

REPRO = Path(__file__).resolve().parents[1]


def models(base: whitening.Inputs, recipe: dict, config: dict, seed: int,
           args: argparse.Namespace) -> dict[str, DARTS]:
    """{"DARTS", "DART"} of `seed`: loaded, or trained on `base` with fold `seed` and saved."""
    out = {}
    for name, override in (("DARTS", {}), ("DART", {"context": 0})):
        path = f"seed{seed}/{name.lower()}.pt"
        if not args.train:
            out[name] = DARTS.load(REPRO / "checkpoints" / "experiment2" / path)
            continue
        start = time.time()
        spec = recipe | {"model": recipe["model"] | config["model"] | override,
                         "training": recipe["training"] | {"steps": config["steps"],
                                                           "patience": config["patience"]}}
        model, record = train(base, spec, config["learning_rate"], seed, args.device)
        (args.out / "checkpoints" / f"seed{seed}").mkdir(parents=True, exist_ok=True)
        model.cpu().save(args.out / "checkpoints" / path, **record)
        print(f"seed {seed} {name}: {record}, {time.time() - start:.0f} s", flush=True)
        out[name] = model.cpu()
    return out


def scores(model: DARTS, inputs: whitening.Inputs, sigma: float,
           amplitudes: torch.Tensor) -> dict[str, np.ndarray]:
    """The model's Rao test and GLRT (P,) on a scene at sigma."""
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    c = statistics.contexts(model, z, mask, sigma)
    return {"Rao": statistics.rao(model, pixels(z), c, inputs.t, sigma).double().numpy(),
            "GLRT": statistics.glrt(model, pixels(z), c, inputs.t, sigma, amplitudes).numpy()}


def oracles(scene: synthetic.Synthetic, fitted: tuple[torch.Tensor, torch.Tensor],
            x: torch.Tensor, level: float, config: dict) -> dict[tuple, np.ndarray]:
    """{(detector, sigma): score (P,)} of the oracles on spectra x (P, B), the smoothed one at the
    matched level `level`."""
    g = config["oracle_glrt"]
    return {("oracle Rao", None): synthetic.oracle_rao(scene, x).numpy(),
            ("oracle GLRT", None): synthetic.oracle_glrt(scene, x, g["theta_max"],
                                                         g["count"]).numpy(),
            ("region Rao oracle", None): synthetic.region_score(scene, x).numpy(),
            ("region Rao oracle, matched level", level):
                synthetic.region_smoothed(scene, fitted, x, level).numpy()}


def seed_rows(seed: int, recipe: dict, config: dict,
              args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """(metric rows, rule rows) of one seed, every amplitude."""
    scene = synthetic.draw(config, seed)
    clean = synthetic.implanted(scene, 0.0)
    fitted = whitening.whitening(torch.from_numpy(clean.spectra()))
    base = dataclasses.replace(whitening.prepare(clean, recipe["screen"], fitted),
                               usable=torch.ones(len(clean.labels), dtype=torch.bool))
    trained = models(base, recipe, config, seed, args)
    held = held_out(base.usable, seed, recipe["training"]["folds"])[0]
    pool = noise_level.reference_pool(held, base.usable, torch.from_numpy(clean.exists))
    rule = noise_level.Rule.from_config(recipe, tuple(config["rule_amplitudes"]))
    grid = recipe["glrt_amplitudes"]
    amplitudes = torch.logspace(grid["log10_first"], grid["log10_last"], grid["count"])
    reads = {name: noise_level.held_out_reads(model, base, pool, rule)
             for name, model in trained.items()}
    levels = synthetic.matched_levels(scene, fitted, config["amplitudes"],
                                      config["matched_level_grid"])
    rows, rules = [], []
    for alpha, level in zip(config["amplitudes"], levels, strict=True):
        start = time.time()
        implanted = synthetic.implanted(scene, alpha)
        inputs = whitening.prepare(implanted, recipe["screen"], fitted)
        screened = torch.from_numpy(np.flatnonzero(implanted.exists & ~inputs.usable.numpy()))
        maps = oracles(scene, fitted, torch.from_numpy(implanted.spectra()), level, config)
        for name, model in trained.items():
            chosen = noise_level.select(model, reads[name], inputs, screened, rule)
            for test, score in scores(model, inputs, chosen["sigma_hat"], amplitudes).items():
                maps[f"{name} {test}", chosen["sigma_hat"]] = score
            rules.append({"seed": seed, "alpha": alpha, "model": name,
                          **{k: f"{chosen[k]:.4f}" for k in ("sigma_hat", "theta_hat", "rho_hat")}})
        for (detector, sigma), score in maps.items():
            metrics = detection_metrics(implanted.labels, score)
            rows.append({"seed": seed, "alpha": alpha, "detector": detector,
                         "sigma": "" if sigma is None else f"{sigma:.4f}",
                         **{k: f"{v:.4f}" for k, v in metrics.items()}})
        print(f"seed {seed} alpha {alpha}: sigma_m {level:.4f}, sigma-hat "
              + ", ".join(f"{r['model']} {r['sigma_hat']}" for r in rules[-len(trained):]) + " | "
              + " ".join(f"{r['detector']} {r['pd']}" for r in rows[-len(maps):])
              + f" | {time.time() - start:.0f} s", flush=True)
    return rows, rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train DARTS and DART anew")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "experiment2")
    args = parser.parse_args()
    recipe = json.loads((REPRO / "configs" / "darts.json").read_text())
    config = json.loads((REPRO / "configs" / "experiment2.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows, rules = [], []
    for seed in config["seeds"]:
        seed_out, seed_rules = seed_rows(seed, recipe, config, args)
        rows, rules = rows + seed_out, rules + seed_rules
        write(args.out / "runs.csv", rows)
        write(args.out / "rule.csv", rules)


if __name__ == "__main__":
    main()
