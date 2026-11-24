"""Experiment 3 (Fig. 3): the noise level at nonzero amplitudes, on Pavia University and MUUFL.

Per background (configs/experiment3.json), one model is trained on the implant-free scene and
serves every signature. Each implanted scene (a class mean of Pavia University or a cloth
signature of MUUFL, at every amplitude) is whitened by the implant-free scene's fit and screened
on its own; its contexts leave out its screened pixels. Rows, per scene:
- DARTS: its Rao test at every level of the rule's grid, the rule's sigma-hat and sigma-u, and
  its plug-in GLRT at sigma-hat. The rule's reference pool is the model's held-out pixels plus the
  same share of the implant-free scene's screened pixels (`noise_level.reference_pool`).
- GMM: the Rao test of the mixture fitted on the pixels the model may read (its usable pixels,
  held-out fold included; EM, seed 0), and of the same mixture smoothed exactly at each level
  sigma: N(0, sigma^2 I) in the whitened coordinates adds sigma^2 U U' to every component
  covariance in the mixture's own coordinates. DARTS's sigma is instead in its standardized
  coordinates, the whitened bands divided by their standard deviation over the training pixels
  (0.81-0.99 on Pavia University, 0.87-1.00 on MUUFL), so its levels are close to, not equal to,
  the GMM's.
Fig. 3's third model, the DARTS network trained with LRao's objective, is not computed here: its
curves are stored rows, results/experiment3/<background>-lrao-objective.csv.
The DARTS models are checkpoints/experiment3/<background>/darts.pt, trained on the implant-free
scene minus its screen (their record says so); --train trains anew on every pixel of it, on
--device, for the recipe's steps, stopping early on MUUFL only (a background's `patience` null:
no early stopping, the final weights kept), and saves to <out>/checkpoints/<background>/darts.pt.
Statistics run on the CPU.

Writes <out>/<background>-darts.csv (target, alpha, sigma, test, auc, pd, pauc),
<out>/<background>-rule.csv (target, alpha, sigma_hat, theta_hat, rho_hat, sigma_u) and
<out>/<background>-gmm.csv (target, alpha, sigma, auc, pd, pauc). Usage, from
repro/: python -m experiments.experiment3 [--train] [--device DEVICE] [--data DIR] [--out DIR]
[--backgrounds pavia,muufl]
"""

import argparse
import dataclasses
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from baselines import gmm
from dartsdet import data, noise_level, statistics, whitening
from dartsdet.metrics import detection_metrics
from dartsdet.model import DARTS, pixels
from dartsdet.training import held_out, train
from experiments import default_device, write

REPRO = Path(__file__).resolve().parents[1]


def background(name: str, config: dict, recipe: dict, registry: dict,
               args: argparse.Namespace) -> tuple[dict, data.Scene, tuple, whitening.Inputs,
                                                  DARTS]:
    """(the scene entry, the implant-free scene, its whitening fit, its prepared inputs with the
    model's training pixels usable, the model)."""
    entry = registry[config["backgrounds"][name]["scene"]]
    pre = data.without_targets(data.load(entry | {"alpha": 0.0}, args.data))
    fitted = whitening.whitening(torch.from_numpy(pre.spectra()[pre.exists]))
    base = whitening.prepare(pre, recipe["screen"], fitted)
    every = dataclasses.replace(base, usable=torch.from_numpy(pre.exists))
    if not args.train:
        model = DARTS.load(REPRO / "checkpoints" / "experiment3" / name / "darts.pt")
        return entry, pre, fitted, base if model.record["pixels"] == "screened" else every, model
    start = time.time()
    plan = config["backgrounds"][name]
    spec = recipe | {"training": recipe["training"]
                     | ({"patience": plan["patience"]} if "patience" in plan else {})}
    model, record = train(every, spec, config["learning_rate"], config["fold"], args.device)
    path = args.out / "checkpoints" / name / "darts.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().save(path, **record, pixels="all")
    print(f"{name}: {record}, {time.time() - start:.0f} s", flush=True)
    return entry, pre, fitted, every, model


def scenes(name: str, config: dict, recipe: dict, setup: tuple, args: argparse.Namespace,
           keys: Iterable[tuple[int, float]] | None = None,
           ) -> Iterator[tuple[int, float, data.Scene, whitening.Inputs]]:
    """(target, alpha, scene, inputs) of the implanted scenes of a background, whitened by the
    implant-free scene's fit and screened on its own, loaded one at a time (a Pavia University
    scene and its inputs hold about 0.3 GB): every (target, alpha) of the plan, target by target,
    or those of `keys` in their order."""
    entry, _, fitted, _, _ = setup
    plan = config["backgrounds"][name]
    for target, alpha in keys or [(t, a) for t in plan["targets"] for a in plan["amplitudes"]]:
        scene = data.load(entry | {"target": target, "alpha": alpha}, args.data)
        yield target, alpha, scene, whitening.prepare(scene, recipe["screen"], fitted)


def gmm_rows(name: str, config: dict, setup: tuple, implanted: Iterable[tuple]) -> list[dict]:
    """The smoothed GMM's Rao test on every implanted scene, at every level of `gmm.sigmas`."""
    base, settings = setup[3], config["gmm"]
    model = gmm.fit(base.fx[base.usable], settings["components"][name], settings["seed"],
                    settings["reg_covar"])
    print(name, f"GMM converged {model.converged}", flush=True)
    rows = []
    for target, alpha, scene, inputs in implanted:
        for sigma in settings["sigmas"]:
            unmix = model.unmix
            smoothed = model._replace(covariances=model.covariances + sigma**2 * (unmix @ unmix.T))
            terms = gmm.line(smoothed, inputs.fx, inputs.t)
            score = gmm.rao(smoothed.weights.double().log(), terms)
            metrics = detection_metrics(scene.labels, score.numpy())
            rows.append({"target": target, "alpha": alpha, "sigma": sigma,
                         **{k: f"{v:.4f}" for k, v in metrics.items()}})
    return rows


def darts_rows(name: str, config: dict, recipe: dict, setup: tuple,
               implanted: Iterable[tuple]) -> tuple[list[dict], list[dict]]:
    """(metric rows, rule rows) of DARTS on every implanted scene of a background."""
    _, pre, _, base, model = setup
    plan = config["backgrounds"][name]
    held = held_out(base.usable, config["fold"], recipe["training"]["folds"])[0]
    pool = noise_level.reference_pool(held, base.usable, torch.from_numpy(pre.exists))
    rule = noise_level.Rule.from_config(recipe, (min(plan["amplitudes"]), max(plan["amplitudes"])))
    grid = recipe["glrt_amplitudes"]
    amplitudes = torch.logspace(grid["log10_first"], grid["log10_last"], grid["count"])
    rows, rules, reads = [], [], {}
    for target, alpha, scene, inputs in implanted:
        start = time.time()
        if target not in reads:  # one signature, one set of held-out reads
            reads[target] = noise_level.held_out_reads(
                model, dataclasses.replace(base, t=inputs.t), pool, rule)
        screened = torch.from_numpy(np.flatnonzero(scene.exists & ~inputs.usable.numpy()))
        chosen = noise_level.select(model, reads[target], inputs, screened, rule)
        z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
        key = {"target": target, "alpha": alpha}
        for sigma in rule.sigmas:
            c = statistics.contexts(model, z, mask, sigma)
            scores = {"Rao": statistics.rao(model, pixels(z), c, inputs.t, sigma).double()}
            if sigma == chosen["sigma_hat"]:
                scores["GLRT"] = statistics.glrt(model, pixels(z), c, inputs.t, sigma,
                                                 amplitudes)
            for test, score in scores.items():
                metrics = detection_metrics(scene.labels, score.numpy())
                rows.append(key | {"sigma": f"{sigma:.4f}", "test": test,
                                   **{k: f"{v:.4f}" for k, v in metrics.items()}})
        rules.append(key | {k: f"{chosen[k]:.4f}" for k in
                            ("sigma_hat", "theta_hat", "rho_hat", "sigma_u")})
        print(name, key, rules[-1], f"{time.time() - start:.0f} s", flush=True)
    return rows, rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train the DARTS models anew")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "experiment3")
    parser.add_argument("--backgrounds", default="pavia,muufl")
    args = parser.parse_args()
    recipe = json.loads((REPRO / "configs" / "darts.json").read_text())
    config = json.loads((REPRO / "configs" / "experiment3.json").read_text())
    registry = json.loads((REPRO / "configs" / "scenes.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.backgrounds.split(","):
        setup = background(name, config, recipe, registry, args)
        write(args.out / f"{name}-gmm.csv",
              gmm_rows(name, config, setup, scenes(name, config, recipe, setup, args)))
        rows, rules = darts_rows(name, config, recipe, setup,
                                 scenes(name, config, recipe, setup, args))
        write(args.out / f"{name}-darts.csv", rows)
        write(args.out / f"{name}-rule.csv", rules)


if __name__ == "__main__":
    main()
