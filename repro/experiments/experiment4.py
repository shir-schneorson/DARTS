"""Experiment 4 (Table III): DARTS at six choices of the noise level, on the 45 Pavia University
and 24 MUUFL scenes of Experiment 3 and the four scenes of Experiment 1.

Per scene, the Rao test of the shipped model at every level sigma of the rule's grid gives its
detection probability and its threshold kappa(sigma), the (floor(PF n) + 1)-th largest value over
the n background pixels. The amplitude a level is matched to, eq. (sigma-amplitude) solved for
theta in the model's standardized units (amplitude score U = |t| times the Rao statistic,
signature step t / std per unit amplitude), is

    theta_m(sigma) = 2 |t| kappa(sigma) sigma^2 / (omega |t / std|^2),

with omega = cos^2(t / std, J t). J t = E_0[grad U] (Section IV-E) is estimated at the smallest
level: over `pixels` background pixels drawn without replacement, noised at that level,
g1 = mean grad U and J t = g1 + sigma^2 mean(H g1), H = -Hess log p_sigma, the Jacobian of
eps_hat / sigma (`statistics.rao_gradient` along g1). The draws replay the paper's computation:
one numpy generator seeded `seed` per Experiment 3 background, its scenes taken amplitude by
amplitude, then signature; a fresh one per Experiment 1 scene; the noise from a torch generator
seeded `noise_seed`.

The choices of the level (Table III's rows):
- sigma_min: the smallest level;
- fixed: one level per target signature, the best mean PD over its amplitudes (labels);
- uniform: sigma-u, the rule without its amplitude estimate;
- matched (and matched_omega1, omega = 1): the true amplitude log-interpolated on the increasing
  part of theta_m(sigma) (its strict running maxima), then the nearest level in log; San Diego and
  ABU-beach-4 have no known amplitude, so theirs is the best level of that part (labels);
- rule: sigma-hat;
- best: the level with the highest PD (labels).
sigma-hat, sigma-u and theta-hat (snapped to the rule's amplitude grid) come from Experiment 3's
rule.csv files (--results) and, for Experiment 1, from `experiment1.choice`. Those files must come
from the shipped models, which this experiment scores: a --results directory holding retrained
checkpoints is refused. Statistics run on the CPU.

Fig. 4 (`figure_data`): on one signature per background (configs/experiment4.json `figure`), the
rule's responses on three scenes and the gray band of candidate implants, the min and max of the
normalized reference responses over the rule's amplitude grid without its sixth-octave offset and
the shares `band_rhos`, as in the paper's figure.

Writes <out>/thresholds.csv (set, scene, target, alpha, sigma, k, pd, and theta_m before the
division by omega) and <out>/levels.csv (set, scene, target, alpha, omega, theta_hat, and
sigma_<choice>, pd_<choice>) and <out>/figure4.json; then prints the numbers the text quotes.
Usage, from repro/: python -m experiments.experiment4 [--data DIR] [--results DIR] [--out DIR]
"""

import argparse
import csv
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dartsdet import data, noise_level, statistics, whitening
from dartsdet.metrics import PF, detection_metrics
from dartsdet.model import DARTS, pixels
from dartsdet.training import held_out
from experiments import experiment1, experiment3, write

REPRO = Path(__file__).resolve().parents[1]
CHOICES = ("sigma_min", "fixed", "uniform", "matched", "matched_omega1", "rule", "best")
BACKGROUNDS = {"pavia": "Pavia University", "muufl": "MUUFL"}


def omega(model: DARTS, scene: data.Scene, inputs: whitening.Inputs, sigma: float,
          rng: np.random.Generator, spec: dict) -> float:
    """cos^2(t / std, J t) of a scene, J t = E_0[grad U] estimated at level sigma over
    spec["pixels"] of its background pixels drawn from `rng` and noised with the generator seeded
    spec["noise_seed"]."""
    background = np.flatnonzero(scene.labels == 0)
    pick = torch.from_numpy(np.sort(rng.choice(background, min(spec["pixels"], len(background)),
                                               replace=False)))
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    c = statistics.contexts(model, z, mask, sigma)[pick]
    noise = torch.Generator().manual_seed(spec["noise_seed"])
    y = pixels(z)[pick]
    y = y + sigma * torch.randn(y.shape, generator=noise)
    step = inputs.t.float() / model.std
    g1 = statistics.rao_gradient(model, y, c, step, sigma).double().mean(0)
    jt = g1 + sigma**2 * statistics.rao_gradient(model, y, c, g1.float(), sigma).double().mean(0)
    step = step.double()
    return float((step @ jt) ** 2 / (step.norm() ** 2 * jt.norm() ** 2))


def levels(model: DARTS, scene: data.Scene, inputs: whitening.Inputs,
           sigmas: tuple[float, ...]) -> list[dict]:
    """Per level: sigma, the Rao test's threshold k and PD on the scene, and
    2 |t| k sigma^2 / |t / std|^2 (theta_m before dividing by omega)."""
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    background = scene.labels == 0
    step = inputs.t.float() / model.std
    out = []
    for sigma in sigmas:
        c = statistics.contexts(model, z, mask, sigma)
        rao = statistics.rao(model, pixels(z), c, inputs.t, sigma).double().numpy()
        k = float(np.sort(rao[background])[::-1][math.floor(PF * background.sum())])
        out.append({"sigma": sigma, "k": k, "pd": detection_metrics(scene.labels, rao)["pd"],
                    "theta_m": 2 * float(inputs.t.norm()) * k * sigma**2
                    / float(step.norm()) ** 2})
    return out


def matched(curve: list[dict], w: float, amplitude: float | None) -> float:
    """The level matched to `amplitude` by theta_m / w on the increasing part of theta_m, its
    strict running maxima (the best-PD level of that part if the amplitude is unknown)."""
    grid = np.array([r["sigma"] for r in curve])
    log_theta = np.log([r["theta_m"] / w for r in curve])
    keep = np.r_[True, np.diff(np.maximum.accumulate(log_theta)) > 0]
    if amplitude is None:
        pd = {r["sigma"]: r["pd"] for r in curve}
        return max(grid[keep], key=pd.get)
    solved = np.exp(np.interp(np.log(amplitude), log_theta[keep], np.log(grid[keep])))
    return float(grid[int(np.argmin(np.abs(np.log(grid / solved))))])


def chosen_rows(key: dict, curve: list[dict], w: float, amplitude: float | None,
                rule: dict) -> dict:
    """The scene's row of levels.csv without "fixed", which needs the other scenes of its
    signature."""
    pd = {r["sigma"]: r["pd"] for r in curve}
    grid = [r["sigma"] for r in curve]
    sigma = {"sigma_min": grid[0], "uniform": rule["sigma_u"],
             "matched": matched(curve, w, amplitude),
             "matched_omega1": matched(curve, 1.0, amplitude),
             "rule": rule["sigma_hat"], "best": max(grid, key=pd.get)}
    nearest = {name: min(grid, key=lambda g, s=s: abs(g - s)) for name, s in sigma.items()}
    return key | {"omega": w, "theta_hat": rule["theta_hat"], "curve": pd} | {
        f"sigma_{name}": s for name, s in nearest.items()}


def experiment3_rows(args: argparse.Namespace, recipe: dict, registry: dict,
                     spec: dict) -> tuple[list[dict], list[dict]]:
    """(levels rows without "fixed", thresholds rows) of the Experiment 3 scenes."""
    config = json.loads((REPRO / "configs" / "experiment3.json").read_text())
    rows, thresholds = [], []
    for name, plan in config["backgrounds"].items():
        setup = experiment3.background(name, config, recipe, registry,
                                       argparse.Namespace(train=False, data=args.data))
        rule = noise_level.Rule.from_config(
            recipe, (min(plan["amplitudes"]), max(plan["amplitudes"])))
        thetas = np.array(rule.thetas)
        rules = {}
        for r in csv.DictReader((args.results / f"{name}-rule.csv").open()):
            rules[int(r["target"]), float(r["alpha"])] = {k: float(v) for k, v in r.items()} | {
                "theta_hat": float(thetas[np.argmin(np.abs(np.log(
                    thetas / float(r["theta_hat"]))))])}
        rng = np.random.default_rng(spec["seed"])
        keys = sorted(((t, a) for t in plan["targets"] for a in plan["amplitudes"]),
                      key=lambda k: (k[1], k[0]))
        for target, alpha, scene, inputs in experiment3.scenes(
                name, config, recipe, setup, argparse.Namespace(data=args.data), keys):
            key = {"set": BACKGROUNDS[name], "scene": name, "target": target, "alpha": alpha}
            curve = levels(setup[4], scene, inputs, rule.sigmas)
            w = omega(setup[4], scene, inputs, rule.sigmas[0], rng, spec)
            thresholds += [key | r for r in curve]
            rows.append(chosen_rows(key, curve, w, alpha, rules[target, alpha]))
            print(key, f"omega {w:.4f}", flush=True)
    return rows, thresholds


def experiment1_rows(args: argparse.Namespace, recipe: dict, registry: dict,
                     spec: dict) -> tuple[list[dict], list[dict]]:
    """(levels rows without "fixed", thresholds rows) of the Experiment 1 scenes."""
    config = json.loads((REPRO / "configs" / "experiment1.json").read_text())
    sigmas = noise_level.Rule.from_config(recipe, tuple(config["rule_amplitudes"])).sigmas
    rows, thresholds = [], []
    for name in config["scenes"]:
        scene = data.load(registry[name], args.data)
        inputs = whitening.prepare(scene, recipe["screen"])
        model = DARTS.load(REPRO / "checkpoints" / "experiment1" / f"{name}.pt")
        rule = experiment1.choice(scene, inputs, model, recipe, config)
        key = {"set": "Experiment 1", "scene": name, "target": 0,
               "alpha": registry[name].get("alpha", 0.0)}
        curve = levels(model, scene, inputs, sigmas)
        w = omega(model, scene, inputs, sigmas[0], np.random.default_rng(spec["seed"]), spec)
        thresholds += [key | r for r in curve]
        rows.append(chosen_rows(key, curve, w, registry[name].get("alpha"), rule))
        print(key, f"omega {w:.4f}", {k: rule[k] for k in ("sigma_hat", "sigma_u")}, flush=True)
    return rows, thresholds


def figure_data(args: argparse.Namespace, recipe: dict, registry: dict, spec: dict) -> dict:
    """Fig. 4's curves per background: the levels; the band's min and max (S - 1,) over the
    reference responses r_{theta,rho} of the offset-free amplitude grid and spec["band_rhos"]; and
    per amplitude of spec[background]: theta-hat, rho-hat, sigma-hat, the scene's response and the
    matched reference (S - 1,), each divided by its maximum over the levels after the first, the
    held-out PD averaged over [theta-hat / 2, theta-hat] and the scene's PD (S,)."""
    config = json.loads((REPRO / "configs" / "experiment3.json").read_text())
    out = {}
    for name, title in BACKGROUNDS.items():
        plan, show = config["backgrounds"][name], spec[name]
        _, pre, _, base, model = setup = experiment3.background(
            name, config, recipe, registry, argparse.Namespace(train=False, data=args.data))
        held = held_out(base.usable, config["fold"], recipe["training"]["folds"])[0]
        pool = noise_level.reference_pool(held, base.usable, torch.from_numpy(pre.exists))
        low = min(plan["amplitudes"])
        rule = noise_level.Rule.from_config(recipe, (low, max(plan["amplitudes"])))
        grid = dataclasses.replace(rule, rhos=tuple(spec["band_rhos"]), thetas=tuple(
            low * 2 ** ((k - 1) / 3) for k in range(len(rule.thetas))))
        scenes = list(experiment3.scenes(name, config, recipe, setup,
                                         argparse.Namespace(data=args.data),
                                         [(show["target"], a) for a in show["amplitudes"]]))
        signature = dataclasses.replace(base, t=scenes[0][3].t)
        reads = noise_level.held_out_reads(model, signature, pool, rule)
        band = noise_level.held_out_reads(model, signature, pool, grid)
        order = np.random.default_rng(100).permutation(len(band["pick"]))
        curves = np.array([normalized(noise_level.mix_response(band, i, rho, order, grid))
                           for i in range(len(grid.thetas)) for rho in grid.rhos])
        out[name] = {"title": title, "sigmas": list(rule.sigmas),
                     "band": [curves.min(0).tolist(), curves.max(0).tolist()],
                     "columns": [column(model, reads, alpha, scene, inputs, rule)
                                 for _, alpha, scene, inputs in scenes]}
    return out


def column(model: DARTS, reads: dict, alpha: float, scene: data.Scene,
           inputs: whitening.Inputs, rule: noise_level.Rule) -> dict:
    """The curves of Fig. 4 at one amplitude alpha (see `figure_data`)."""
    screened = torch.from_numpy(np.flatnonzero(scene.exists & ~inputs.usable.numpy()))
    chosen = noise_level.select(model, reads, inputs, screened, rule)
    out = {"alpha": alpha, **{k: chosen[k] for k in ("theta_hat", "rho_hat", "sigma_hat")},
           "response": normalized(chosen["response"]).tolist(),
           "reference": normalized(chosen["reference"]).tolist(),
           "band_pd": chosen["band_pd"].tolist(),
           "scene_pd": [r["pd"] for r in levels(model, scene, inputs, rule.sigmas)]}
    print({k: v for k, v in out.items() if not isinstance(v, list)}, flush=True)
    return out


def normalized(response: np.ndarray) -> np.ndarray:
    """A response over the levels after the first, divided by its maximum there."""
    return response[1:] / max(response[1:].max(), 1e-12)


def with_fixed(rows: list[dict]) -> list[dict]:
    """levels.csv rows: each scene's chosen levels and their PD, "fixed" the level with the best
    mean PD over the scenes of its signature."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["set"], r["scene"], r["target"]].append(r)
    out = []
    for r in rows:
        group = groups[r["set"], r["scene"], r["target"]]
        fixed = max(r["curve"], key=lambda s: np.mean([g["curve"][s] for g in group]))
        sigma = {name: fixed if name == "fixed" else r[f"sigma_{name}"] for name in CHOICES}
        out.append({k: v for k, v in r.items() if k != "curve" and not k.startswith("sigma_")}
                   | {f"sigma_{n}": f"{s:.4f}" for n, s in sigma.items()}
                   | {f"pd_{n}": f"{r['curve'][s]:.4f}" for n, s in sigma.items()})
    return out


def report(rows: list[dict], near: float) -> None:
    """Mean PD of each choice per set, its mean and worst regret against the best level and how
    many scenes are within `near` of it (on the 4-decimal PDs of levels.csv); and the Experiment 3
    scenes where theta-hat exceeds the amplitude by more than half an octave."""
    for name in dict.fromkeys(r["set"] for r in rows):
        group = [r for r in rows if r["set"] == name]
        best = np.array([float(r["pd_best"]) for r in group])
        print(f"{name}: {len(group)} scenes")
        for choice in CHOICES:
            regret = np.round(best - np.array([float(r[f"pd_{choice}"]) for r in group]), 4)
            print(f"  {choice:15s} mean PD {np.mean(best - regret):.3f}  regret mean "
                  f"{regret.mean():.3f} worst {regret.max():.3f}  within {near}: "
                  f"{int((regret <= near).sum())}/{len(group)}")
    over = [r for r in rows if r["set"] != "Experiment 1"
            and math.log2(r["theta_hat"] / r["alpha"]) > 0.5 + 1e-9]
    loss = max((float(r["pd_best"]) - float(r["pd_rule"]) for r in over), default=0.0)
    print(f"theta-hat more than half an octave above alpha: {len(over)} scenes, at amplitudes "
          f"{sorted({(r['set'], r['alpha']) for r in over})}; worst loss of sigma-hat {loss:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    parser.add_argument("--results", type=Path, default=REPRO / "results" / "experiment3",
                        help="Experiment 3's rule.csv files, from the shipped models")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "experiment4")
    args = parser.parse_args()
    if (args.results / "checkpoints").exists():
        parser.error(f"{args.results} holds retrained models; Experiment 4 scores the shipped "
                     "ones, so pass the rule files of a run without --train")
    recipe = json.loads((REPRO / "configs" / "darts.json").read_text())
    config = json.loads((REPRO / "configs" / "experiment4.json").read_text())
    registry = json.loads((REPRO / "configs" / "scenes.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows3, thresholds3 = experiment3_rows(args, recipe, registry, config["omega"])
    rows1, thresholds1 = experiment1_rows(args, recipe, registry, config["omega"])
    rows = with_fixed(rows3 + rows1)
    write(args.out / "thresholds.csv", thresholds3 + thresholds1)
    write(args.out / "levels.csv", rows)
    figure = figure_data(args, recipe, registry, config["figure"])
    (args.out / "figure4.json").write_text(json.dumps(figure))
    report(rows, config["near_best"])


if __name__ == "__main__":
    main()
