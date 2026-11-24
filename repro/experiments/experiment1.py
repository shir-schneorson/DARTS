"""Experiment 1 (Table II and Fig. 1): DARTS and the baselines on four scenes.

San Diego and ABU-beach-4 contain real targets; Pavia University receives class 9 at amplitude
0.75 and MUUFL signature 0 at 0.1 (configs/scenes.json). Each scene is whitened and screened on
its own (configs/darts.json); configs/experiment1.json sets the rest. Rows, per scene:
- AMF, ACE, CEM, and the local AMF and ACE (`baselines.classical`);
- GMM-Rao and GMM-GLRT, with global and window weights (`baselines.gmm`), fitted on the
  unscreened pixels;
- DARTS Rao at the rule's sigma-hat, and Table II's reference row, DARTS GLRT at sigma-hat_G, the
  level the rule's second stage picks from the GLRT's held-out PD (`noise_level.held_out_glrt`):
  fold 0 of the recipe, from checkpoints/experiment1/<scene>.pt or trained anew with --train.
The detectors run with their authors' code are stored rows (results/experiment1/baselines.csv).

The statistics are computed on the CPU, where the paper's numbers come from. A row's time is the
mean of `timing_repeats` computations of its map from the fitted or trained model, after one
warm-up: on --device for DARTS, on the CPU otherwise.

Writes <out>/detectors.csv (scene, detector, sigma, auc, pd, pauc, time) and <out>/rule.csv
(scene, sigma_hat, theta_hat, rho_hat, sigma_hat_g); with --train, also the models in
<out>/checkpoints/.
Usage, from repro/: python -m experiments.experiment1 [--train] [--device DEVICE] [--data DIR]
[--out DIR]
"""

import argparse
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from baselines import gmm
from baselines.classical import ace, cem, local_amf_ace, window_batches
from dartsdet import data, noise_level, statistics, whitening
from dartsdet.metrics import detection_metrics
from dartsdet.model import DARTS, pixels
from dartsdet.training import held_out, train
from experiments import default_device, write

REPRO = Path(__file__).resolve().parents[1]


def timed(compute: Callable[[], object], repeats: int) -> tuple[object, float]:
    """compute()'s output and its mean seconds over `repeats` calls after one warm-up call;
    compute must return CPU values, so that each call has finished."""
    out = compute()
    start = time.perf_counter()
    for _ in range(repeats):
        compute()
    return out, (time.perf_counter() - start) / repeats


def row(scene: data.Scene, detector: str, score: np.ndarray, seconds: float | None = None,
        sigma: float | None = None) -> dict:
    """One output row: the metrics of a map (P,) on the scene's labels, to 6 decimals, so that
    Table II rounds each value once."""
    metrics = detection_metrics(scene.labels, score)
    return {"detector": detector, "sigma": "" if sigma is None else f"{sigma:.4f}",
            **{k: f"{v:.6f}" for k, v in metrics.items()},
            "time": "" if seconds is None else f"{seconds:.3f}"}


def classical_rows(scene: data.Scene, inputs: whitening.Inputs, config: dict) -> list[dict]:
    """AMF, ACE, CEM and the local AMF and ACE; the local pair shares one computation and time."""
    exists, repeats = scene.exists, config["timing_repeats"]
    window = config["local_window"]
    need = window["pixels_per_band"] * scene.cube.shape[2]

    def full(compute: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        out = np.zeros(len(exists))
        out[exists] = compute(inputs.fx[torch.from_numpy(exists)].numpy(), inputs.t.numpy())
        return out

    rows = [row(scene, name, *timed(lambda f=f: full(f), repeats)) for name, f in
            (("AMF", whitening.amf), ("ACE", ace),
             ("CEM", lambda *_: cem(scene.spectra()[exists], scene.target)))]
    (local_amf, local_ace), seconds = timed(lambda: local_amf_ace(
        inputs.fx, inputs.usable, inputs.t, window_batches(
            inputs.usable, torch.from_numpy(exists), inputs.grid, window["guard"], need)), repeats)
    return rows + [row(scene, "Local AMF", local_amf, seconds),
                   row(scene, "Local ACE", local_ace, seconds)]


def gmm_rows(scene: data.Scene, inputs: whitening.Inputs, components: int,
             config: dict) -> list[dict]:
    """GMM-Rao and GMM-GLRT with global and window weights, the mixture fitted on the unscreened
    pixels; each map timed from the fitted mixture."""
    settings, window = config["gmm"], config["local_window"]
    model = gmm.fit(inputs.fx[inputs.usable], components, settings["seed"], settings["reg_covar"])
    need = window["pixels_per_band"] * scene.cube.shape[2]

    def score(test: Callable, spatial: bool) -> np.ndarray:
        terms, weights = gmm.line(model, inputs.fx, inputs.t), model.weights.double().log()
        if spatial:
            batches = window_batches(inputs.usable, torch.from_numpy(scene.exists), inputs.grid,
                                     window["guard"], need)
            weights = gmm.window_weights(weights, terms, inputs.usable, batches)
        return test(weights, terms).numpy()

    return [row(scene, name, *timed(lambda t=test, s=spatial: score(t, s),
                                    config["timing_repeats"]))
            for name, test, spatial in (("GMM-Rao", gmm.rao, False), ("GMM-GLRT", gmm.glrt, False),
                                        ("GMM-Rao, window", gmm.rao, True),
                                        ("GMM-GLRT, window", gmm.glrt, True))]


def choice(scene: data.Scene, inputs: whitening.Inputs, model: DARTS, recipe: dict,
           config: dict) -> dict:
    """The rule's choice on a scene (`noise_level.select`): held-out reads on the model's held-out
    fold, the scene's screened pixels as the pool."""
    rule = noise_level.Rule.from_config(recipe, tuple(config["rule_amplitudes"]))
    held = held_out(inputs.usable, config["fold"], recipe["training"]["folds"])[0]
    pool = torch.from_numpy(np.flatnonzero(scene.exists & ~inputs.usable.numpy()))
    reads = noise_level.held_out_reads(model, inputs, held, rule)
    return noise_level.select(model, reads, inputs, pool, rule)


def glrt_level(inputs: whitening.Inputs, model: DARTS, recipe: dict, config: dict,
               theta_hat: float, amplitudes: torch.Tensor) -> float:
    """sigma-hat_G: the rule's second stage, eq. (sigma-hat), on the plug-in GLRT's held-out PD
    over [theta-hat / 2, theta-hat], theta-hat the rule's own."""
    rule = noise_level.Rule.from_config(recipe, tuple(config["rule_amplitudes"]))
    held = held_out(inputs.usable, config["fold"], recipe["training"]["folds"])[0]
    pd = noise_level.held_out_glrt(model, inputs, held, rule, amplitudes)
    i = int(np.argmin(np.abs(np.asarray(rule.thetas) - theta_hat)))
    return rule.sigmas[int(np.argmax(noise_level.band_average(pd, i, rule)))]


def darts_rows(scene: data.Scene, inputs: whitening.Inputs, model: DARTS, recipe: dict,
               config: dict, device: str) -> tuple[list[dict], dict]:
    """(DARTS Rao at sigma-hat and DARTS GLRT at sigma-hat_G, each timed on `device`; the rule's
    choice with sigma_hat_g)."""
    chosen = choice(scene, inputs, model, recipe, config)
    grid = recipe["glrt_amplitudes"]
    amplitudes = torch.logspace(grid["log10_first"], grid["log10_last"], grid["count"])
    chosen["sigma_hat_g"] = glrt_level(inputs, model, recipe, config, chosen["theta_hat"],
                                       amplitudes)
    on_device = copy.deepcopy(model).to(device)
    zd, md = on_device.image(inputs.fx, inputs.usable, inputs.grid)
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    rows = []
    for name, sigma, test in (("DARTS Rao", chosen["sigma_hat"], statistics.rao),
                              ("DARTS GLRT", chosen["sigma_hat_g"], statistics.glrt)):
        extra = (amplitudes,) if test is statistics.glrt else ()
        _, seconds = timed(lambda s=sigma, f=test, e=extra: f(
            on_device, pixels(zd), statistics.contexts(on_device, zd, md, s), inputs.t, s,
            *e).cpu(), config["timing_repeats"])
        score = test(model, pixels(z), statistics.contexts(model, z, mask, sigma), inputs.t,
                     sigma, *extra)
        rows.append(row(scene, name, score.double().numpy(), seconds, sigma))
    return rows, chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="retrain DARTS instead of loading")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--data", type=Path, default=REPRO.parent / "data")
    parser.add_argument("--out", type=Path, default=REPRO / "outputs" / "experiment1")
    args = parser.parse_args()
    recipe = json.loads((REPRO / "configs" / "darts.json").read_text())
    config = json.loads((REPRO / "configs" / "experiment1.json").read_text())
    registry = json.loads((REPRO / "configs" / "scenes.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows, rules = [], []
    for name in config["scenes"]:
        start = time.time()
        scene = data.load(registry[name], args.data)
        inputs = whitening.prepare(scene, recipe["screen"])
        if args.train:
            model, record = train(inputs, recipe, config["learning_rate"][name], config["fold"],
                                  args.device)
            (args.out / "checkpoints").mkdir(exist_ok=True)
            model.cpu().save(args.out / "checkpoints" / f"{name}.pt", **record)
        else:
            model = DARTS.load(REPRO / "checkpoints" / "experiment1" / f"{name}.pt")
        darts, chosen = darts_rows(scene, inputs, model, recipe, config, args.device)
        here = (classical_rows(scene, inputs, config)
                + gmm_rows(scene, inputs, config["gmm"]["components"][name], config) + darts)
        rows += [{"scene": name} | r for r in here]
        rules.append({"scene": name, **{k: f"{chosen[k]:.4f}" for k in
                                        ("sigma_hat", "theta_hat", "rho_hat", "sigma_hat_g")}})
        for r in here:
            print(name, " ".join(f"{k} {v}" for k, v in r.items() if v != ""), flush=True)
        print(name, rules[-1], f"{time.time() - start:.0f} s", flush=True)
        write(args.out / "detectors.csv", rows)
        write(args.out / "rule.csv", rules)


if __name__ == "__main__":
    main()
