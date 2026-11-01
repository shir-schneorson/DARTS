"""Experiment 3: the Rao test and the GLRT on a synthetic spatial scene, against the exact scene
density and the exact context.

Per seed, both learned models are fitted once on the target-free scene (`spatial_scene`), whitened
by its own fit, on its pixels minus its one-sided AMF screen: DART (`experiments.dart`, the score
head without context) and DARTS (fold `seed` of configs/darts.json), whose noise level comes from
the rule on that fold's reads. Each implanted scene is whitened by the same fit, so t = W mu_t
exactly, and DARTS's contexts leave out the implanted scene's own screen. Rows per amplitude:

    oracle/rao, oracle/glrt     the exact scene density
    oracle_context/rao          the exact density given each pixel's region
    dart/rao, dart/glrt         DART at the recipe's sigma
    darts/rao, darts/glrt       DARTS at sigma_hat

Writes <out>/seed<s>.csv, which `figures/figure3.py` draws.

Usage: python -m experiments.experiment3 [--device DEVICE] [--checkpoints DIR] [--out DIR]
                                         [seed ...]
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dartsdet import evaluation, noise_level, pipeline, preprocessing
from dartsdet.model import ContextFreeScore, Standardize, checked, save_atomically
from dartsdet.reads import contexts, glrt, rao
from experiments import dart
from experiments import spatial_scene as sp
from experiments.cli import REPRO, arguments

FIELDS = ("row", "alpha", "seed", "sigma", "auc", "pd", "pauc")


def dart_model(fx: torch.Tensor, seed: int, config: dict, device: str,
               path: Path) -> ContextFreeScore:
    """DART on the whitened kept spectra fx (n, B): loaded from `path` if it is there, refused
    unless it was fitted with this seed, recipe, scene settings and number of spectra; else
    fitted and saved. The scene settings stand for the training data, which they fix up to the
    platform's arithmetic, so a checkpoint stays valid across machines."""
    spec = config["dart"]
    scene = {k: config[k] for k in ("scene", "background", "target", "screen_fraction")}
    record = {"seed": seed, "spec": spec, "scene": scene, "n": len(fx)}
    if path.exists():
        saved = checked(torch.load(path, weights_only=True), record, path)
        model = ContextFreeScore(Standardize(torch.zeros(fx.shape[1]), torch.ones(fx.shape[1])),
                                 spec["width"], spec["depth"], spec["network"])
        model.load_state_dict(saved["state"])
        return model
    model, epochs = dart.fit(fx.to(device, torch.float32), seed, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_atomically({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                     "stop_epoch": epochs} | record, path)
    return model.cpu()


def models(scene: dict, config: dict, seed: int, args: argparse.Namespace) -> dict:
    """The seed's whitening, DARTS model and sigma_hat, and DART in float64 and float32, all from
    the target-free scene. Its pixel vectors are row-major, so the encoder sees the scene
    transposed, which is immaterial for this square, isotropic scene."""
    x = scene["background"]
    fitted = preprocessing.whitening(x)
    keep = ~preprocessing.screened(x.numpy(), scene["target_mean"].numpy(),
                                   config["screen_fraction"])
    inputs = preprocessing.Inputs((x - fitted[0]) @ fitted[1], torch.from_numpy(keep),
                                  scene["target_mean"] @ fitted[1], (scene["side"],) * 2)
    spec = json.loads((REPRO / "configs" / "darts.json").read_text())
    folder = args.checkpoints / "experiment3" / f"seed{seed}"
    sigmas = noise_level.grid(spec["sigma_max"])
    darts = pipeline.fold_model(inputs, spec, seed, args.device, folder / "darts.pt")
    reads = pipeline.rule_reads(darts, inputs, np.ones(len(keep), dtype=bool), seed, sigmas,
                                args.device)
    fit = dart_model(inputs.fx[keep], seed, config, args.device, folder / "dart.pt")
    fit = fit.eval().requires_grad_(False)
    sigma = pipeline.sigma_hat([reads], sigmas)
    return {"fitted": fitted, "darts": darts, "sigma": sigma,
            "dart64": copy.deepcopy(fit).double(), "dart32": fit.float()}


def learned_rows(m: dict, x: torch.Tensor, keep: np.ndarray, side: int, t: torch.Tensor,
                 config: dict, device: str) -> dict[str, tuple[float, np.ndarray]]:
    """(sigma, statistic (P,)) of the DART and DARTS rows on one implanted scene: DART's Rao in
    float64 on the CPU, its GLRT in float32; DARTS's on `device`."""
    mean, matrix = m["fitted"]
    fx, t_w = (x - mean) @ matrix, t @ matrix
    grid = (config["glrt"]["theta_max"], config["glrt"]["n_theta"])
    t_hat, t_norm = t_w / t_w.norm(), float(t_w.norm())
    s_dart, s_darts = config["dart"]["sigma"], m["sigma"]
    z64, z32 = m["dart64"].coordinates(fx), m["dart32"].coordinates(fx.float())
    empty = torch.zeros(len(x), 0)
    zc, ctx = contexts(m["darts"], fx.float().to(device), torch.from_numpy(keep).to(device),
                       (side, side))
    t32 = t_hat.float().to(device)
    with torch.no_grad():
        dart_rao = rao(m["dart64"], z64, empty.double(), t_hat, s_dart).numpy()
        darts_rao = rao(m["darts"], zc, ctx, t32, s_darts).cpu().double().numpy()
    return {"dart/rao": (s_dart, dart_rao),
            "dart/glrt": (s_dart, glrt(m["dart32"], z32, empty, t_hat.float(), t_norm, s_dart,
                                       grid).double().numpy()),
            "darts/rao": (s_darts, darts_rao),
            "darts/glrt": (s_darts, glrt(m["darts"], zc, ctx, t32, t_norm, s_darts,
                                         grid).double().numpy())}


def amplitude_rows(scene: dict, m: dict, alpha: float, config: dict, device: str) -> list[dict]:
    """Every row of one amplitude."""
    x, labels = sp.implanted(scene, alpha)
    t = scene["target_mean"]
    keep = ~preprocessing.screened(x.numpy(), t.numpy(), config["screen_fraction"])
    glrt_grid = (config["glrt"]["theta_max"], config["glrt"]["n_theta"])
    stats = {"oracle/rao": (0.0, sp.oracle_rao(scene["oracle"], x, t).numpy()),
             "oracle/glrt": (0.0, sp.oracle_glrt(scene["oracle"], x, t, *glrt_grid).numpy()),
             "oracle_context/rao": (0.0, sp.oracle_context(scene, x, t).numpy())}
    stats |= learned_rows(m, x, keep, scene["side"], t, config, device)
    return [{"row": row, "alpha": alpha, "seed": scene["seed"], "sigma": sigma,
             **evaluation.detection_metrics(labels, stat)} for row, (sigma, stat) in stats.items()]


def main() -> None:
    parser = arguments(__doc__, "experiment3", device=True, data=False, checkpoints=True)
    config = json.loads((REPRO / "configs" / "experiment3.json").read_text())
    parser.add_argument("seeds", nargs="*", type=int, default=list(range(config["n_seeds"])),
                        help="scene seeds to run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        scene = sp.draw_scene(config, seed) | {"seed": seed}
        m = models(scene, config, seed, args)
        print(f"seed {seed}: sigma_hat {m['sigma']:.4g}", flush=True)
        rows = [r for alpha in config["alphas"]
                for r in amplitude_rows(scene, m, alpha, config, args.device)]
        with (args.out / f"seed{seed}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
