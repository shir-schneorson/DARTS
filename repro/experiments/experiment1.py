"""Experiment 1: are tractable background families misspecified on the hyperspectral scenes?

For every scene, Gaussian mixtures with K of `gmm_components` (K = 1 is the Gaussian) and Student-t
mixtures with K of `student_components` are fitted by maximum likelihood on one random half of
the scene's background pixels and tested on the other half; then the halves swap and the counts
are pooled. The test (`calibration`): along the whitened target direction and `n_directions`
random whitened directions of the training half, the share of held-out pixels in each tail of each
fitted 1-D marginal, against the tail's probability. The reference repeats the protocol on draws
from the model fitted to the first half, as many as the scene has background pixels: the test
where the family is exactly right, at the scene's sample size and band count.

Background: the pixels labelled 0, or on MUUFL those outside every cloth panel's halo; there is no
screen. The spectra are raw and t = d - m (MUUFL: lab signature 0). Writes <out>/runs.csv, one line
per (scene, data, family, K), data "scene" or "reference": the mean train and held-out
log-likelihood per pixel, per half whether the fit converged and, for the t mixtures, its dof, and
per tail p and side the held-out share over p along t (`*_t_*`), its median over the random
directions (`*_median_*`) and the share of random directions the binomial test rejects at
`level` (`*_rejected_*`). `figures/figure1.py` draws it.

Usage: python -m experiments.experiment1 [--data DIR] [--out DIR] [scene ...]
"""

import csv
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from dartsdet import muufl, scenes
from experiments.calibration import binomial_rejected, projections, tail_counts
from experiments.cli import REPRO, arguments
from experiments.gmm import GMM, cholesky_whitening
from experiments.student_mixture import StudentMixture

Model = GMM | StudentMixture
SIDES = ("upper", "lower")


def background(entry: dict, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(background spectra (N, B), signature t = d - m (B,)) of a registry entry, float64."""
    if entry["kind"] == "mat":
        scene = scenes.real_targets(data_dir / entry["file"])
        spectra = scene.spectra()
        keep, d = scene.labels == 0, scene.target
    else:
        campus = muufl.campus(data_dir / entry["collection"], entry["scene"], entry["columns"])
        spectra = campus["cube"].reshape(-1, campus["cube"].shape[2], order="F")
        halos = muufl.halo_mask(campus["cube"].shape[:2], campus["row"], campus["col"],
                                campus["size"])
        keep, d = ~scenes.fortran(halos), campus["signatures"][:, entry["target"]]
    return torch.from_numpy(spectra[keep]), torch.from_numpy(d - spectra.mean(axis=0))


def fitter(family: str, k: int, config: dict) -> Callable[[torch.Tensor], Model]:
    """The family's maximum-likelihood fit with K = k."""
    if family == "gmm":
        return lambda x: GMM.fit(x, k, config["fit_seed"], config["reg_covar"])
    grid = torch.tensor(config["dof_grid"], dtype=torch.float64)
    return lambda x: StudentMixture.fit(x, k, config["fit_seed"], config["reg_covar"], grid)


def draws(model: Model, n: int, seed: int) -> torch.Tensor:
    """n draws (n, B) from a fitted model, from torch's global generator seeded with `seed`."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if isinstance(model, GMM):
            z = model.distribution().sample((n,))
            return model.shift + z @ torch.linalg.inv(model.unmix).T
        components = torch.multinomial(model.weights, n, replacement=True)
        dof = torch.tensor(model.dof, dtype=torch.float64)
        scales = torch.distributions.Chi2(dof).sample((n,))
    return model.sample(components, scales / model.dof, torch.Generator().manual_seed(seed))


def two_fold(x: torch.Tensor, t: torch.Tensor, fit: Callable[[torch.Tensor], Model],
             config: dict) -> tuple[dict, Model]:
    """Fit on each half, test on the other: (pooled counts, mean log-likelihoods, and per half
    the dof and whether the fit converged; the model fitted to the first half)."""
    halves = torch.from_numpy(np.random.default_rng(config["split_seed"]).permutation(len(x)))
    counts, ll_train, ll_test, dofs, converged, first = {}, 0.0, 0.0, [], [], None
    for train, test in zip(halves.chunk(2), reversed(halves.chunk(2)), strict=True):
        model = fit(x[train])
        first = model if first is None else first
        a = projections(t, cholesky_whitening(x[train])[1], config["n_directions"],
                        config["direction_seed"])
        for key, c in tail_counts(*model.marginal_cdf(x[test], a), config["tails"]).items():
            counts[key] = counts.get(key, 0) + c
        ll_train += float(model.log_prob(x[train]).sum())
        ll_test += float(model.log_prob(x[test]).sum())
        dofs.append(getattr(model, "dof", None))
        converged.append(model.converged)
    return {"counts": counts, "ll_train": ll_train / len(x), "ll_test": ll_test / len(x),
            "dofs": dofs, "converged": converged}, first


def record(result: dict, n: int, config: dict) -> dict:
    """The runs.csv columns of a `two_fold` result over n pixels."""
    out = {"ll_train": result["ll_train"], "ll_test": result["ll_test"],
           "converged": "/".join("yes" if c else "no" for c in result["converged"]),
           "dof": "/".join(f"{d:g}" for d in result["dofs"]) if result["dofs"][0] else ""}
    for (side, p), c in result["counts"].items():
        out[f"{side}_t_{p:g}"] = c[0] / (n * p)
        out[f"{side}_median_{p:g}"] = float(np.median(c[1:])) / (n * p)
        out[f"{side}_rejected_{p:g}"] = float(binomial_rejected(c[1:], n, p,
                                                                config["level"]).mean())
    return out


def scene_rows(name: str, x: torch.Tensor, t: torch.Tensor, config: dict) -> list[dict]:
    """Every (data, family, K) line of one scene."""
    lines = []
    for family, ks in (("gmm", config["gmm_components"]),
                       ("student", config["student_components"])):
        for k in ks:
            result, model = two_fold(x, t, fitter(family, k, config), config)
            reference, _ = two_fold(draws(model, len(x), config["reference_seed"]), t,
                                    fitter(family, k, config), config)
            for data, res in (("scene", result), ("reference", reference)):
                lines.append({"scene": name, "data": data, "family": family, "components": k,
                              "pixels": len(x), "bands": x.shape[1],
                              **record(res, len(x), config)})
            print(name, family, k, f"held-out log-likelihood {result['ll_test']:.2f}",
                  flush=True)
    return lines


def main() -> None:
    config = json.loads((REPRO / "configs" / "experiment1.json").read_text())
    parser = arguments(__doc__, "experiment1", device=False, data=True, checkpoints=False)
    parser.add_argument("scenes", nargs="*", default=config["scenes"], help="scenes to run")
    args = parser.parse_args()
    if unknown := sorted(set(args.scenes) - set(config["scenes"])):
        parser.error(f"unknown scenes {unknown}; choose from {config['scenes']}")
    registry = json.loads((REPRO / "configs" / "scenes.json").read_text())
    columns = ["scene", "data", "family", "components", "pixels", "bands", "ll_train", "ll_test",
               "converged", "dof", *(f"{side}_{kind}_{p:g}" for p in config["tails"]
                                     for side in SIDES for kind in ("t", "median", "rejected"))]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "runs.csv"
    with out.with_suffix(".partial").open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for name in args.scenes:
            x, t = background(registry[name], args.data)
            writer.writerows(scene_rows(name, x, t, config))
    os.replace(out.with_suffix(".partial"), out)


if __name__ == "__main__":
    main()
