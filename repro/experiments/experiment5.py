"""Experiment 5: the noise-level rule with an exact score.

The background is `exact_background`'s, whose smoothed score is exact. The learned models are
context-free DSM fits of the DARTS score head (`ContextFreeScore`, width 128, depth 3), each on n
background samples with a fifth held out for early stopping; the held-out fifth then feeds the
rule. For every noise level of `SIGMAS`, for the exact score and for each model:

- the likelihood-ratio rate r(sigma) / alpha on the held-out samples, each moved by one draw of
  smoothing noise;
- the share of clean held-out samples at which -log p_sigma is not convex along t (C <= 0);
- the Rao test's PD at PF = 1e-2 against targets x = b + THETA t, thresholded at the exact
  (1 - PF) quantile of the statistic on a fresh null sample.

sigma_hat = max(sigma_LR, sigma_cvx) with beta = PF: the scene has no screen, so no plateau part.
Writes <out>/curves.npz, which `figures/figure5.py` draws.

Usage: python -m experiments.experiment5 [--checkpoints DIR] [--out DIR]
"""

import math
from pathlib import Path

import numpy as np
import torch

from dartsdet import noise_level
from dartsdet.model import ContextFreeScore, Standardize, checked, save_atomically
from dartsdet.reads import log_ratio, rao, rao_and_curvature
from dartsdet.training import dsm_loss, optimization
from experiments import exact_background as eb
from experiments.cli import arguments

PF, THETA = 1e-2, 2.5
SIGMAS = np.array([0.005 * 200 ** (k / 24) for k in range(31)])
NS, SEEDS = (2000, 8000, 32000, 128000), (0, 1, 2)
HEAD = {"activation": "GELU", "embed_dim": 64, "fourier_scale": 0.25}
TRAIN = {"lr": 1e-3, "max_steps": 30000, "batch": 1024, "eval_every": 200, "patience": 15,
         "training": {"weight_decay": 1e-4, "warmup": 500, "clip": 1.0, "ema": 0.999}}


def noise_levels(n: int, generator: torch.Generator) -> torch.Tensor:
    """n training noise levels, log-uniform on [0.005, 4]; written as exp(log a + u log(b / a)),
    not `training.log_uniform`'s a (b / a)^u, whose rounding differs."""
    span = math.log(4) - math.log(0.005)
    return torch.exp(math.log(0.005) + span * torch.rand(n, generator=generator))


def fit(x: np.ndarray, seed: int) -> tuple[ContextFreeScore, torch.Tensor]:
    """(the EMA state with the lowest held-out loss, the held-out rows) of a fit on x (n, B)."""
    generator = torch.Generator().manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    perm = torch.randperm(len(xt), generator=generator)
    val, kept = perm[: len(xt) // 5], perm[len(xt) // 5:]
    coordinates = Standardize(xt[kept].mean(0), xt[kept].std(0))  # unbiased, as the paper's fits
    z, empty = coordinates(xt), torch.zeros(len(xt), 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ContextFreeScore(coordinates, 128, 3, HEAD)
    optimizer, scheduler, ema = optimization(model, TRAIN)
    val_sigma = noise_levels(len(val), generator)
    val_noise = torch.randn(len(val), eb.B, generator=generator)
    best, best_step, state = math.inf, 0, None
    for step in range(1, TRAIN["max_steps"] + 1):
        rows = kept[torch.randint(len(kept), (TRAIN["batch"],), generator=generator)]
        sigma = noise_levels(TRAIN["batch"], generator)
        noise = torch.randn(TRAIN["batch"], eb.B, generator=generator)
        loss = dsm_loss(model, z[rows], empty[rows], sigma, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["training"]["clip"])
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        if step % TRAIN["eval_every"]:
            continue
        with torch.no_grad():
            current = float(dsm_loss(ema.module, z[val], empty[val], val_sigma, val_noise))
        if current < best:
            best, best_step = current, step
            state = {k: v.clone() for k, v in ema.module.state_dict().items()}
        if (step - best_step) // TRAIN["eval_every"] >= TRAIN["patience"]:
            break
    if state is None:
        raise RuntimeError(f"n {len(x)} seed {seed}: no finite validation loss in {step} steps")
    model.load_state_dict(state)
    print(f"[fit] n {len(x)} seed {seed}: best step {best_step}", flush=True)
    return model.eval().requires_grad_(False), val


def rao_power(null: np.ndarray, alternative: np.ndarray) -> float:
    """PD at exact size PF: the share of the alternative above the null's (1 - PF) quantile."""
    return float(np.mean(alternative > np.quantile(null, 1 - PF)))


def rule(ell: np.ndarray, C: np.ndarray) -> float:
    """sigma_hat from the held-out log ratios (n, S, len(LR_THETAS)) and clean curvatures
    (n, S)."""
    return max(noise_level.lr_sigma(ell, SIGMAS),
               noise_level.convex_sigma(C, SIGMAS, beta=PF))


def exact_curves(fs: list, q: np.ndarray, data: dict) -> dict[str, np.ndarray | float]:
    """The exact score's curves, sigma_hat and the PD there; data: source coordinates `null`,
    `alt` and `cal` (calibration samples), and the calibration samples' noise `noise`."""
    pd, C, ell = [], [], []
    for sigma in SIGMAS:
        psi, dpsi = eb.scores(fs, sigma)
        pd.append(rao_power(eb.exact_reads(data["null"], q, psi, dpsi)[0],
                            eb.exact_reads(data["alt"], q, psi, dpsi)[0]))
        C.append(eb.exact_reads(data["cal"], q, psi, dpsi)[1])
        ell.append(eb.exact_log_ratio(data["cal"] + sigma * data["noise"], q, fs, sigma,
                                      noise_level.LR_THETAS))
    C, ell = np.stack(C, 1), np.stack(ell, 1)
    sigma_hat = rule(ell, C)
    psi, dpsi = eb.scores(fs, sigma_hat)
    at = rao_power(eb.exact_reads(data["null"], q, psi, dpsi)[0],
                   eb.exact_reads(data["alt"], q, psi, dpsi)[0])
    return {"lr_rate": noise_level.lr_rate(ell), "share": np.mean(C <= 0, 0), "pd": np.array(pd),
            "sigma": sigma_hat, "pd_at_sigma": at}


def learned_curves(model: ContextFreeScore, held: np.ndarray, seed: int, x: dict,
                   t_hat: torch.Tensor) -> dict[str, np.ndarray | float]:
    """A model's curves, sigma_hat and the PD there; held: its held-out samples (n, B); x: the
    `null` and `alt` samples in x coordinates; held-out noise drawn with seed 1000 + seed."""
    z = {k: model.coordinates(torch.as_tensor(v, dtype=torch.float32))
         for k, v in (("held", held), ("null", x["null"]), ("alt", x["alt"]))}
    empty = {k: torch.zeros(len(v), 0) for k, v in z.items()}
    eps = torch.randn(len(held), eb.B, generator=torch.Generator().manual_seed(1000 + seed))

    def power(sigma: float) -> float:
        with torch.no_grad():
            A0, A1 = (rao(model, z[k], empty[k], t_hat, sigma).double().numpy()
                      for k in ("null", "alt"))
        return rao_power(A0, A1)

    _, C = rao_and_curvature(model, z["held"], empty["held"], t_hat, list(SIGMAS))
    ell = np.stack([log_ratio(model, z["held"] + s * eps, empty["held"], t_hat, s,
                              noise_level.LR_THETAS).double().numpy() for s in SIGMAS], 1)
    C = C.double().numpy()
    sigma_hat = rule(ell, C)
    return {"lr_rate": noise_level.lr_rate(ell), "share": np.mean(C <= 0, 0),
            "pd": np.array([power(s) for s in SIGMAS]), "sigma": sigma_hat,
            "pd_at_sigma": power(sigma_hat)}


def scene() -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
    """(sources, rotation Q, unit signature t_hat, q = Q' t_hat, samples), drawn in a fixed order:
    `null` (200,000), `alt` (20,000 background plus THETA q) and `cal` (40,000) in source
    coordinates, and `noise`, the calibration samples' smoothing noise."""
    rng = np.random.default_rng(0)
    fs = eb.sources()
    Q = np.linalg.qr(rng.standard_normal((eb.B, eb.B)))[0]
    t_hat = rng.standard_normal(eb.B)
    t_hat /= np.linalg.norm(t_hat)
    q = Q.T @ t_hat
    null, background = eb.sample(fs, 200000, rng), eb.sample(fs, 20000, rng)
    data = {"null": null, "alt": background + THETA * q, "cal": eb.sample(fs, 40000, rng)}
    data["noise"] = np.random.default_rng(1000).standard_normal(data["cal"].shape)
    return fs, Q, t_hat, q, data


def trained(n: int, seed: int, fs: list, Q: np.ndarray,
            path: Path) -> tuple[ContextFreeScore, np.ndarray]:
    """(model, its held-out samples) for n training samples drawn with seed 100 + seed; loaded
    from `path` if it is there, else fitted and saved there."""
    x = eb.sample(fs, n, np.random.default_rng(100 + seed)) @ Q.T
    record = {"seed": seed, "n": n, "head": HEAD, "train": TRAIN}
    if path.exists():
        saved = checked(torch.load(path, weights_only=True), record, path)
        model = ContextFreeScore(Standardize(torch.zeros(eb.B), torch.ones(eb.B)), 128, 3, HEAD)
        model.load_state_dict(saved["state"])
        val = saved["val"]
    else:
        model, val = fit(x, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_atomically({"state": model.state_dict(), "val": val} | record, path)
    return model.eval().requires_grad_(False), x[val.numpy()]


def main() -> None:
    parser = arguments(__doc__, "experiment5", device=False, data=False, checkpoints=True)
    args = parser.parse_args()
    torch.set_num_threads(4)  # as the paper's fits ran; the thread count moves float reductions
    fs, Q, t_hat, q, data = scene()
    out = {"sigmas": SIGMAS, "pf": PF} | {f"exact_{k}": v
                                          for k, v in exact_curves(fs, q, data).items()}
    x = {"null": data["null"] @ Q.T, "alt": data["alt"] @ Q.T}
    t = torch.as_tensor(t_hat, dtype=torch.float32)
    for n in NS:
        for seed in SEEDS:
            path = args.checkpoints / "experiment5" / f"n{n}_s{seed}.pt"
            model, held = trained(n, seed, fs, Q, path)
            curves = learned_curves(model, held, seed, x, t)
            out |= {f"n{n}_s{seed}_{k}": v for k, v in curves.items()}
            print(f"n {n} seed {seed}: sigma_hat {curves['sigma']:.3g}, "
                  f"PD {curves['pd_at_sigma']:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "curves.npz", **out)


if __name__ == "__main__":
    main()
