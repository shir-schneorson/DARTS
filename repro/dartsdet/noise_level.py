"""The two-stage noise-level rule (Section V-F, Algorithm 1): the amplitude estimate of eqs.
(loud) and (theta-hat), then the level of eq. (sigma-hat). No labels are read.

The rule reads one trained model on two pools: H, the held-out pixels of its training fold, and
S, the screened pixels of the evaluated scene. Every statistic is the model's (`statistics`),
each pixel's context held at its clean value.

Loudness, eq. (loud): a pixel y is loud at level sigma if the mean over the distances d in A of
exp l(d; y + sigma eps) is at least 1 / alpha, eps a fixed standard normal draw and A whitened
distances along t_hat.

Responses: r(sigma) is the loud share of S. The reference r_{theta,rho}(sigma) is the loud share
of the top `share` of H by projection on t_hat, after theta t is added to a share rho of
`implants` held-out pixels drawn once; the flags of the clean top pixels and of the implanted
pixels at every theta are computed once, and a mix only re-screens. Each response is divided by
its maximum over the levels after the first, and (theta-hat, rho-hat) minimize the root mean
squared difference over those levels.

Selection: sigma-hat maximizes the PD at `metrics.PF` of the held-out pixels with theta t added,
thresholded at the (1 - PF) quantile of the clean held-out pixels, averaged over the candidate
amplitudes theta in [theta-hat / 2, theta-hat]. sigma-hat_G (Table II's GLRT row) is the same
second stage on the held-out PD of the plug-in GLRT (`held_out_glrt`), on the implanted pixels.

Candidate amplitudes are third octaves offset by a sixth of an octave, so that none equals an
amplitude an experiment implants. The draws are seeded: the implanted held-out pixels with 0, the
loudness noise of H with 1000 and of S with 1500, the order in which implants enter a mix with 100.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp

from dartsdet import statistics
from dartsdet.metrics import PF
from dartsdet.model import DARTS, pixels
from dartsdet.whitening import Inputs


@dataclass(frozen=True)
class Rule:
    """The rule's grids and constants: noise levels, candidate amplitudes theta (multiples of t),
    target shares rho, loudness distances A (whitened units) and level alpha, the number of
    implanted held-out pixels, and the screened share."""

    sigmas: tuple[float, ...]
    thetas: tuple[float, ...]
    rhos: tuple[float, ...]
    distances: torch.Tensor
    alpha: float
    implants: int
    share: float

    @classmethod
    def from_config(cls, recipe: dict, amplitudes: tuple[float, float]) -> "Rule":
        """The rule of the recipe (configs/darts.json) over candidate amplitudes from about
        amplitudes[0] to amplitudes[1]."""
        rule, levels = recipe["rule"], recipe["rule"]["noise_levels"]
        sigmas = tuple(levels["first"] * levels["base"] ** (k / levels["steps_per_base"])
                       for k in range(levels["count"]))
        low, high = amplitudes
        thetas = tuple(low * 2 ** ((k - 1) / 3 + 1 / 6)
                       for k in range(round(3 * math.log2(high / low)) + 3))
        return cls(sigmas, thetas, tuple(rule["rhos"]), torch.tensor(rule["loud_distances"]),
                   rule["loud_alpha"], rule["implants"], recipe["screen"])


def reference_pool(held: torch.Tensor, usable: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
    """The pool of a model trained on an implant-free scene, sorted: its held-out pixels `held`
    plus the same share (seed 7) of that scene's screened pixels, existing (P,) bool but not
    usable (P,) bool, which the model never trained on; just H for an unscreened model."""
    extra = np.flatnonzero(exists.numpy() & ~usable.numpy())
    share = len(held) / int(usable.sum())
    extra = np.random.default_rng(7).permutation(extra)[:round(share * len(extra))]
    return torch.cat([held, torch.from_numpy(extra)]).sort().values


def loud(model: DARTS, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor, sigma: float,
         eps: torch.Tensor, rule: Rule) -> np.ndarray:
    """(n,) bool, eq. (loud), of centres y (n, B) with contexts c (n, C) and noise eps (n, B)."""
    ell = statistics.log_ratio(model, y + sigma * eps, c, t, sigma, rule.distances)[:, 1:]
    return logsumexp(ell.numpy(), 1) - np.log(len(rule.distances)) >= np.log(1 / rule.alpha)


def held_out_reads(model: DARTS, inputs: Inputs, held: torch.Tensor, rule: Rule) -> dict:
    """What the rule reads on the held-out pixels `held` of `inputs`, contexts reading its usable
    pixels: `projection` (n,) on t_hat; `top` and `pick`, indices into `held` of the screened
    share and of the implanted pixels; their loud flags `clean` (top, S) and `flags`
    (thetas, pick, S); and the PD `pd` (thetas, S) of the held-out pixels with theta t added."""
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    t, y, n = inputs.t, pixels(z)[held], len(held)
    projection = (inputs.fx[held].float() @ (t / t.norm()).float()).numpy()
    top = np.sort(np.argsort(-projection, kind="stable")[:round(rule.share * n)])
    pick = np.sort(np.random.default_rng(0).choice(n, min(rule.implants, n), replace=False))
    top_t, pick_t = torch.from_numpy(top), torch.from_numpy(pick)
    eps_top, eps_pick = (torch.randn(len(i), y.shape[1], generator=torch.Generator().manual_seed(
        1000)).to(y.device) for i in (top, pick))
    step = t.float().to(y.device) / model.std
    pd = np.zeros((len(rule.thetas), len(rule.sigmas)))
    clean = np.zeros((len(top), len(rule.sigmas)), dtype=bool)
    flags = np.zeros((len(rule.thetas), len(pick), len(rule.sigmas)), dtype=bool)
    for i, sigma in enumerate(rule.sigmas):
        c = statistics.contexts(model, z, mask, sigma)[held]
        null = np.sort(statistics.rao(model, y, c, t, sigma).double().numpy())[::-1]
        clean[:, i] = loud(model, y[top_t], c[top_t], t, sigma, eps_top, rule)
        for j, theta in enumerate(rule.thetas):
            shifted = y + theta * step
            hits = statistics.rao(model, shifted, c, t, sigma).double().numpy()
            pd[j, i] = np.mean(hits > null[int(np.floor(PF * n))])
            flags[j, :, i] = loud(model, shifted[pick_t], c[pick_t], t, sigma, eps_pick, rule)
    return {"projection": projection, "t_norm": float(t.norm()), "top": top, "pick": pick,
            "clean": clean, "flags": flags, "pd": pd}


def mix_response(reads: dict, i: int, rho: float, order: np.ndarray, rule: Rule) -> np.ndarray:
    """r_{theta,rho} (S,), theta = rule.thetas[i]: the loud share of the top `share` of the
    held-out pixels by projection, after theta t is added to the first round(rho n) implanted
    pixels in `order`."""
    n = len(reads["projection"])
    implanted = reads["pick"][order[:round(rho * n)]]
    projection = reads["projection"].copy()
    projection[implanted] += rule.thetas[i] * reads["t_norm"]
    screen = np.argsort(-projection, kind="stable")[:round(rule.share * n)]
    is_implant = np.zeros(n, dtype=bool)
    is_implant[implanted] = True
    in_top, in_pick = np.full(n, -1), np.full(n, -1)
    in_top[reads["top"]] = np.arange(len(reads["top"]))
    in_pick[reads["pick"]] = np.arange(len(reads["pick"]))
    clean, hits = screen[~is_implant[screen]], screen[is_implant[screen]]
    count = reads["clean"][in_top[clean]].sum(0) + reads["flags"][i][in_pick[hits]].sum(0)
    return count / len(screen)


def scene_response(model: DARTS, inputs: Inputs, pool: torch.Tensor, rule: Rule) -> np.ndarray:
    """r (S,): the loud share of the pixels `pool` of `inputs`, contexts reading its usable
    pixels."""
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    y = pixels(z)[pool]
    eps = torch.randn(len(pool), y.shape[1], generator=torch.Generator().manual_seed(1500))
    return np.array([loud(model, y, statistics.contexts(model, z, mask, sigma)[pool], inputs.t,
                          sigma, eps.to(y.device), rule).mean() for sigma in rule.sigmas])


def select(model: DARTS, reads: dict, scene: Inputs, pool: torch.Tensor, rule: Rule) -> dict:
    """sigma-hat, eq. (sigma-hat), of the model on the scene `scene`, whose screened pixels `pool`
    form S, from `held_out_reads` along the scene's signature. Also returns theta-hat, rho-hat, the
    responses r (S,) and r_{theta-hat,rho-hat} (S,), the held-out PD averaged over
    [theta-hat / 2, theta-hat] (S,), and sigma-u: the level without the amplitude estimate, whose
    held-out PD averaged over every candidate amplitude is highest."""
    order = np.random.default_rng(100).permutation(len(reads["pick"]))
    keys = [(i, rho) for i in range(len(rule.thetas)) for rho in rule.rhos]
    references = np.array([mix_response(reads, i, rho, order, rule) for i, rho in keys])
    response = scene_response(model, scene, pool, rule)
    normalized = [r[..., 1:] / np.maximum(r[..., 1:].max(-1, keepdims=True), 1e-12)
                  for r in (references, response)]
    k = int(np.argmin(np.sqrt(np.mean((normalized[0] - normalized[1]) ** 2, axis=1))))
    i, rho = keys[k]
    band_pd = band_average(reads["pd"], i, rule)
    return {"sigma_hat": rule.sigmas[int(np.argmax(band_pd))], "theta_hat": rule.thetas[i],
            "rho_hat": rho, "response": response, "reference": references[k], "band_pd": band_pd,
            "sigma_u": rule.sigmas[int(np.argmax(reads["pd"].mean(0)))]}


def band_average(pd: np.ndarray, i: int, rule: Rule) -> np.ndarray:
    """A held-out PD table (thetas, S) averaged over the candidate amplitudes in
    [thetas[i] / 2, thetas[i]] (S,), the sum of eq. (sigma-hat)."""
    thetas = np.asarray(rule.thetas)
    band = (thetas >= thetas[i] / 2 * (1 - 1e-9)) & (thetas <= thetas[i] * (1 + 1e-9))
    return pd[band].mean(0)


def held_out_glrt(model: DARTS, inputs: Inputs, held: torch.Tensor, rule: Rule,
                  amplitudes: torch.Tensor) -> np.ndarray:
    """The plug-in GLRT's held-out PD (thetas, S), for sigma-hat_G: at every level, the GLRT
    (`statistics.glrt` over `amplitudes`) of the `implants` held-out pixels that `held_out_reads`
    picks, with theta t added, against the (floor(PF n) + 1)-th largest GLRT over all of `held`.
    The GLRT costs one log-ratio row per amplitude, hence the implanted pixels only."""
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    t, y, n = inputs.t, pixels(z)[held], len(held)
    pick = torch.from_numpy(np.sort(np.random.default_rng(0).choice(n, min(rule.implants, n),
                                                                    replace=False)))
    step = t.float().to(y.device) / model.std
    pd = np.zeros((len(rule.thetas), len(rule.sigmas)))
    for i, sigma in enumerate(rule.sigmas):
        c = statistics.contexts(model, z, mask, sigma)[held]
        null = np.sort(statistics.glrt(model, y, c, t, sigma, amplitudes).numpy())[::-1]
        for j, theta in enumerate(rule.thetas):
            hits = statistics.glrt(model, y[pick] + theta * step, c[pick], t, sigma, amplitudes)
            pd[j, i] = np.mean(hits.numpy() > null[int(np.floor(PF * n))])
    return pd
