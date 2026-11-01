"""Experiment 2: relative masses in a multi-noise score model.

(a) Where the KL divergence between two densities sits across noise levels, KL = the integral of
sigma^2 F_sigma over log sigma, for a fine-scale error (a ripple) and a coarse-scale one (the
relative mass of two modes), on a grid. (b) A 1-D background 0.8 N(-3, 0.5^2) + 0.2 N(3, 0.5^2),
standardized: each route's estimate of the log ratio from mode A's centre to mode B's, whose
exact value is log 0.25. (c) A 10-band background of three separated Gaussian clusters (masses
0.6, 0.3, 0.1), standardized, with targets x = b + delta t, t from cluster 1 toward cluster 2 and
delta a fraction of their distance: the one-sided GLRT 2 max_theta l(theta) from each route, and
the learned Rao test, scored by PD at PF = 1e-2 on held-out background.

Every network's initialization is seeded with its training seed. Writes <out>/kl.npz,
<out>/toy1d.json and <out>/toy_gmm.json, which `figures/figure2.py` draws.

Usage: python -m experiments.experiment2 [--out DIR]
"""

import json
import math

import numpy as np
import torch
from scipy.signal import fftconvolve

from experiments import relative_mass as rm
from experiments.cli import arguments

D, N_TRAIN, N_BACKGROUND, N_TARGETS = 10, 20000, 3000, 300
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
SIGMA0, SIGMA1 = 0.3, 1.0


def levels(extra: tuple[float, float]) -> torch.Tensor:
    """The classifier's 32 log-spaced levels on [0.01, 100], plus `extra`, sorted."""
    grid = torch.exp(torch.linspace(math.log(0.01), math.log(100), 32))
    return torch.cat([grid, torch.tensor(extra)]).sort()[0]


def kl_profiles() -> dict[str, np.ndarray]:
    """sigma^2 F_sigma against sigma, for a ripple and for a mass mismatch, in 1-D, each
    normalized by its integral over log sigma on the grid (the KL, up to the grid's range)."""
    x = np.linspace(-48, 48, 48001)
    dx = x[1] - x[0]

    def normal(m: float, s: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    def norm(f: np.ndarray) -> np.ndarray:
        return f / (f.sum() * dx)

    def smooth(f: np.ndarray, s: float) -> np.ndarray:
        return norm(np.maximum(fftconvolve(f, normal(0, s) * dx, mode="same"), 1e-300))

    def fisher(p: np.ndarray, q: np.ndarray) -> float:
        sp, sq = np.gradient(np.log(p), dx), np.gradient(np.log(q), dx)
        m = p > 1e-12
        return float(((sp - sq) ** 2 * p)[m].sum() * dx)

    sigmas = np.geomspace(0.01, 6, 120)
    pairs = {"ripple": (normal(0, 1), norm(normal(0, 1) * (1 + 0.3 * np.sin(8 * x)))),
             "mass": (0.5 * normal(-3, .5) + 0.5 * normal(3, .5),
                      0.9 * normal(-3, .5) + 0.1 * normal(3, .5))}
    out = {"sigmas": sigmas}
    for name, (p, q) in pairs.items():
        density = np.array([s**2 * fisher(smooth(p, s), smooth(q, s)) for s in sigmas])
        out[name] = density / np.trapezoid(density, np.log(sigmas))
    return out


def trained(kind: str, x: torch.Tensor, out: int, width: int, seed: int,
            steps: int, **options: object) -> rm.MLP:
    """A network of `kind` on x, initialized and trained from `seed`."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = rm.MLP(x.shape[1], out, width)
    return rm.train(network, x, kind, steps, seed, **options)


def toy1d(n: int, seed: int) -> dict[str, float]:
    """Each route's log ratio from mode A's centre to mode B's, with n training samples."""
    weights, centres, sd = torch.tensor([0.8, 0.2]), torch.tensor([[-3.0], [3.0]]), 0.5
    m = float((weights[:, None] * centres).sum())
    s = math.sqrt(float((weights * (sd**2 + centres[:, 0] ** 2)).sum()) - m**2)
    generator = torch.Generator().manual_seed(seed)
    component = torch.multinomial(weights, n, replacement=True, generator=generator)
    x = ((centres[component] + sd * torch.randn(n, 1, generator=generator)) - m) / s
    dsm = trained("dsm", x, 1, 128, seed, 15000)
    energy = trained("energy", x, 1, 128, seed + 100, 15000)
    clf = trained("classifier", x, 1, 128, seed + 200, 15000, levels=levels((0.02, 0.3)),
                  batch=256)
    a, b = (centres[:1] - m) / s, (centres[1:] - m) / s
    with torch.no_grad():
        return {"direct@0.02": float(rm.direct(dsm, a, b, 0.02)[0]),
                "direct@0.3": float(rm.direct(dsm, a, b, 0.3)[0]),
                "rect_energy@0.3": float(rm.rect_energy(dsm, energy, a, b, 0.3, 1.0)[0]),
                "classifier@0.3": float(rm.classifier(
                    lambda y, lv: rm.logits(clf, y, lv), levels((0.02, 0.3)), a, b, 0.3)[0]),
                "itd@0": float(rm.itd(dsm, a, b, n_mc=4096)[0])}


class GaussianMixture:
    """sum_k pi_k N(mu_k, V_k), with its smoothed log density log p_sigma, sigma^2 I added."""

    def __init__(self, weights: torch.Tensor, means: torch.Tensor, covs: torch.Tensor) -> None:
        self.w, self.mu, self.v = weights.double(), means.double(), covs.double()

    def log_prob(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """log p_sigma(y) (n,) of points y (n, d), float64."""
        cov = self.v + sigma**2 * torch.eye(self.mu.shape[1], dtype=torch.float64)
        components = torch.distributions.MultivariateNormal(self.mu, covariance_matrix=cov)
        return torch.logsumexp(components.log_prob(y.double()[:, None, :]) + self.w.log(), 1)


def clusters(seed: int) -> tuple[GaussianMixture, torch.Tensor, float]:
    """The standardized three-cluster mixture, the unit target direction from cluster 1 toward
    cluster 2, and their distance."""
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=generator))
    means = torch.zeros(3, D)
    means[1, 0], means[2, 0], means[2, 1] = 7.0, 3.5, 6.0
    means = means @ q.T
    covs = []
    for _ in range(3):
        rotation = torch.linalg.qr(torch.randn(D, D, generator=generator))[0]
        covs.append(rotation @ torch.diag(0.3 + 1.2 * torch.rand(D, generator=generator))
                    @ rotation.T)
    covs, w = torch.stack(covs), torch.tensor([0.6, 0.3, 0.1])
    mean = (w[:, None] * means).sum(0)
    second = (w[:, None, None] * (covs + means[:, :, None] * means[:, None, :])).sum(0)
    unmix = torch.linalg.inv(torch.linalg.cholesky(second - mean[:, None] * mean[None]))
    g = GaussianMixture(w, (means - mean) @ unmix.T, unmix @ covs @ unmix.T)
    step = (g.mu[1] - g.mu[0]).float()
    return g, step / step.norm(), float(step.norm())


def draw(g: GaussianMixture, n: int, generator: torch.Generator) -> torch.Tensor:
    """n draws (n, D) of the mixture, float32."""
    component = torch.multinomial(g.w.float(), n, replacement=True, generator=generator)
    chol = torch.linalg.cholesky(g.v).float()
    return g.mu.float()[component] + (chol[component] @ torch.randn(n, D, 1,
                                                                    generator=generator))[..., 0]


@torch.no_grad()
def cumulative(dsm: rm.MLP, x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor,
               sigma: float) -> torch.Tensor:
    """l(theta) (n, J) of the learned score: -t' s_sigma integrated along -t on a grid 8 times
    finer than theta, read at theta."""
    fine = torch.linspace(0, float(theta[-1]), 8 * (len(theta) - 1) + 1)
    slopes = torch.stack([-(rm.score(dsm, x - u * t, sigma) * t).sum(1) for u in fine], 1)
    ell = torch.cat([torch.zeros(len(x), 1), torch.cumulative_trapezoid(slopes, fine, dim=1)], 1)
    return ell[:, ::8]


@torch.no_grad()
def profiles(route: str, x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor,
             nets: dict) -> torch.Tensor:
    """l(theta_j) (n, J) at SIGMA0 by one route."""
    n, j = len(x), len(theta)
    shifted = (x[:, None, :] - theta[None, :, None] * t).reshape(-1, D)
    repeated = x.repeat_interleave(j, 0)
    if route == "oracle":
        g = nets["oracle"]
        return (g.log_prob(shifted, SIGMA0) - g.log_prob(repeated, SIGMA0)).reshape(n, j).float()
    if route == "classifier":
        return rm.classifier(lambda y, lv: rm.logits(nets["clf"], y, lv), levels((0.1, 0.3)),
                             repeated, shifted, SIGMA0).reshape(n, j)
    if route == "direct":
        return cumulative(nets["dsm"], x, t, theta, SIGMA0)
    if route != "rect_energy":
        raise ValueError(f"unknown route {route!r}")

    def up(y: torch.Tensor) -> torch.Tensor:
        return rm.vertical_energy(nets["energy"], y, SIGMA0, SIGMA1, n=32)

    return (up(x)[:, None] + cumulative(nets["dsm"], x, t, theta, SIGMA1)
            - up(shifted).reshape(n, j))


def power(background: torch.Tensor, targets: torch.Tensor) -> float:
    """PD at PF = 1e-2: the share of targets above the background's 0.99 quantile."""
    return float((targets.double() > torch.quantile(background.double(), 0.99)).float().mean())


def toy_gmm(seed: int) -> dict[str, list[float]]:
    """PD per target fraction of the GLRT by each route, and of the learned Rao test."""
    g, t, distance = clusters(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    x = draw(g, N_TRAIN, generator)
    nets = {"oracle": g, "dsm": trained("dsm", x, D, 256, seed, 20000),
            "energy": trained("energy", x, 1, 256, seed + 100, 20000),
            "clf": trained("classifier", x, 1, 256, seed + 200, 20000,
                           levels=levels((0.1, 0.3)), batch=256)}
    pixels = [draw(g, N_BACKGROUND, generator)] + [draw(g, N_TARGETS, generator) + f * distance * t
                                                  for f in FRACTIONS]
    everything, sizes = torch.cat(pixels), [len(p) for p in pixels]
    theta = torch.linspace(0, 1.3 * distance, 33)
    out = {}
    for route in ("oracle", "direct", "classifier", "rect_energy"):
        stats = (2 * profiles(route, everything, t, theta, nets).max(1).values).split(sizes)
        out[f"{route}@{SIGMA0}"] = [power(stats[0], s) for s in stats[1:]]
    with torch.no_grad():
        rao = (-(rm.score(nets["dsm"], everything, SIGMA0) * t).sum(1)).split(sizes)
    out[f"rao@{SIGMA0}"] = [power(rao[0], s) for s in rao[1:]]
    return out


def main() -> None:
    parser = arguments(__doc__, "experiment2", device=False, data=False, checkpoints=False)
    args = parser.parse_args()
    torch.set_num_threads(8)  # as the paper's runs; the thread count moves float reductions
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "kl.npz", **kl_profiles())
    runs = [{"N": n, "seed": seed, **toy1d(n, seed)} for n in (2000, 20000) for seed in range(3)]
    (args.out / "toy1d.json").write_text(json.dumps(runs, indent=1))
    runs = [{"seed": seed, "pd": toy_gmm(seed)} for seed in range(3)]
    (args.out / "toy_gmm.json").write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
