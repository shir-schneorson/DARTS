"""Experiment 3's synthetic spatial scene, with an exact scene density and an exact context.

A `side` x `side` scene is cut into `cells` Voronoi regions. Each region draws a component k of a
Gaussian mixture (`n_components` of them) and a scale w ~ chi2_nu / nu, and its pixels are
N(mu_k, Sigma_k / w).
A pixel's marginal, the scene density, is then the Student-t mixture sum_k pi_k t_nu(mu_k, Sigma_k);
given its region, a pixel is Gaussian. Targets sit one per `per_background` pixels on a
`spacing`-pixel lattice: x = b + alpha d_i, d_i = mu_t + std eps, mu_t at `distance` in a random
direction. The same sites and draws serve every alpha; the detectors know only mu_t.

Pixels are numbered row-major. All arithmetic is float64 on the CPU.
"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.student_mixture import StudentMixture

DTYPE = torch.float64


def draw_mixture(spec: dict, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """(weights (K,), means (K, B), covariances (K, B, B)) of the Gaussian mixture: means
    `separation` N(0, I), covariances randomly rotated with log-uniform eigenvalues."""
    k, dim = spec["n_components"], spec["dim"]
    means = spec["separation"] * torch.randn(k, dim, generator=generator, dtype=DTYPE)
    rotations = torch.linalg.qr(torch.randn(k, dim, dim, generator=generator, dtype=DTYPE)).Q
    low, high = (math.log(v) for v in spec["eigenvalues"])
    eigenvalues = (low + (high - low) * torch.rand(k, dim, generator=generator, dtype=DTYPE)).exp()
    covariances = rotations @ torch.diag_embed(eigenvalues) @ rotations.mT
    return torch.full((k,), 1 / k, dtype=DTYPE), means, (covariances + covariances.mT) / 2


def draw_scene(config: dict, seed: int) -> dict:
    """One seed's scene: `oracle`, `background` (P, B), each pixel's `component` and `scale`,
    the target `sites` and `targets` (their draws d_i), `target_mean` mu_t and `side`."""
    generator, rng = torch.Generator().manual_seed(seed), np.random.default_rng(seed)
    spec, target = config["scene"], config["target"]
    weights, means, covariances = draw_mixture(config["background"], generator)
    oracle = StudentMixture.from_covariances(weights, means, covariances, spec["dof"])
    side = spec["side"]
    rows, cols = np.divmod(np.arange(side * side), side)
    region = cKDTree(rng.uniform(0, side, (spec["cells"], 2))).query(np.c_[rows, cols])[1]
    k = torch.from_numpy(rng.choice(len(weights), spec["cells"], p=weights.numpy()))
    w = torch.from_numpy(rng.chisquare(spec["dof"], spec["cells"]) / spec["dof"])
    component, scale = k[region], w[region]
    background = oracle.sample(component, scale, generator)
    lattice = np.flatnonzero((rows % target["spacing"] == 0) & (cols % target["spacing"] == 0))
    sites = torch.from_numpy(np.sort(rng.choice(lattice, side * side // target["per_background"],
                                                replace=False)))
    direction = torch.randn(config["background"]["dim"], generator=generator, dtype=DTYPE)
    mean = target["distance"] * direction / direction.norm()
    draws = mean + target["std"] * torch.randn(len(sites), len(mean), generator=generator,
                                               dtype=DTYPE)
    return {"oracle": oracle, "background": background, "component": component, "scale": scale,
            "sites": sites, "targets": draws, "target_mean": mean, "side": side}


def implanted(scene: dict, alpha: float) -> tuple[torch.Tensor, np.ndarray]:
    """The pixels (P, B) with the targets added at amplitude alpha, and labels (P,), 1 on them."""
    x = scene["background"].clone()
    x[scene["sites"]] += alpha * scene["targets"]
    labels = np.zeros(len(x), np.int64)
    labels[scene["sites"].numpy()] = 1
    return x, labels


def oracle_rao(oracle: StudentMixture, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """One-sided Rao statistic of the scene density, -t' s(x) over its root mean square over the
    scene, (P,)."""
    projected = torch.cat([oracle.score(block) @ t for block in x.split(4096)])
    return -projected / projected.square().mean().sqrt()


def oracle_glrt(oracle: StudentMixture, x: torch.Tensor, t: torch.Tensor, theta_max: float,
                n_theta: int) -> torch.Tensor:
    """One-sided GLRT 2 max_{theta >= 0} l(theta) of the scene density, (P,): l exact on n_theta
    points of [0, theta_max] multiples of t, plus one step below 0 as a neighbour only."""
    theta = torch.arange(-1, n_theta, dtype=DTYPE) * (theta_max / (n_theta - 1))
    out = []
    for block in x.split(256):
        shifted = (block[:, None, :] - theta[:, None] * t).reshape(-1, x.shape[1])
        profile = (oracle.log_prob(shifted).reshape(len(block), len(theta))
                   - oracle.log_prob(block)[:, None])
        out.append(2 * refined_maximum(profile, first=1))
    return torch.cat(out)


def refined_maximum(profile: torch.Tensor, first: int) -> torch.Tensor:
    """Row maxima (n,) over grid points `first`.., refined by the parabola through the maximum
    and its two neighbours when it is concave and its vertex lies at or past `first`. Unlike
    `dartsdet.reads.refined_maximum`, the grid holds points before `first` (theta < 0) as
    neighbours only, so a maximum just above 0 is still refined."""
    allowed = profile.clone()
    allowed[:, :first] = -torch.inf
    peak = allowed.argmax(dim=1, keepdim=True)
    centre = peak.clamp(1, profile.shape[1] - 2)
    left, middle, right = (profile.gather(1, centre + step).squeeze(1) for step in (-1, 0, 1))
    curvature = left - 2 * middle + right
    safe = curvature.where(curvature < 0, -1.0)
    vertex = centre.squeeze(1) + (left - right) / (2 * safe)
    refinable = (peak.squeeze(1) == centre.squeeze(1)) & (curvature < 0) & (vertex >= first)
    lift = -(right - left).square() / (8 * safe)
    return profile.gather(1, peak).squeeze(1) + lift.where(refinable, 0.0)


def oracle_context(scene: dict, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rao statistic given each pixel's true (k, w), sized per context, (P,):
    w t' Sigma_k^-1 (x - mu_k) / sqrt(w t' Sigma_k^-1 t)."""
    oracle, component, scale = scene["oracle"], scene["component"], scene["scale"]
    u = -oracle.conditional_score(x, component, scale) @ t
    pull = torch.cholesky_solve(t.expand(len(oracle.means), -1)[..., None], oracle.scale_tril)
    return u / (scale * (pull[..., 0] @ t)[component]).sqrt()
