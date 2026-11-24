"""Experiment 2's synthetic scene, its exact densities, and the oracle detectors.

The scene has side x side pixels cut into `regions` Voronoi cells. Each cell draws one of the
`components` Gaussian components of a background mixture (equal weights, means `separation` N(0, I),
covariances with random axes and eigenvalues log-uniform on `eigenvalues`) and a scale
w ~ chi2_nu / nu; its pixels are N(mu_k, S_k / w). Marginalizing w, each pixel's density, the scene
density, is the Student-t mixture sum_k pi_k t_nu(mu_k, S_k); given its region it is Gaussian.

Targets: one per `background_per_target` pixels, on a lattice of spacing `spacing`, the same sites
and draws at every amplitude. A target is x = b + alpha d_i, d_i = mu_t + spread eps, mu_t of norm
`distance` in a random direction; the detectors know only mu_t.

Oracles, along t = mu_t in the units of x:
- Rao: -t' grad log p(x) of the exact scene density, up to its scale;
- GLRT: 2 max over 0 <= theta <= theta_max of log p(x - theta t) - log p(x), on a uniform grid,
  refined by a parabola;
- region Rao oracle: the conditional Rao test given the pixel's true region, the amplitude score
  U(x | r) = w t' S_k^-1 (x - mu_k) of N(mu_k, S_k / w), one threshold for the whole scene;
- region Rao oracle at the matched level: the same test smoothed exactly, in the whitened units of
  the implant-free scene's fit, at the level sigma_m that eq. (sigma-amplitude) matches to the
  amplitude.

The draws follow the original generator, stream by stream: a torch generator seeded with the seed
draws the mixture, the background noise and the targets; a numpy generator with the same seed
draws the cells, their components and scales, and the sites.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from baselines.gmm import profile_maximum
from dartsdet.data import Scene
from dartsdet.metrics import PF

CHUNK = 4096  # pixels per density evaluation; memory only


@dataclass(frozen=True)
class Synthetic:
    """One seed's scene. weights (K,), means (K, B), scale_tril (K, B, B) and dof: the Student-t
    mixture. background (H, W, B), and per pixel in Fortran order: region (P,), its Voronoi cell,
    and that cell's component (P,) and scale (P,). sites: the target pixels (n,) in Fortran
    order; draws (n, B) their d_i; target (B,): mu_t."""

    weights: torch.Tensor
    means: torch.Tensor
    scale_tril: torch.Tensor
    dof: float
    background: torch.Tensor
    region: torch.Tensor
    component: torch.Tensor
    scale: torch.Tensor
    sites: torch.Tensor
    draws: torch.Tensor
    target: torch.Tensor


def draw(config: dict, seed: int) -> Synthetic:
    """The scene of `seed` (configs/experiment2.json's `scene`, `background` and `target`)."""
    generator, rng = torch.Generator().manual_seed(seed), np.random.default_rng(seed)
    spec, mixture, target = config["scene"], config["background"], config["target"]
    k, bands, f64 = mixture["components"], mixture["bands"], torch.float64
    means = mixture["separation"] * torch.randn(k, bands, generator=generator, dtype=f64)
    rotations = torch.linalg.qr(torch.randn(k, bands, bands, generator=generator, dtype=f64)).Q
    low, high = (math.log(v) for v in mixture["eigenvalues"])
    eigenvalues = (low + (high - low) * torch.rand(k, bands, generator=generator, dtype=f64)).exp()
    covariances = rotations @ torch.diag_embed(eigenvalues) @ rotations.mT
    scale_tril = torch.linalg.cholesky((covariances + covariances.mT) / 2)
    weights = torch.full((k,), 1 / k, dtype=f64)
    side = spec["side"]
    rows, cols = np.divmod(np.arange(side * side), side)
    region = cKDTree(rng.uniform(0, side, (spec["regions"], 2))).query(np.c_[rows, cols])[1]
    cell_component = torch.from_numpy(rng.choice(k, spec["regions"], p=weights.numpy()))
    cell_scale = torch.from_numpy(rng.chisquare(spec["dof"], spec["regions"]) / spec["dof"])
    component, scale = cell_component[region], cell_scale[region]
    noise = torch.randn(len(component), bands, 1, generator=generator, dtype=f64)
    background = means[component] + (scale_tril[component] @ noise)[..., 0] / scale.sqrt()[:, None]
    lattice = np.flatnonzero((rows % target["spacing"] == 0) & (cols % target["spacing"] == 0))
    sites = np.sort(rng.choice(lattice, side * side // target["background_per_target"],
                               replace=False))
    direction = torch.randn(bands, generator=generator, dtype=f64)
    mean = target["distance"] * direction / direction.norm()
    draws = mean + target["spread"] * torch.randn(len(sites), bands, generator=generator, dtype=f64)
    fortran = np.arange(side * side).reshape(side, side).ravel(order="F")  # row-major index by p
    position = np.empty(side * side, dtype=np.int64)
    position[fortran] = np.arange(side * side)
    order = np.argsort(position[sites])
    return Synthetic(weights, means, scale_tril, spec["dof"], background.reshape(side, side, bands),
                     torch.from_numpy(region[fortran]), component[fortran], scale[fortran],
                     torch.from_numpy(position[sites][order]), draws[torch.from_numpy(order)], mean)


def implanted(synthetic: Synthetic, alpha: float) -> Scene:
    """The scene with alpha d_i added at every site, as a `data.Scene` (Fortran pixel order)."""
    H, W, B = synthetic.background.shape
    spectra = synthetic.background.numpy().reshape(-1, B, order="F").copy()
    spectra[synthetic.sites.numpy()] += alpha * synthetic.draws.numpy()
    labels = np.zeros(H * W)
    labels[synthetic.sites.numpy()] = 1
    return Scene(spectra.reshape(H, W, B, order="F"), labels, np.ones(H * W, dtype=bool),
                 synthetic.target.numpy(), additive=True)


def log_density(synthetic: Synthetic, x: torch.Tensor) -> torch.Tensor:
    """log p(x) (n,) of the scene density at spectra x (n, B), float64."""
    nu, bands = synthetic.dof, x.shape[1]
    centred = (x[None, :, :] - synthetic.means[:, None, :]).transpose(1, 2)  # (K, B, n)
    q = torch.linalg.solve_triangular(synthetic.scale_tril, centred, upper=False).square().sum(1)
    log_det = 2 * synthetic.scale_tril.diagonal(dim1=1, dim2=2).log().sum(1)
    constant = (math.lgamma((nu + bands) / 2) - math.lgamma(nu / 2)
                - bands / 2 * math.log(nu * math.pi))
    log_t = constant - log_det[:, None] / 2 - (nu + bands) / 2 * torch.log1p(q / nu)
    return torch.logsumexp(log_t + synthetic.weights.log()[:, None], dim=0)


def oracle_rao(synthetic: Synthetic, x: torch.Tensor) -> torch.Tensor:
    """-t' grad log p(x) (n,) at spectra x (n, B), t = mu_t."""
    out = []
    for block in x.split(CHUNK):
        block = block.detach().requires_grad_(True)
        with torch.enable_grad():
            (score,) = torch.autograd.grad(log_density(synthetic, block).sum(), block)
        out.append(-score @ synthetic.target)
    return torch.cat(out)


def oracle_glrt(synthetic: Synthetic, x: torch.Tensor, theta_max: float,
                count: int) -> torch.Tensor:
    """2 max l(theta) (n,) over `count` amplitudes on [0, theta_max], multiples of t = mu_t, and one
    step below 0 as a neighbour, refined by `baselines.gmm.profile_maximum`."""
    theta = torch.arange(-1, count, dtype=torch.float64) * (theta_max / (count - 1))
    out = []
    for block in x.split(256):
        shifted = (block[:, None, :] - theta[:, None] * synthetic.target).reshape(-1, x.shape[1])
        profile = (log_density(synthetic, shifted).reshape(len(block), len(theta))
                   - log_density(synthetic, block)[:, None])
        out.append(2 * profile_maximum(profile, first=1))
    return torch.cat(out)


def region_score(synthetic: Synthetic, x: torch.Tensor) -> torch.Tensor:
    """The region Rao oracle U(x | r) (P,) at the scene's spectra x (P, B), in pixel order."""
    k = synthetic.component
    centred = (x - synthetic.means[k])[..., None]
    pull = torch.cholesky_solve(centred, synthetic.scale_tril[k])[..., 0]
    return synthetic.scale * (pull @ synthetic.target)


def region_covariances(synthetic: Synthetic, fitted: tuple[torch.Tensor, torch.Tensor]) -> tuple:
    """In the whitened units y = (x - mean) W of `fitted` = (mean, W): per region cell, the centre
    m_r (C, B) and the eigenvalues lam (C, B) and eigenvectors Q (C, B, B) of C_r = W S_k W / w, the
    covariance of a pixel given its region; and each pixel's cell (P,)."""
    mean, W = fitted
    present, cell = torch.unique(synthetic.region, return_inverse=True)
    component = torch.empty(len(present), dtype=torch.long)
    component[cell] = synthetic.component
    scale = torch.empty(len(present), dtype=torch.float64)
    scale[cell] = synthetic.scale
    covariance = W @ (synthetic.scale_tril @ synthetic.scale_tril.mT)[component] @ W
    lam, Q = torch.linalg.eigh(covariance / scale[:, None, None])
    return (synthetic.means[component] - mean) @ W, lam, Q, cell


def region_smoothed(synthetic: Synthetic, fitted: tuple[torch.Tensor, torch.Tensor],
                    x: torch.Tensor, sigma: float) -> torch.Tensor:
    """The region Rao oracle smoothed at sigma, U_sigma(y | r) = t' (C_r + sigma^2 I)^-1 (y - m_r)
    (P,), eq. (U-sigma-cond), exact because a pixel given its region is N(m_r, C_r): spectra
    x (P, B) and t = mu_t in the whitened units of `fitted`. sigma = inf gives the limit
    t' (y - m_r)."""
    mean, W = fitted
    m, lam, Q, cell = region_covariances(synthetic, fitted)
    along = (Q.mT @ (synthetic.target @ W)[:, None])[..., 0]  # (C, B): Q' t
    parts = (Q.mT[cell] @ ((x - mean) @ W - m[cell])[..., None])[..., 0] * along[cell]
    return parts.sum(1) if math.isinf(sigma) else (parts / (lam[cell] + sigma**2)).sum(1)


def matched_levels(synthetic: Synthetic, fitted: tuple[torch.Tensor, torch.Tensor],
                   amplitudes: list[float], grid: dict) -> list[float]:
    """The matched level sigma_m of each amplitude theta, eq. (sigma-amplitude), in the whitened
    units of `fitted`: kappa_sigma sigma^2 = omega theta |t|^2 / 2, kappa_sigma the 1 - PF quantile
    of `region_smoothed` over the background pixels and omega = cos^2(t, J t), J t = E_0[grad U_0]
    = the background mean of C_r^-1 t (Section IV-E). Solved by brentq in log sigma between
    the levels of `grid` (log-spaced) that bracket it; inf where no level reaches it."""
    clean = implanted(synthetic, 0.0)
    x, background = torch.from_numpy(clean.spectra()), clean.labels == 0
    t = synthetic.target @ fitted[1]
    _, lam, Q, cell = region_covariances(synthetic, fitted)
    jt = (Q @ ((Q.mT @ t[:, None])[..., 0] / lam)[..., None])[..., 0][cell][background].mean(0)
    omega = float((t @ jt) ** 2 / (t @ t) / (jt @ jt))

    def c(sigma: float) -> float:
        u = region_smoothed(synthetic, fitted, x, sigma).numpy()[background]
        return float(np.quantile(u, 1 - PF)) * sigma**2

    levels = np.logspace(grid["log10_first"], grid["log10_last"], grid["count"])
    curve = np.array([c(s) for s in levels])
    return [level(c, levels, curve, omega * theta * float(t @ t) / 2) for theta in amplitudes]


def level(c: Callable[[float], float], levels: np.ndarray, curve: np.ndarray,
          target: float) -> float:
    """The sigma with c(sigma) = target, bracketed by the first of `levels` whose `curve`
    (= c(levels)) reaches it; the smallest level if the first does, inf if none does."""
    above = np.flatnonzero(curve >= target)
    if len(above) == 0:
        return math.inf
    if above[0] == 0:
        return float(levels[0])
    bracket = math.log(levels[above[0] - 1]), math.log(levels[above[0]])
    return math.exp(brentq(lambda v: math.log(c(math.exp(v)) / target), *bracket, xtol=1e-6))
