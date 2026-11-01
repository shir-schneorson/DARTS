"""What the detectors read off a trained score model, at one noise level sigma.

All reads are in whitened units along the unit signature t_hat, with each pixel's context held
fixed. Moving a pixel along -t_hat by u (the background-only direction of an additive target):

    A(x)     = -t_hat' s_sigma(x | c)                    the Rao statistic (`rao`)
    C(x)     = -dA/du (x - u t_hat) at u = 0             its curvature (`rao_and_curvature`)
    l(theta) = log p_sigma(x - theta t_hat | c) - log p_sigma(x | c)
             = integral_0^theta A(x - v t_hat) dv       the log-likelihood ratio (`log_ratio`)

and the one-sided GLRT is 2 max_theta l(theta) (`glrt`). Inputs y and zc are standardized centres
(`contexts`); the whitened shift u t_hat is u t_hat / std in those coordinates.
"""

import torch

from dartsdet.model import ContextFreeScore, ScoreModel, vp_coefficients

Model = ScoreModel | ContextFreeScore

CHUNK = 8192            # pixels per network call; memory only
PATH_STEP = 0.05        # the log ratio's trapezoid step, whitened units


def contexts(model: ScoreModel, fx: torch.Tensor, usable: torch.Tensor,
             grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(standardized centres (P, B), contexts (P, C)) of every pixel, in pixel order."""
    with torch.no_grad():
        z, mask = model.image(fx, usable, grid)
        return z[0].flatten(1).T, model.encoder(z, mask)[0].flatten(1).T


def rao(model: Model, y: torch.Tensor, context: torch.Tensor, t_hat: torch.Tensor,
        sigma: float) -> torch.Tensor:
    """A (n,) at standardized centres y (n, B) with contexts (n, C); t_hat (B,) has unit norm."""
    a, _ = vp_coefficients(sigma)
    eps = model.epsilon(a * y, context, sigma)
    return (eps / (sigma * model.coordinates.std)) @ t_hat


def rao_and_curvature(model: Model, zc: torch.Tensor, context: torch.Tensor,
                      t_hat: torch.Tensor,
                      sigmas: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """(A, C), each (n, S): pixels in rows, the noise levels `sigmas` in columns; on the CPU."""
    step = t_hat / model.coordinates.std
    A, C = (torch.zeros(len(zc), len(sigmas), dtype=zc.dtype) for _ in range(2))
    for j, sigma in enumerate(sigmas):
        for block in torch.arange(len(zc)).split(CHUNK):
            u = torch.zeros(len(block), dtype=zc.dtype, device=zc.device, requires_grad=True)
            with torch.enable_grad():
                a = rao(model, zc[block] - u[:, None] * step, context[block], t_hat, sigma)
                (du,) = torch.autograd.grad(a.sum(), u)
            A[block, j], C[block, j] = a.detach().cpu(), -du.cpu()
    return A, C


def path_rao(model: Model, y: torch.Tensor, context: torch.Tensor, t_hat: torch.Tensor,
             sigma: float, v: torch.Tensor) -> torch.Tensor:
    """A at y - v_k t_hat for every whitened distance v_k of v (K,), (n, K)."""
    step = t_hat / model.coordinates.std
    return torch.stack([rao(model, y - vk * step, context, t_hat, sigma) for vk in v], 1)


def cumulative_trapezoid(A: torch.Tensor, dv: float) -> torch.Tensor:
    """l at every grid point (n, K) of A (n, K) on a uniform grid of spacing dv, from l = 0."""
    increments = torch.cumulative_trapezoid(A, dx=dv, dim=1)
    return torch.cat([A.new_zeros(len(A), 1), increments], 1)


def log_ratio(model: Model, y: torch.Tensor, context: torch.Tensor, t_hat: torch.Tensor,
              sigma: float, thetas: tuple[float, ...]) -> torch.Tensor:
    """l(theta) (n, len(thetas)) on the CPU, by the trapezoid rule on a `PATH_STEP` grid of
    whitened distances from 0 to max(thetas); every theta must lie on that grid."""
    v = torch.arange(0, max(thetas) + PATH_STEP / 2, PATH_STEP, device=y.device)
    at = [round(theta / PATH_STEP) for theta in thetas]
    if any(abs(k * PATH_STEP - theta) > 1e-9 for k, theta in zip(at, thetas, strict=True)):
        raise ValueError(f"amplitudes {thetas} are not on the {PATH_STEP} grid")
    out = torch.zeros(len(y), len(thetas), dtype=y.dtype)
    with torch.no_grad():
        for block in torch.arange(len(y)).split(CHUNK):
            A = path_rao(model, y[block], context[block], t_hat, sigma, v)
            out[block] = cumulative_trapezoid(A, PATH_STEP)[:, at].cpu()
    return out


def glrt(model: Model, zc: torch.Tensor, context: torch.Tensor, t_hat: torch.Tensor,
         t_norm: float, sigma: float, theta_grid: tuple[float, int]) -> torch.Tensor:
    """One-sided GLRT 2 max(0, max_theta l(theta)) (n,) on the CPU, theta on the grid
    (theta_max, n_theta): n_theta uniform points of [0, theta_max] multiples of t (|t| = t_norm
    whitened units), an interior grid maximum refined by `refined_maximum`."""
    theta_max, n_theta = theta_grid
    v = torch.linspace(0, theta_max * t_norm, n_theta, device=zc.device)
    out = torch.zeros(len(zc), dtype=zc.dtype)
    with torch.no_grad():
        for block in torch.arange(len(zc)).split(CHUNK):
            ell = cumulative_trapezoid(path_rao(model, zc[block], context[block], t_hat, sigma, v),
                                       float(v[1] - v[0]))
            out[block] = 2 * refined_maximum(ell).clamp_min(0).cpu()
    return out


def refined_maximum(ell: torch.Tensor) -> torch.Tensor:
    """Row maxima (n,) of ell (n, K), an interior one refined by the parabola through it and its
    two neighbours when that parabola is concave. The grid starts at theta = 0, so a maximum there
    is kept as it is (Experiment 3's exact GLRT, `spatial_scene.refined_maximum`, also reads one
    step below 0)."""
    top, k = ell.max(1)
    inner = (k > 0) & (k < ell.shape[1] - 1)
    km = k.clamp(1, ell.shape[1] - 2)
    lo, mid, hi = (ell.gather(1, (km + d)[:, None])[:, 0] for d in (-1, 0, 1))
    curvature = 2 * mid - lo - hi
    refined = mid + (hi - lo) ** 2 / (8 * curvature.clamp_min(1e-12))
    return torch.where(inner & (curvature > 0), refined, top)
