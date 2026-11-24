"""The detection statistics of a trained model at one noise level sigma (Sections III and V).

Statistics are per pixel, read at standardized centres y (n, B) whose contexts c (n, C) are held
fixed (`contexts`: each pixel's, from its observed neighbours). In whitened units, along the
signature t = |t| t_hat:

    U_sigma(x | c) = -t_hat' s_sigma(x | c)                              eq. (U-sigma-cond)
    l(theta; x)    = log p_sigma(x - theta t | c) - log p_sigma(x | c)   eq. (llr)
    T_GLRT(x)      = 2 max over theta >= 0 of l(theta; x)                eq. (glrt)

U_sigma is the Rao statistic, eq. (rao), up to its normalization, which is the same for every
pixel. With eps_hat of eq. (cond-eps), the log density of the model in standardized units is
log p_sigma(y | c) = -psi(a y, c, sigma) / sigma - |a y|^2 / 2 + const, so l is a difference of
potentials and needs no path integral.
"""

import math

import torch

from dartsdet.model import DARTS, pixels, vp

CHUNK = 4096  # rows `log_ratio` reads at a time, for memory


def contexts(model: DARTS, z: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """c (P, C) of every pixel of the standardized image z (1, B, W, H) at level sigma, read from
    the clean a z at the pixels of `mask` (1, 1, W, H); in pixel order."""
    a, _ = vp(sigma)
    log_sigma = torch.full((1,), math.log(sigma), device=z.device)
    with torch.no_grad():
        return pixels(model.contexts(a * z, mask, log_sigma))


def rao(model: DARTS, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor,
        sigma: float) -> torch.Tensor:
    """U_sigma (n,) at centres y (n, B) with contexts c (n, C); t (B,) float64, the whitened
    signature. On y's device."""
    v = (t / t.norm()).float().to(y.device) / model.std
    a, _ = vp(sigma)
    return model.epsilon(a * y, c, sigma) @ v / sigma


def rao_gradient(model: DARTS, y: torch.Tensor, c: torch.Tensor, v: torch.Tensor,
                 sigma: float) -> torch.Tensor:
    """The gradient (n, B) with respect to the centres y (n, B) of eps_hat(a y, c, sigma)' v /
    sigma, the Rao statistic along v (B,) in standardized units, contexts c (n, C) held fixed: one
    Hessian-vector product of the potential per row. Its mean over background pixels estimates
    J t = E_0[grad U] (Section IV-E). On y's device."""
    a, s = vp(sigma)
    out = []
    for yb, cb in zip(y.split(CHUNK), c.split(CHUNK), strict=True):
        log_sigma = torch.full((len(yb),), math.log(sigma), device=y.device)
        with torch.enable_grad():
            yb = yb.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(model.head(a * yb, cb, log_sigma).sum(), yb,
                                          create_graph=True)
            (g,) = torch.autograd.grad(((grad + s * a * yb) @ v / sigma).sum(), yb)
        out.append(g)
    return torch.cat(out)


def log_ratio(model: DARTS, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor, sigma: float,
              distances: torch.Tensor) -> torch.Tensor:
    """l (n, 1 + K) float64 on the CPU at whitened distance 0 (column 0, where l = 0) and at
    `distances` (K,) along -t_hat, for `rao`'s arguments. With u(d) = a (y - d t_hat / std),

        l(d) = -[psi(u(d)) - psi(u(0))] / sigma - (|u(d)|^2 - |u(0)|^2) / 2."""
    a, s = vp(sigma)
    v = (t / t.norm()).float().to(y.device) / model.std
    d = torch.cat([torch.zeros(1), distances]).to(y.device)
    out = []
    with torch.no_grad():
        for yb, cb in zip(y.split(CHUNK), c.split(CHUNK), strict=True):
            u = a * (yb[:, None] - d[None, :, None] * v)  # (n, 1 + K, B)
            log_sigma = torch.full((len(yb) * len(d),), math.log(sigma), device=y.device)
            psi = model.head(u.flatten(0, 1), cb.repeat_interleave(len(d), 0), log_sigma)
            psi = psi.reshape(len(yb), len(d))
            ell = -(a / s) * (psi - psi[:, :1]) - 0.5 * ((u**2).sum(-1) - (u[:, :1] ** 2).sum(-1))
            out.append(ell.cpu().double())
    return torch.cat(out)


def glrt(model: DARTS, y: torch.Tensor, c: torch.Tensor, t: torch.Tensor, sigma: float,
         amplitudes: torch.Tensor) -> torch.Tensor:
    """T_GLRT (n,) float64 on the CPU, for `rao`'s arguments: the maximum of `log_ratio` over
    theta = 0 and `amplitudes` (G,), refined by `refined_maximum`."""
    ell = log_ratio(model, y, c, t, sigma, float(t.norm()) * amplitudes)
    return 2 * refined_maximum(ell).clamp_min(0)


def refined_maximum(ell: torch.Tensor) -> torch.Tensor:
    """Row maxima (n,) of ell (n, K), an interior one replaced by the vertex of the parabola
    through it and its two neighbours, in grid index, when that parabola is concave."""
    top, k = ell.max(1)
    inner = (k > 0) & (k < ell.shape[1] - 1)
    km = k.clamp(1, ell.shape[1] - 2)
    lo, mid, hi = (ell.gather(1, (km + d)[:, None])[:, 0] for d in (-1, 0, 1))
    curvature = 2 * mid - lo - hi
    refined = mid + (hi - lo) ** 2 / (8 * curvature.clamp_min(1e-12))
    return torch.where(inner & (curvature > 0), refined, top)
