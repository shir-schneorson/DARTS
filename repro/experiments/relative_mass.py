"""Experiment 2's networks and estimators of a log ratio log p_s0(x2) - log p_s0(x1).

The data are standardized. Noise is variance-exploding, y = x + sigma eps; the networks see
a y with a = (1 + sigma^2)^(-1/2), plus Fourier features of log sigma:

    DSM          eps_hat(y, sigma) ~ E[eps | y], so s_sigma(y) = -eps_hat / sigma
    energy       e2_hat(y, sigma) ~ E[|eps|^2 | y] - d, the noise-energy head
    classifier   f_k(y) = log N(y; 0, (1 + sigma_k^2) I) + g(a_k y, sigma_k), logits over fixed
                 levels; at the optimum f_k - f_j = log p_sigma_k - log p_sigma_j

and the routes to the log ratio:

    direct       the integral of s_s0 along the segment from x1 to x2 (the GLRT's path)
    rect_energy  a rectangle through s1: up from s0 to s1 at x1, across at s1, down at x2, the
                 vertical legs integrating e2_hat over log sigma (the heat equation)
    classifier   the vertical legs from the classifier, up to its top level, where p is close to
                 N(0, (1 + s_top^2) I) and the horizontal leg is analytic
    itd          pointwise I-MMSE to s0 = 0 from the denoiser's errors, with common noise draws
"""

import math
from collections.abc import Callable

import torch

SIGMA_LOW, SIGMA_HIGH = 0.005, 100.0  # the training range of the DSM and energy networks


def features(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """[a y, sin and cos of log(sigma) k / 4 for k = 1..8] (n, d + 16)."""
    a = (1 + sigma**2).rsqrt()[:, None]
    frequencies = torch.arange(1, 9, dtype=y.dtype, device=y.device)
    phase = sigma.log()[:, None] * frequencies / 4
    return torch.cat([a * y, phase.sin(), phase.cos()], 1)


class MLP(torch.nn.Module):
    """A SiLU network of `features`, (y (n, d), sigma (n,)) -> (n, out)."""

    def __init__(self, d: int, out: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers, w = [], d + 16
        for _ in range(depth):
            layers += [torch.nn.Linear(w, width), torch.nn.SiLU()]
            w = width
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(w, out))

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(features(y, sigma))


def logits(model: MLP, y: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """The classifier's f_k(y) at every level, (n, K)."""
    n, d, k = len(y), y.shape[1], len(levels)
    sigma = levels.repeat(n)
    repeated = y.repeat_interleave(k, 0)
    var = 1 + sigma**2
    gauss = -0.5 * (repeated.square().sum(1) / var + d * torch.log(2 * math.pi * var))
    return (gauss + model(repeated, sigma)[:, 0]).reshape(n, k)


def train(model: MLP, x: torch.Tensor, kind: str, steps: int, seed: int,
          levels: torch.Tensor | None = None, batch: int = 512) -> MLP:
    """Fit `model` on data x (n, d) as kind "dsm", "energy" or "classifier", with Adam and a
    cosine schedule, every draw from `seed`."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(steps):
        xb = x[torch.randint(len(x), (batch,), generator=generator)]
        eps = torch.randn(xb.shape, generator=generator)
        if kind == "classifier":
            k = torch.randint(len(levels), (batch,), generator=generator)
            loss = torch.nn.functional.cross_entropy(logits(model, xb + levels[k, None] * eps,
                                                            levels), k)
        else:
            sigma = SIGMA_LOW * (SIGMA_HIGH / SIGMA_LOW) ** torch.rand(batch, generator=generator)
            target = eps if kind == "dsm" else eps.square().sum(1, keepdim=True) - x.shape[1]
            loss = (model(xb + sigma[:, None] * eps, sigma) - target).square().sum(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()


def score(dsm: MLP, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """s_sigma(y) = -eps_hat / sigma (n, d)."""
    return -dsm(y, torch.full((len(y),), sigma, dtype=y.dtype)) / sigma


def direct(dsm: MLP, x1: torch.Tensor, x2: torch.Tensor, sigma: float) -> torch.Tensor:
    """The integral of s_sigma along the segment from x1 to x2 (n,), trapezoid on 129 points."""
    u = torch.linspace(0, 1, 129, dtype=x1.dtype)
    step = x2 - x1
    values = torch.stack([(score(dsm, x1 + ui * step, sigma) * step).sum(1) for ui in u], 1)
    return torch.trapezoid(values, u, dim=1)


def vertical_energy(energy: MLP, y: torch.Tensor, sigma0: float, sigma1: float,
                    n: int = 64) -> torch.Tensor:
    """log p_sigma1(y) - log p_sigma0(y) (n,), the integral of e2_hat over log sigma."""
    sigmas = torch.exp(torch.linspace(math.log(sigma0), math.log(sigma1), n))
    values = torch.stack([energy(y, torch.full((len(y),), float(s)))[:, 0] for s in sigmas], 1)
    return torch.trapezoid(values, sigmas.log(), dim=1)


def rect_energy(dsm: MLP, energy: MLP, x1: torch.Tensor, x2: torch.Tensor,
                sigma0: float, sigma1: float) -> torch.Tensor:
    """The rectangle route through sigma1 (n,)."""
    return (vertical_energy(energy, x1, sigma0, sigma1) + direct(dsm, x1, x2, sigma1)
            - vertical_energy(energy, x2, sigma0, sigma1))


def classifier(fk: Callable, levels: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
               sigma0: float) -> torch.Tensor:
    """The classifier route (n,); fk(y, levels) gives the logits (n, K)."""
    k0, top = int((levels - sigma0).abs().argmin()), float(levels[-1])

    def up(y: torch.Tensor) -> torch.Tensor:
        f = fk(y, levels)
        return f[:, -1] - f[:, k0]

    def gauss(y: torch.Tensor) -> torch.Tensor:
        return -0.5 * y.square().sum(1) / (1 + top**2)

    return up(x1) + (gauss(x2) - gauss(x1)) - up(x2)


@torch.no_grad()
def itd(dsm: MLP, x1: torch.Tensor, x2: torch.Tensor, n_sigma: int = 64,
        n_mc: int = 64) -> torch.Tensor:
    """log p(x2) - log p(x1) (n,) at sigma0 = 0: the integral over log sigma of
    e(x1, sigma) - e(x2, sigma), e the denoiser's mean squared error, common draws (seed 0)."""
    generator = torch.Generator().manual_seed(0)
    sigmas = torch.exp(torch.linspace(math.log(SIGMA_LOW), math.log(SIGMA_HIGH), n_sigma))
    diffs = []
    for s in sigmas:
        eps = torch.randn((n_mc, *x1.shape), generator=generator)
        levels = torch.full((n_mc * len(x1),), float(s))

        def error(x: torch.Tensor, eps: torch.Tensor = eps, s: torch.Tensor = s,
                  levels: torch.Tensor = levels) -> torch.Tensor:
            y = (x[None] + s * eps).reshape(-1, x.shape[1])
            residual = eps.reshape(-1, x.shape[1]) - dsm(y, levels)
            return residual.square().sum(1).reshape(n_mc, -1).mean(0)

        diffs.append(error(x1) - error(x2))
    return torch.trapezoid(torch.stack(diffs, 1), sigmas.log(), dim=1)
