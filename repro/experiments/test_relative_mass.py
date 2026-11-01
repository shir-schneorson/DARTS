"""Experiment 2's routes to a log ratio, with exact networks for a standard Gaussian background.

For p0 = N(0, 1) in one dimension, p_sigma = N(0, v), v = 1 + sigma^2, and with y = x + sigma eps:
E[eps | y] = sigma y / v, E[eps^2 | y] - 1 = sigma^2 y^2 / v^2 - sigma^2 / v, and the classifier's
logits are log p_sigma_k(y). Every route should give log p_s0(x2) - log p_s0(x1).
"""

import math

import pytest
import torch

from experiments import relative_mass as rm


class Exact(torch.nn.Module):
    """The exact `rm.MLP` stand-in for N(0, 1): the denoiser, the energy head or the logits."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        v = 1 + sigma[:, None] ** 2
        if self.kind == "dsm":
            return sigma[:, None] * y / v
        return sigma[:, None] ** 2 * (y.square() / v**2 - 1 / v)


def log_p(y: torch.Tensor, sigma: float) -> torch.Tensor:
    v = 1 + sigma**2
    return -0.5 * y[:, 0] ** 2 / v - 0.5 * math.log(2 * math.pi * v)


X1, X2 = torch.tensor([[-1.2], [0.3]]), torch.tensor([[0.8], [2.0]])


def test_direct_route_is_exact() -> None:
    got = rm.direct(Exact("dsm"), X1, X2, 0.3)
    assert torch.allclose(got, log_p(X2, 0.3) - log_p(X1, 0.3), atol=1e-6)


def test_rectangle_route() -> None:
    got = rm.rect_energy(Exact("dsm"), Exact("energy"), X1, X2, 0.3, 1.0)
    assert torch.allclose(got, log_p(X2, 0.3) - log_p(X1, 0.3), atol=1e-3)


def test_classifier_route_is_exact_for_a_standard_background() -> None:
    levels = torch.tensor([0.1, 0.3, 1.0, 100.0])

    def logits(y: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return torch.stack([log_p(y, float(s)) for s in lv], 1)

    got = rm.classifier(logits, levels, X1, X2, 0.3)
    assert torch.allclose(got, log_p(X2, 0.3) - log_p(X1, 0.3), atol=1e-5)


def test_itd_route() -> None:
    got = rm.itd(Exact("dsm"), X1, X2, n_mc=4096)
    assert got.tolist() == pytest.approx((log_p(X2, 0.0) - log_p(X1, 0.0)).tolist(), abs=0.05)
