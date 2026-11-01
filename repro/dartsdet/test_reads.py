"""The reads against a Gaussian background, where every one has a closed form.

For p0 = N(mu, S) in standardized coordinates z = (x - m) / std, the exact denoiser of the
variance-preserving kernel is eps_hat(a y) = sigma (S + sigma^2 I)^-1 (y - mu), and in whitened
units, with P = D^-1 (S + sigma^2 I)^-1 D^-1 and D = diag(std):

    A = t' D^-1 (S + sigma^2 I)^-1 (z - mu)     (the AMF of the smoothed Gaussian)
    C = t' P t
    l(theta) = theta A - theta^2 C / 2
    GLRT = A^2 / C if A > 0, else 0
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dartsdet import reads


class GaussianDenoiser(torch.nn.Module):
    """The exact eps_hat of a Gaussian background, context ignored."""

    def __init__(self, mu: torch.Tensor, cov: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mu, self.cov = mu, cov
        self.coordinates = SimpleNamespace(std=std)

    def epsilon(self, noised: torch.Tensor, context: torch.Tensor,  # noqa: ARG002
                sigma: float) -> torch.Tensor:
        a = (1 + sigma**2) ** -0.5
        smoothed = self.cov + sigma**2 * torch.eye(len(self.mu), dtype=self.cov.dtype)
        return sigma * torch.linalg.solve(smoothed, (noised / a - self.mu).T).T


@pytest.fixture
def setup() -> tuple:
    g = torch.Generator().manual_seed(0)
    n_bands = 5
    root = torch.randn(n_bands, n_bands, generator=g, dtype=torch.float64)
    cov = root @ root.T / n_bands + 0.5 * torch.eye(n_bands, dtype=torch.float64)
    mu = torch.randn(n_bands, generator=g, dtype=torch.float64)
    std = 0.5 + torch.rand(n_bands, generator=g, dtype=torch.float64)
    model = GaussianDenoiser(mu, cov, std)
    z = mu + torch.randn(200, n_bands, generator=g, dtype=torch.float64)
    t = torch.randn(n_bands, generator=g, dtype=torch.float64)
    return model, z, torch.zeros(len(z), 0, dtype=torch.float64), t / t.norm()


def closed_form(model: GaussianDenoiser, z: torch.Tensor, t_hat: torch.Tensor,
                sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(A (n,), C scalar) of the smoothed Gaussian."""
    inverse = torch.linalg.inv(model.cov + sigma**2 * torch.eye(len(model.mu), dtype=z.dtype))
    w = t_hat / model.coordinates.std
    return (z - model.mu) @ inverse @ w, w @ inverse @ w


@pytest.mark.parametrize("sigma", [0.05, 0.3, 2.0])
def test_rao_and_curvature(setup: tuple, sigma: float) -> None:
    model, z, context, t_hat = setup
    A, C = closed_form(model, z, t_hat, sigma)
    assert torch.allclose(reads.rao(model, z, context, t_hat, sigma), A)
    A_read, C_read = reads.rao_and_curvature(model, z, context, t_hat, [sigma])
    assert torch.allclose(A_read[:, 0], A)
    assert torch.allclose(C_read[:, 0], C.expand(len(z)))


def test_log_ratio_is_quadratic(setup: tuple) -> None:
    model, z, context, t_hat = setup
    sigma, thetas = 0.3, (0.25, 1.0, 4.0)
    A, C = closed_form(model, z, t_hat, sigma)
    expected = torch.stack([theta * A - theta**2 * C / 2 for theta in thetas], 1)
    got = reads.log_ratio(model, z, context, t_hat, sigma, thetas)
    assert torch.allclose(got, expected, atol=1e-9)


def test_log_ratio_refuses_off_grid_amplitudes(setup: tuple) -> None:
    model, z, context, t_hat = setup
    with pytest.raises(ValueError, match="not on the"):
        reads.log_ratio(model, z, context, t_hat, 0.3, (0.33,))


def test_glrt_is_one_sided_squared_amf(setup: tuple) -> None:
    model, z, context, t_hat = setup
    sigma = 0.3
    A, C = closed_form(model, z, t_hat, sigma)
    expected = torch.where(A > 0, A**2 / C, torch.zeros_like(A))
    theta_max, n_theta = 10.0, 250
    got = reads.glrt(model, z, context, t_hat, 1.0, sigma, (theta_max, n_theta))
    # The parabola through three grid points of a quadratic l is exact, wherever the maximizer
    # theta* = A / C has a grid point on either side of its grid maximum.
    step = theta_max / (n_theta - 1)
    theta_star = A / C
    resolved = ((theta_star > step) & (theta_star < theta_max - step)) | (A <= 0)
    assert resolved.sum() > 150
    np.testing.assert_allclose(got[resolved].numpy(), expected[resolved].numpy(), rtol=1e-6)
