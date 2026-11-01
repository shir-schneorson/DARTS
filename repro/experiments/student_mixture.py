"""Mixture of multivariate Student-t distributions: a heavy-tailed background with an exact score.

    p0(x) = sum_k pi_k t_nu(x; mu_k, Sigma_k),   Sigma_k = L_k L_k'
    log t_nu(x; mu, Sigma) = log Gamma((nu + B)/2) - log Gamma(nu/2) - (B/2) log(nu pi)
                             - (1/2) log|Sigma| - ((nu + B)/2) log(1 + q/nu),
    q = (x - mu)' Sigma^-1 (x - mu)

It is a Gaussian scale mixture: a draw of component k at scale w ~ chi2_nu / nu is
N(mu_k, Sigma_k / w). In Experiment 3's scene a region carries one (k, w), so this is each pixel's
marginal while N(mu_k, Sigma_k / w) is its density given the region. Experiment 1 fits it by
maximum likelihood (ECME), starting from the Gaussian mixture of `gmm` in that fit's whitened
coordinates. All components share nu. Everything is float64 on the CPU.
"""

import math

import numpy as np
import torch
from scipy import stats

from experiments.gmm import GMM


class StudentMixture:
    """Weights (K,), locations (K, B), scale Cholesky factors (K, B, B), common dof, in the
    units of x."""

    def __init__(self, weights: torch.Tensor, means: torch.Tensor, scale_tril: torch.Tensor,
                 dof: float) -> None:
        self.weights, self.means, self.scale_tril, self.dof = weights, means, scale_tril, dof
        self.converged = True

    @classmethod
    def from_covariances(cls, weights: torch.Tensor, means: torch.Tensor,
                         covariances: torch.Tensor, dof: float) -> "StudentMixture":
        """The mixture with scale matrices `covariances` (K, B, B)."""
        return cls(weights, means, torch.linalg.cholesky(covariances), dof)

    @classmethod
    def fit(cls, x: torch.Tensor, n_components: int, seed: int, reg_covar: float,
            dof_grid: torch.Tensor, max_iter: int = 1000, tol: float = 1e-6) -> "StudentMixture":
        """Maximum likelihood by ECME on spectra x (n, B): each iteration takes the dof on
        `dof_grid` that maximizes the observed log-likelihood, then updates the weights,
        locations and scales by EM with the scale-mixture weights (nu + B) / (nu + q), until the
        mean log-likelihood changes by less than `tol` nats or `max_iter` iterations have run;
        `converged` records which."""
        start = GMM.fit(x, n_components, seed, reg_covar)
        z, n_bands = (x - start.shift) @ start.unmix.T, x.shape[1]
        weights, means, scales = start.weights, start.means, start.covariances
        history = []
        for _ in range(max_iter):
            q, half_log_det = mahalanobis(z, means, scales)
            fits = torch.stack([log_joint(nu, weights, q, half_log_det, n_bands)
                                .logsumexp(0).mean() for nu in dof_grid])
            dof = float(dof_grid[fits.argmax()])
            history.append(float(fits.max()))
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                break
            weights, means, scales = em_step(z, dof, weights, q, half_log_det, reg_covar)
        eye = torch.eye(n_bands, dtype=x.dtype, device=x.device)
        unwhiten = torch.linalg.solve_triangular(start.unmix, eye, upper=False)
        model = cls(weights, start.shift + means @ unwhiten.T,
                    unwhiten @ torch.linalg.cholesky(scales), dof)
        model.history = history
        model.converged = len(history) > 1 and abs(history[-1] - history[-2]) < tol
        return model

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log p0(x) (n,) of spectra x (n, B)."""
        nu, n_bands = self.dof, x.shape[1]
        centred = (x[None, :, :] - self.means[:, None, :]).transpose(1, 2)
        q = torch.linalg.solve_triangular(self.scale_tril, centred, upper=False).square().sum(1)
        log_det = 2 * self.scale_tril.diagonal(dim1=1, dim2=2).log().sum(1)
        constant = (math.lgamma((nu + n_bands) / 2) - math.lgamma(nu / 2)
                    - n_bands / 2 * math.log(nu * math.pi))
        log_t = constant - log_det[:, None] / 2 - (nu + n_bands) / 2 * torch.log1p(q / nu)
        return torch.logsumexp(log_t + self.weights.log()[:, None], dim=0)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """grad log p0(x) (n, B), by autograd."""
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            return torch.autograd.grad(self.log_prob(x).sum(), x)[0]

    def sample(self, components: torch.Tensor, scales: torch.Tensor,
               generator: torch.Generator) -> torch.Tensor:
        """One draw of N(mu_k, Sigma_k / w) per given (k, w), (n, B)."""
        noise = torch.randn(len(components), self.means.shape[1], 1, generator=generator,
                            dtype=self.means.dtype)
        spread = (self.scale_tril[components] @ noise)[..., 0] / scales.sqrt()[:, None]
        return self.means[components] + spread

    def conditional_score(self, x: torch.Tensor, components: torch.Tensor,
                          scales: torch.Tensor) -> torch.Tensor:
        """grad log N(x; mu_k, Sigma_k / w) (n, B) at each spectrum's (k, w)."""
        centred = (x - self.means[components])[..., None]
        return -scales[:, None] * torch.cholesky_solve(centred, self.scale_tril[components])[..., 0]

    def marginal_cdf(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(P(a'X <= a'x), P(a'X > a'x)), each (n, D), of the 1-D marginals along the rows of
        a (D, B): along a, component k is a univariate t_nu with location a'mu_k and scale
        sqrt(a' Sigma_k a); each tail is summed from scipy's own t tails."""
        locations = (self.means @ a.T).numpy()
        spreads = (self.scale_tril.transpose(1, 2) @ a.T).square().sum(1).sqrt().numpy()
        y = (x @ a.T).numpy()
        lower, upper = np.zeros_like(y), np.zeros_like(y)
        for weight, location, spread in zip(self.weights.numpy(), locations, spreads, strict=True):
            standard = (y - location) / spread
            lower += weight * stats.t.cdf(standard, self.dof)
            upper += weight * stats.t.sf(standard, self.dof)
        return torch.from_numpy(lower), torch.from_numpy(upper)


def mahalanobis(z: torch.Tensor, means: torch.Tensor,
                scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q_kn = (z_n - mu_k)' Sigma_k^-1 (z_n - mu_k) (K, n), and (1/2) log|Sigma_k| (K, 1)."""
    tril = torch.linalg.cholesky(scales)
    q = torch.stack([torch.linalg.solve_triangular(tril[k], (z - means[k]).T, upper=False)
                     .square().sum(0) for k in range(len(means))])
    return q, tril.diagonal(dim1=1, dim2=2).log().sum(1)[:, None]


def log_joint(nu: torch.Tensor | float, weights: torch.Tensor, q: torch.Tensor,
              half_log_det: torch.Tensor, n_bands: int) -> torch.Tensor:
    """log pi_k + log t_nu(z_n; mu_k, Sigma_k) (K, n), from `mahalanobis`."""
    nu = torch.as_tensor(nu, dtype=q.dtype)
    constant = (torch.lgamma((nu + n_bands) / 2) - torch.lgamma(nu / 2)
                - n_bands / 2 * torch.log(nu * math.pi))
    tail = (nu + n_bands) / 2 * torch.log1p(q / nu)
    return weights.log()[:, None] + constant - half_log_det - tail


def em_step(z: torch.Tensor, dof: float, weights: torch.Tensor, q: torch.Tensor,
            half_log_det: torch.Tensor, reg_covar: float) -> tuple[torch.Tensor, ...]:
    """Weights, locations and scales after one EM update at a fixed dof."""
    n_bands = z.shape[1]
    resp = torch.softmax(log_joint(dof, weights, q, half_log_det, n_bands), dim=0)
    pulled = resp * (dof + n_bands) / (dof + q)
    means = (pulled @ z) / pulled.sum(1, keepdim=True)
    scales = torch.stack([(pulled[k, :, None] * (z - means[k])).T @ (z - means[k]) / resp[k].sum()
                          for k in range(len(means))])
    eye = torch.eye(n_bands, dtype=z.dtype, device=z.device)
    return resp.mean(1), means, scales + reg_covar * eye
