"""Experiment 1's Gaussian mixture, fitted by EM in the data's own whitened coordinates.

The mixture lives in z = (x - m) U', with m the data mean and U the inverse Cholesky factor of the
data covariance, both in float64: an exact reparametrization that makes the fit independent of
the units of x. EM follows sklearn's GaussianMixture (a k-means start, tol 1e-3 nats, 100
iterations, the 10-eps guard on the counts) but seeds plain k-means++ and runs Lloyd until no
label changes. It is written in torch, rather than taken from sklearn, so that it shares the
whitened coordinates and float64 arithmetic of the rest of the experiment. `reg_covar` is added
to every covariance diagonal in z. The fit records whether EM converged.
"""

import torch

KMEANS_ITER = 300  # sklearn's KMeans default; Lloyd stops earlier once no assignment changes
CHUNK = 4096       # spectra per E-step block; its sums accumulate per block, so it is fixed


def cholesky_whitening(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(m (B,), U (B, B)) of spectra x (n, B), z = (x - m) U'; float64."""
    exact = x.detach().cpu().double()
    lower = torch.linalg.cholesky(torch.cov(exact.T, correction=0))
    unmix = torch.linalg.solve_triangular(lower, torch.eye(len(lower), dtype=torch.float64),
                                          upper=False)
    return exact.mean(dim=0).to(x), unmix.to(x)


class GMM:
    """sum_k pi_k N(z; mu_k, Sigma_k) in whitened coordinates z = (x - shift) unmix'."""

    def __init__(self, weights: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor,
                 coordinates: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.weights, self.means, self.covariances = weights, means, covariances
        self.shift, self.unmix = coordinates
        self.converged = True

    @classmethod
    def fit(cls, x: torch.Tensor, n_components: int, seed: int, reg_covar: float,
            max_iter: int = 100, tol: float = 1e-3) -> "GMM":
        """Maximum likelihood by EM on spectra x (n, B), float64."""
        coordinates = cholesky_whitening(x)
        z = (x - coordinates[0]) @ coordinates[1].T
        weights, means, covariances = initial(z, n_components, seed, reg_covar)
        history = []
        for _ in range(max_iter):
            statistics, log_likelihood = e_step(z, weights, means, covariances)
            weights, means, covariances = m_step(statistics, means, reg_covar)
            history.append(log_likelihood)
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                break
        model = cls(weights, means, covariances, coordinates)
        model.history = history
        model.converged = len(history) > 1 and abs(history[-1] - history[-2]) < tol
        return model

    def distribution(self) -> torch.distributions.MixtureSameFamily:
        """The mixture as a torch distribution over z."""
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=self.weights),
            torch.distributions.MultivariateNormal(self.means, covariance_matrix=self.covariances))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log p0(x) (n,), per unit volume of x: the density of z plus log |U|."""
        z = (x - self.shift) @ self.unmix.T
        return self.distribution().log_prob(z) + self.unmix.diagonal().abs().log().sum()

    def marginal_cdf(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(P(a'X <= a'x), P(a'X > a'x)), each (n, D), of the 1-D marginals along the rows of
        a (D, B); each component is N(a'mu_k, a' Sigma_k a) along a, and each tail is summed
        from its own normal tails."""
        pulled = torch.linalg.solve_triangular(self.unmix.T, a.T, upper=True)
        means = self.shift @ a.T + self.means @ pulled
        sds = torch.einsum("bd,kbc,cd->kd", pulled, self.covariances, pulled).sqrt()
        y = x @ a.T
        lower, upper = torch.zeros_like(y), torch.zeros_like(y)
        for weight, mean, sd in zip(self.weights, means, sds, strict=True):
            standard = (y - mean) / sd
            lower += weight * torch.special.ndtr(standard)
            upper += weight * torch.special.ndtr(-standard)
        return lower, upper


def initial(z: torch.Tensor, n_components: int, seed: int,
            reg_covar: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weights, means and covariances of the k-means clusters of z (n, B)."""
    centres = kmeans_plus_plus(z, n_components, torch.Generator().manual_seed(seed))
    labels, centres = lloyd(z, centres)
    statistics = None
    for block, block_labels in zip(z.split(CHUNK), labels.split(CHUNK), strict=True):
        resp = torch.nn.functional.one_hot(block_labels, n_components).to(z.dtype)
        moments = moments_about(block, resp, centres)
        statistics = moments if statistics is None else tuple(
            a + b for a, b in zip(statistics, moments, strict=True))
    return m_step(statistics, centres, reg_covar)


def squared_distances(z: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """||z_n - c_k||^2 (n, K)."""
    return (z.square().sum(dim=1, keepdim=True) - 2 * z @ centres.T
            + centres.square().sum(dim=1)).clamp_min(0)


def kmeans_plus_plus(z: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """k centres (k, B): the first uniform, each next drawn with probability proportional to its
    squared distance from the nearest centre so far."""
    first = int(torch.randint(len(z), (1,), generator=generator))
    centres = z[first:first + 1]
    nearest = squared_distances(z, centres)[:, 0]
    for _ in range(k - 1):
        draw = int(torch.multinomial(nearest.cpu().double(), 1, generator=generator))
        centres = torch.cat([centres, z[draw:draw + 1]])
        nearest = torch.minimum(nearest, squared_distances(z, centres[-1:])[:, 0])
    return centres


def lloyd(z: torch.Tensor, centres: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(labels (n,), centres (K, B)) after Lloyd iterations; an empty cluster keeps its centre,
    and the centres returned are the means of the labels returned."""
    labels = squared_distances(z, centres).argmin(dim=1)
    for _ in range(KMEANS_ITER):
        centres = cluster_means(z, labels, centres)
        updated = squared_distances(z, centres).argmin(dim=1)
        if torch.equal(updated, labels):
            break
        labels = updated
    return labels, cluster_means(z, labels, centres)


def cluster_means(z: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Each cluster's mean (K, B); an empty cluster keeps its current centre."""
    counts = torch.bincount(labels, minlength=len(centres)).to(z.dtype)
    sums = torch.zeros_like(centres).index_add_(0, labels, z)
    return torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centres)


def moments_about(z: torch.Tensor, resp: torch.Tensor,
                  means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Responsibility-weighted count (K,), first (K, B) and second (K, B, B) moments about
    `means`, which keeps the covariance update free of cancellation."""
    offsets = z[:, None, :] - means
    return (resp.sum(dim=0), torch.einsum("nk,nkb->kb", resp, offsets),
            torch.einsum("nk,nkb,nkc->kbc", resp, offsets, offsets))


def e_step(z: torch.Tensor, weights: torch.Tensor, means: torch.Tensor,
           covariances: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], float]:
    """Moments about the current means under the responsibilities, and the mean log-likelihood."""
    components = torch.distributions.MultivariateNormal(means, covariance_matrix=covariances)
    statistics = (torch.zeros_like(weights), torch.zeros_like(means),
                  torch.zeros_like(covariances))
    total = torch.zeros((), dtype=z.dtype, device=z.device)
    for block in z.split(CHUNK):
        log_joint = components.log_prob(block[:, None, :]) + weights.log()
        norm = torch.logsumexp(log_joint, dim=1)
        resp = (log_joint - norm[:, None]).exp()
        statistics = tuple(a + b for a, b in
                           zip(statistics, moments_about(block, resp, means), strict=True))
        total += norm.sum()
    return statistics, float(total) / len(z)


def m_step(statistics: tuple[torch.Tensor, ...], means: torch.Tensor,
           reg_covar: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weights, means and covariances from the E-step statistics, with sklearn's 10 eps guard
    on the counts."""
    count, first, second = statistics
    count = count + 10 * torch.finfo(count.dtype).eps
    step = first / count[:, None]
    covariances = second / count[:, None, None] - step[:, :, None] * step[:, None, :]
    covariances = (covariances + covariances.transpose(1, 2)) / 2
    eye = torch.eye(means.shape[1], dtype=means.dtype, device=means.device)
    return count / count.sum(), means + step, covariances + reg_covar * eye
