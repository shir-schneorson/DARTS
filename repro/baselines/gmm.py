"""The Gaussian mixture baselines of Table II: GMM-Rao and GMM-GLRT, with global or window weights.

Fit: EM in the data's own whitened coordinates z = (x - m) U', m the mean and U the inverse
Cholesky factor of the covariance of the training spectra, float64. It follows sklearn's
GaussianMixture (a k-means start, tol 1e-3 nats, at most 100 iterations, the 10 eps guard on the
counts), with a plain k-means++ seed and Lloyd iterations until no label changes. `reg_covar` is
added to every covariance diagonal in z.

Detectors: along x - theta t, with q_k = (x - mu_k)' S_k^-1 (x - mu_k), b_k = t' S_k^-1 (x - mu_k)
and c_k = t' S_k^-1 t, component k's log density is log pi_k - log|S_k| / 2 - (q_k - 2 theta b_k
+ theta^2 c_k) / 2 up to a constant, so that

    Rao  = sum_k gamma_k(x) b_k,  gamma_k the responsibilities,
    GLRT = 2 max over 0 <= theta <= 100 of log p(x - theta t) - log p(x),

the GLRT on a grid of 2001 amplitudes, refined by a parabola. The window rows replace pi_k, for
pixel i, by the mean responsibility over the allowed pixels of i's window (`classical`'s).
"""

from collections.abc import Iterator
from typing import NamedTuple

import torch

KMEANS_ITER = 300  # sklearn's KMeans default; Lloyd stops once no assignment changes
EM_ITER, EM_TOL = 100, 1e-3  # sklearn's GaussianMixture defaults: iterations, nats
CHUNK = 4096       # spectra per E-step block; its sums accumulate per block, so it is fixed
THETA_MAX, N_THETA = 100.0, 2001  # the GLRT grid, in multiples of t
GLRT_CHUNK = 1024  # pixels per GLRT block; memory only


class GMM(NamedTuple):
    """sum_k pi_k N(z; mu_k, S_k) in whitened coordinates z = (x - shift) unmix'."""

    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    shift: torch.Tensor
    unmix: torch.Tensor
    converged: bool


def fit(x: torch.Tensor, n_components: int, seed: int, reg_covar: float) -> GMM:
    """Maximum likelihood by EM on spectra x (n, B), float64."""
    exact = x.double()
    lower = torch.linalg.cholesky(torch.cov(exact.T, correction=0))
    unmix = torch.linalg.solve_triangular(lower, torch.eye(len(lower), dtype=torch.float64),
                                          upper=False)
    shift = exact.mean(dim=0)
    z = (exact - shift) @ unmix.T
    centres = kmeans_plus_plus(z, n_components, torch.Generator().manual_seed(seed))
    labels, centres = lloyd(z, centres)
    statistics = None
    for block, block_labels in zip(z.split(CHUNK), labels.split(CHUNK), strict=True):
        resp = torch.nn.functional.one_hot(block_labels, n_components).to(z.dtype)
        moments = moments_about(block, resp, centres)
        statistics = moments if statistics is None else tuple(
            a + b for a, b in zip(statistics, moments, strict=True))
    weights, means, covariances = m_step(statistics, centres, reg_covar)
    history = []
    for _ in range(EM_ITER):
        statistics, log_likelihood = e_step(z, weights, means, covariances)
        weights, means, covariances = m_step(statistics, means, reg_covar)
        history.append(log_likelihood)
        if len(history) > 1 and abs(history[-1] - history[-2]) < EM_TOL:
            break
    converged = len(history) > 1 and abs(history[-1] - history[-2]) < EM_TOL
    return GMM(weights, means, covariances, shift, unmix, converged)


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

    def cluster_means(labels: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=len(centres)).to(z.dtype)
        sums = torch.zeros_like(centres).index_add_(0, labels, z)
        return torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centres)

    labels = squared_distances(z, centres).argmin(dim=1)
    for _ in range(KMEANS_ITER):
        centres = cluster_means(labels, centres)
        updated = squared_distances(z, centres).argmin(dim=1)
        if torch.equal(updated, labels):
            break
        labels = updated
    return labels, cluster_means(labels, centres)


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
    total = torch.zeros((), dtype=z.dtype)
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
    eye = torch.eye(means.shape[1], dtype=means.dtype)
    return count / count.sum(), means + step, covariances + reg_covar * eye


class Line(NamedTuple):
    """Each component's quadratic along x - theta t: q, b (P, K), c (K,), and the log normalizers
    -log|S_k| / 2 (K,)."""

    q: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    log_norm: torch.Tensor


def line(model: GMM, fx: torch.Tensor, t: torch.Tensor) -> Line:
    """The Line of whitened spectra fx (P, B) along the signature t (B,) under `model`."""
    z = (fx.double() - model.shift) @ model.unmix.T
    direction = t.double() @ model.unmix.T
    q, b, c = [], [], []
    factors = torch.linalg.cholesky(model.covariances)
    for mean, factor in zip(model.means, factors, strict=True):
        u = torch.linalg.solve_triangular(factor, (z - mean).T, upper=False)
        v = torch.linalg.solve_triangular(factor, direction[:, None], upper=False)
        q.append(u.square().sum(0))
        b.append((u * v).sum(0))
        c.append(v.square().sum())
    log_norm = -factors.diagonal(dim1=1, dim2=2).log().sum(1)
    return Line(torch.stack(q, 1), torch.stack(b, 1), torch.stack(c), log_norm)


def responsibilities(log_weights: torch.Tensor, terms: Line) -> torch.Tensor:
    """Posterior component probabilities (P, K) at theta = 0, under log weights (K,) or (P, K)."""
    return torch.softmax(log_weights + terms.log_norm - terms.q / 2, dim=1)


def window_weights(log_weights: torch.Tensor, terms: Line, allowed: torch.Tensor,
                   batches: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Log window weights (P, K): the mean responsibility of the `allowed` (P,) pixels of each
    window of `batches` (`classical.window_batches`), the global weights where a pixel has none."""
    gamma = responsibilities(log_weights, terms)
    weights = torch.full_like(gamma, torch.nan)
    for batch, index in batches:
        keep = ((index >= 0) & allowed.cpu()[index.clamp(min=0)]).double()
        weights[batch] = (gamma[index.clamp(min=0)] * keep[..., None]).sum(1) / keep.sum(1, True)
    return weights.where(~weights.isnan(), log_weights.exp()).log()


def rao(log_weights: torch.Tensor, terms: Line) -> torch.Tensor:
    """One-sided Rao statistic (P,)."""
    return (responsibilities(log_weights, terms) * terms.b).sum(1)


def glrt(log_weights: torch.Tensor, terms: Line) -> torch.Tensor:
    """One-sided GLRT (P,) on [0, THETA_MAX], N_THETA points plus one step below 0 as a
    neighbour, the maximum refined by `profile_maximum`."""
    theta = torch.arange(-1, N_THETA, dtype=torch.float64)[:, None] * THETA_MAX / (N_THETA - 1)
    out = []
    for start in range(0, len(terms.q), GLRT_CHUNK):
        rows = slice(start, start + GLRT_CHUNK)
        w, q, b = log_weights.expand_as(terms.q)[rows], terms.q[rows], terms.b[rows]
        delta = q[:, None] - 2 * theta * b[:, None] + theta**2 * terms.c
        base = w + terms.log_norm
        profile = ((base[:, None] - delta / 2).logsumexp(2)
                   - (base - q / 2).logsumexp(1, keepdim=True))
        out.append(2 * profile_maximum(profile, first=1))
    return torch.cat(out)


def profile_maximum(profile: torch.Tensor, first: int) -> torch.Tensor:
    """Row maxima (n,) of a log-ratio profile (n, G) over grid points `first`.., refined by the
    parabola through the grid maximum and its neighbours when it is concave, the maximum is not
    on the grid edge, and the vertex is not before `first`; earlier points are neighbours only."""
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
