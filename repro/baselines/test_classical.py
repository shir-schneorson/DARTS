"""The classical detectors against direct computations."""

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf

from baselines.classical import ace, cem, local_amf_ace, window_batches, window_indices
from dartsdet.whitening import amf


def test_cem_with_identity_correlation_is_a_projection() -> None:
    spectra = np.vstack([np.eye(3), -np.eye(3)]) * np.sqrt(3)  # R = I
    d = np.array([1.0, 2.0, 0.0])
    np.testing.assert_allclose(cem(spectra, d), spectra @ d / (d @ d))


def test_ace_is_the_amf_over_the_mahalanobis_norm() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 4)) @ rng.standard_normal((4, 4))
    t = rng.standard_normal(4)
    y = x - x.mean(0)
    mahalanobis = np.sqrt(np.einsum("nb,bc,nc->n", y, np.linalg.inv(np.cov(x.T, ddof=0)), y))
    np.testing.assert_allclose(ace(x, t), amf(x, t) / mahalanobis)


def test_windows_leave_out_the_guard_and_the_outside() -> None:
    index = window_indices((5, 4), 3, 1, torch.tensor([0, 9]))  # pixel 9: column 2, row 1
    assert (index[0] == -1).sum() == 5 and 0 not in index[0].tolist()
    assert sorted(index[1].tolist()) == [4, 5, 6, 8, 10, 12, 13, 14]


def test_local_filter_matches_the_direct_ledoit_wolf_amf() -> None:
    """With a window holding every allowed pixel, the Woodbury filters equal the AMF and ACE with
    that window's mean and Ledoit-Wolf covariance, computed directly."""
    rng = np.random.default_rng(1)
    W, H, B = 9, 9, 5
    fx = torch.from_numpy(rng.standard_normal((W * H, B)) @ rng.standard_normal((B, B)))
    t = torch.from_numpy(rng.standard_normal(B))
    allowed = torch.ones(W * H, dtype=torch.bool)
    centre = 4 * H + 4
    batches = window_batches(allowed, torch.arange(W * H) == centre, (W, H), 1, W * H - 1)
    local_amf, local_ace = local_amf_ace(fx, allowed, t, batches)
    others = np.delete(fx.numpy(), centre, axis=0)
    inverse = np.linalg.inv(ledoit_wolf(others)[0])
    w, tn = fx[centre].numpy() - others.mean(0), t.numpy()
    expected = tn @ inverse @ w / np.sqrt(tn @ inverse @ tn)
    np.testing.assert_allclose(local_amf[centre], expected, rtol=1e-10)
    np.testing.assert_allclose(local_ace[centre], expected / np.sqrt(w @ inverse @ w), rtol=1e-10)
    assert np.count_nonzero(local_amf) == 1
