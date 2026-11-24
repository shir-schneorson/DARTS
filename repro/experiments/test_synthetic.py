"""The synthetic scene: implants where they belong, and the oracles in the Gaussian limit."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dartsdet import whitening
from dartsdet.metrics import PF
from experiments import synthetic

CONFIG = json.loads((Path(__file__).parents[1] / "configs" / "experiment2.json").read_text())


def small(dof: float = 4.0) -> synthetic.Synthetic:
    config = CONFIG | {"scene": {"side": 30, "regions": 6, "dof": dof},
                       "background": CONFIG["background"] | {"components": 1}}
    return synthetic.draw(config, 0)


def test_implants_are_alpha_d_at_the_sites_only() -> None:
    scene = small()
    clean, implanted = synthetic.implanted(scene, 0.0), synthetic.implanted(scene, 0.5)
    difference = implanted.spectra() - clean.spectra()
    sites = scene.sites.numpy()
    np.testing.assert_allclose(difference[sites], 0.5 * scene.draws.numpy())
    assert not np.delete(difference, sites, axis=0).any()
    assert (implanted.labels[sites] == 1).all() and implanted.labels.sum() == len(sites)


def test_oracles_in_the_gaussian_limit() -> None:
    """One component and nu -> infinity: the scene density is N(mu, S), so the Rao test is
    t' S^-1 (x - mu), the GLRT (t' S^-1 (x - mu))^2 / t' S^-1 t where its amplitude is inside the
    grid and positive, and with w = 1 the region Rao oracle that score."""
    scene = replace(small(1e12), scale=torch.ones(900, dtype=torch.float64))
    x = torch.from_numpy(synthetic.implanted(scene, 1.0).spectra())
    precision = torch.cholesky_inverse(scene.scale_tril[0])
    u = (x - scene.means[0]) @ precision @ scene.target
    information = scene.target @ precision @ scene.target
    torch.testing.assert_close(synthetic.oracle_rao(scene, x), u, rtol=1e-6, atol=1e-8)
    glrt = synthetic.oracle_glrt(scene, x, 10.0, 501)
    inside = (u / information > 0.05) & (u / information < 9.9)
    torch.testing.assert_close(glrt[inside], (u**2 / information)[inside], rtol=1e-5, atol=1e-6)
    assert (glrt[u < 0] == 0).all() and inside.sum() > 10
    torch.testing.assert_close(synthetic.region_score(scene, x), u)


def test_region_oracles_scale_with_w() -> None:
    """Given its region a pixel is N(mu_k, S_k / w): the score scales with w."""
    scene = small()
    x = torch.from_numpy(synthetic.implanted(scene, 1.0).spectra())
    w = scene.scale
    precision = torch.cholesky_inverse(scene.scale_tril[0])
    u = (x - scene.means[0]) @ precision @ scene.target
    assert len(w.unique()) > 1
    torch.testing.assert_close(synthetic.region_score(scene, x), w * u)


def test_smoothed_region_oracle_limits() -> None:
    """In whitened units the smoothed score at sigma = 0 is the region Rao oracle (a change of
    units), and sigma^2 U_sigma tends to its sigma = inf limit t' (y - m_r)."""
    scene = small()
    x = torch.from_numpy(synthetic.implanted(scene, 1.0).spectra())
    fitted = whitening.whitening(torch.from_numpy(synthetic.implanted(scene, 0.0).spectra()))
    torch.testing.assert_close(synthetic.region_smoothed(scene, fitted, x, 0.0),
                               synthetic.region_score(scene, x), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(1e8 * synthetic.region_smoothed(scene, fitted, x, 1e4),
                               synthetic.region_smoothed(scene, fitted, x, float("inf")),
                               rtol=1e-6, atol=1e-6)


def test_matched_level_closed_form() -> None:
    """One component, w = 1, whitened by S^-1/2 about its mean: C_r = I and omega = 1, so
    U_sigma = U_0 / (1 + sigma^2), kappa_sigma sigma^2 = kappa_0 sigma^2 / (1 + sigma^2), and the
    level matched to c = theta |t|^2 / 2 = f kappa_0 is sigma^2 = f / (1 - f), none for f >= 1."""
    scene = replace(small(1e12), scale=torch.ones(900, dtype=torch.float64))
    values, vectors = torch.linalg.eigh(scene.scale_tril[0] @ scene.scale_tril[0].T)
    fitted = (scene.means[0], vectors @ torch.diag(values.rsqrt()) @ vectors.T)
    clean = synthetic.implanted(scene, 0.0)
    u0 = synthetic.region_smoothed(scene, fitted, torch.from_numpy(clean.spectra()), 0.0)
    k0 = float(np.quantile(u0.numpy()[clean.labels == 0], 1 - PF))
    t = scene.target @ fitted[1]
    shares = [0.2, 0.5, 0.9, 1.1]
    thetas = [2 * f * k0 / float(t @ t) for f in shares]
    levels = synthetic.matched_levels(scene, fitted, thetas, CONFIG["matched_level_grid"])
    np.testing.assert_allclose(levels[:3], [np.sqrt(f / (1 - f)) for f in shares[:3]], rtol=1e-5)
    assert levels[3] == float("inf")
