"""The score model's parametrization, coordinates and checkpoints."""

import json
from pathlib import Path

import torch

from dartsdet.model import ScoreModel, Standardize, vp_coefficients

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "darts.json"


def small_model(n_bands: int = 6) -> ScoreModel:
    spec = json.loads(RECIPE.read_text())
    spec |= {"width": 16, "depth": 2}
    spec["encoder"] |= {"channels": 8, "context": 4}
    torch.manual_seed(0)
    return ScoreModel(Standardize(torch.zeros(n_bands), torch.ones(n_bands)), spec)


def test_vp_coefficients() -> None:
    sigma = torch.tensor([0.01, 0.5, 3.0])
    a, s = vp_coefficients(sigma)
    assert torch.allclose(a**2 + s**2, torch.ones(3))
    assert torch.allclose(s / a, sigma)


def test_untrained_model_predicts_the_gaussian_prior() -> None:
    # W_out = 0, so eps_hat = s noised: the exact denoiser when z ~ N(0, I).
    model = small_model()
    noised, context = torch.randn(10, 6), torch.randn(10, 4)
    _, s = vp_coefficients(torch.tensor(0.3))
    assert torch.allclose(model.epsilon(noised, context, 0.3), s * noised)


def test_standardize_fit() -> None:
    x = torch.randn(500, 3, dtype=torch.float64) * torch.tensor([1.0, 5.0, 0.1]) + 7
    z = Standardize.fit(x)(x)
    assert torch.allclose(z.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(z.std(0, correction=0), torch.ones(3, dtype=torch.float64))


def test_save_and_load(tmp_path: Path) -> None:
    model = small_model()
    model.best_step = 7
    model.save(tmp_path / "m.pt", seed=3)
    loaded = ScoreModel.load(tmp_path / "m.pt")
    assert loaded.best_step == 7
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_state_dict_keys_match_the_checkpoints() -> None:
    """The full recipe's keys, as the paper's fold checkpoints store them."""
    pairs = ("weight", "bias")
    expected = (["coordinates.mean", "coordinates.std"]
                + [f"encoder.stem.{i}.{p}" for i in (0, 2) for p in pairs]
                + [f"encoder.branch.{m}.{i}.{p}" for m in ("norms", "convs") for i in range(6)
                   for p in pairs]
                + [f"encoder.norm.{p}" for p in pairs]
                + [f"encoder.out.{i}.{p}" for i in (0, 2) for p in pairs]
                + ["network.embed.0.frequencies"]
                + [f"network.embed.{i}.{p}" for i in (1, 3) for p in pairs]
                + [f"network.input.{p}" for p in pairs]
                + [f"network.blocks.{i}.{m}.{p}" for i in range(3) for m in ("film", "linear")
                   for p in pairs]
                + [f"network.{m}.{p}" for m in ("norm", "output") for p in pairs])
    model = ScoreModel(Standardize(torch.zeros(5), torch.ones(5)), json.loads(RECIPE.read_text()))
    assert list(model.state_dict()) == expected
