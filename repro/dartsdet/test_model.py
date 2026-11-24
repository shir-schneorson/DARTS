"""The model: the encoder's receptive field, eq. (cond-eps) per pixel, and saved checkpoints."""

import json
from pathlib import Path

import pytest
import torch

from dartsdet.model import DARTS, vp

SPEC = json.loads((Path(__file__).parents[1] / "configs" / "darts.json").read_text())["model"]


def random_model(n_bands: int) -> DARTS:
    """The recipe's model in float64 with a nonzero potential and sigma modulation."""
    torch.manual_seed(0)
    model = DARTS(n_bands, SPEC).double()
    for layer in (model.head.output, model.film):
        torch.nn.init.normal_(layer.weight, std=0.3)
    return model


@pytest.mark.parametrize("shape", [(40, 40), (40, 37)])  # square: one batch of rotations; else two
def test_context_reads_the_window_but_not_the_pixel_or_corners(shape: tuple[int, int]) -> None:
    """The context at (i, j) has a nonzero gradient on exactly the 25 x 25 window around it minus
    the pixel and the window's four corners."""
    model = random_model(3)
    image = torch.randn(1, 3, *shape, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1, *shape, dtype=torch.float64)
    i, j = 20, 18
    c = model.contexts(image, mask, torch.zeros(1, dtype=torch.float64))
    (grad,) = torch.autograd.grad(c[..., i, j].sum(), image)
    expected = torch.zeros(shape, dtype=torch.bool)
    expected[i - 12:i + 13, j - 12:j + 13] = True
    for di, dj in ((0, 0), (-12, -12), (-12, 12), (12, -12), (12, 12)):
        expected[i + di, j + dj] = False
    assert torch.equal(grad.abs().sum(dim=(0, 1)) > 0, expected)


def test_context_ignores_the_spectra_of_masked_pixels() -> None:
    model = random_model(3)
    image = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    mask = torch.ones(1, 1, 16, 16, dtype=torch.float64)
    mask[..., 3, 4] = 0
    moved = image.clone()
    moved[..., 3, 4] += 10.0
    log_sigma = torch.zeros(1, dtype=torch.float64)
    before, after = (model.contexts(x, mask, log_sigma) for x in (image, moved))
    assert torch.allclose(after, before)


def test_untrained_model_predicts_the_gaussian_noise() -> None:
    """psi = 0 at initialization, so eps_hat = s u: the score of a standard Gaussian."""
    torch.manual_seed(0)
    model = DARTS(5, SPEC).double()
    u, c = torch.randn(30, 5 + SPEC["context"], dtype=torch.float64).split([5, SPEC["context"]], 1)
    _, s = vp(0.3)
    assert torch.equal(model.epsilon(u, c, 0.3), s * u)


def test_epsilon_is_the_rows_potential_gradient() -> None:
    """eps_hat = a dpsi/du + s u, row by row: a central difference of the head agrees, at one level
    for every row (a float) and at one level per row (a tensor)."""
    model = random_model(4)
    u, c = torch.randn(6, 4 + SPEC["context"], dtype=torch.float64).split([4, SPEC["context"]], 1)
    sigma = torch.linspace(0.1, 2.0, 6, dtype=torch.float64)
    step = 1e-6
    difference = torch.stack([(model.head(u + step * e, c, sigma.log()) -
                               model.head(u - step * e, c, sigma.log())) / (2 * step)
                              for e in torch.eye(4, dtype=torch.float64)], 1)
    a, s = vp(sigma[:, None])
    torch.testing.assert_close(model.epsilon(u, c, sigma), a * difference + s * u, rtol=1e-6,
                               atol=1e-9)
    first = u[:1], c[:1]
    torch.testing.assert_close(model.epsilon(*first, 0.1), model.epsilon(*first, sigma[:1]))


def test_saved_model_loads_unchanged(tmp_path: Path) -> None:
    model = random_model(3).float()
    model.mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    model.save(tmp_path / "model.pt", step=500)
    loaded = DARTS.load(tmp_path / "model.pt")
    for (name, value), other in zip(model.state_dict().items(), loaded.state_dict().values(),
                                    strict=True):
        assert torch.equal(value, other), name


def test_dart_has_no_context() -> None:
    """With `context` 0 the model is DART: no encoder, an empty context, psi of the centre alone."""
    model = DARTS(4, SPEC | {"context": 0})
    assert model.encoder is None and not any(k.startswith(("encoder", "film"))
                                             for k in model.state_dict())
    image = torch.randn(2, 4, 5, 6)
    assert model.contexts(image, torch.ones(2, 1, 5, 6), torch.zeros(2)).shape == (2, 0, 5, 6)
