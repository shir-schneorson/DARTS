"""The context encoder is blind to each pixel's own square and to masked pixels."""

import pytest
import torch

from dartsdet.encoder import BlindContext

SPEC = {"channels": 8, "dilations": [1, 2, 1], "shift": 2, "context": 4, "activation": "GELU"}


def encoder_and_image(height: int, width: int) -> tuple:
    torch.manual_seed(0)
    encoder = BlindContext(3, SPEC).double().eval()
    z = torch.randn(1, 3, height, width, dtype=torch.float64)
    return encoder, z, torch.ones(1, 1, height, width, dtype=torch.float64)


@pytest.mark.parametrize("shape", [(20, 20), (18, 23)])  # square: batched rotations; else looped
def test_context_is_blind_to_the_pixel_and_its_square(shape: tuple[int, int]) -> None:
    """The context at (i, j) has zero gradient on the (2 shift - 1)^2 square around it and a
    nonzero one on every pixel of the ring just outside."""
    encoder, z, mask = encoder_and_image(*shape)
    i, j, reach = 9, 10, SPEC["shift"] - 1
    z.requires_grad_(True)
    (grad,) = torch.autograd.grad(encoder(z, mask)[..., i, j].sum(), z)
    support = grad.abs().sum(dim=(0, 1)) > 0
    assert not support[i - reach:i + reach + 1, j - reach:j + reach + 1].any()
    ring = support[i - reach - 1:i + reach + 2, j - reach - 1:j + reach + 2].clone()
    ring[1:-1, 1:-1] = True
    assert ring.all()


def test_context_ignores_masked_pixels() -> None:
    encoder, z, mask = encoder_and_image(16, 16)
    mask[..., 3, 4] = 0
    moved = z.clone()
    moved[..., 3, 4] += 10.0
    assert torch.allclose(encoder(moved, mask), encoder(z, mask))

