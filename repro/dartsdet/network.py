"""The noise-conditional score head: a residual MLP of a pixel's noised spectrum and its context.

    e       = MLP(Fourier(log sigma))                              one embedding per noise level
    h_0     = W_in [z, c]                                           z: noised centre, c: context
    h_{k+1} = h_k + W_k act(LN(h_k) (1 + gamma_k(e)) + beta_k(e))  k = 0 .. depth - 1
    out     = W_out LN(h_depth)

The noise level conditions every block (FiLM), not only the input. The Fourier frequencies are
fixed draws, N(0, fourier_scale^2) cycles per unit of log sigma. W_out starts at zero, so an
untrained head outputs 0 at every noise level.
"""

import math

import torch


class FourierFeatures(torch.nn.Module):
    """[sin 2 pi w u, cos 2 pi w u] of a scalar u per row, (n,) -> (n, n_features)."""

    def __init__(self, n_features: int, scale: float) -> None:
        super().__init__()
        # Drawn in float64 and then cast: the draws, and every initialization after them, match
        # the trained checkpoints'.
        frequencies = torch.randn(n_features // 2, dtype=torch.float64) * scale
        self.register_buffer("frequencies", frequencies.float())

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        angle = 2 * math.pi * u[:, None] * self.frequencies
        return torch.cat([angle.sin(), angle.cos()], dim=1)


class Block(torch.nn.Module):
    """h + W act(LN(h) (1 + gamma(e)) + beta(e)), (n, width) -> (n, width)."""

    def __init__(self, width: int, embed_dim: int, activation: str) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width, elementwise_affine=False)
        self.film = torch.nn.Linear(embed_dim, 2 * width)
        self.linear = torch.nn.Linear(width, width)
        self.activation = getattr(torch.nn, activation)()

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(e).chunk(2, dim=1)
        return h + self.linear(self.activation(self.norm(h) * (1 + gamma) + beta))


class ScoreMLP(torch.nn.Module):
    """([z, c] (n, B + C), log sigma (n,)) -> (n, B), dimensionless."""

    def __init__(self, n_bands: int, n_context: int, width: int, depth: int, spec: dict) -> None:
        """
        Args:
            n_bands: B, the output dimension and the first B input features.
            n_context: C, the context features that follow them.
            width, depth: hidden units per block and number of residual blocks.
            spec: `activation` (a torch.nn class name), `embed_dim` (even) and `fourier_scale`.
        """
        super().__init__()
        activation, embed_dim = spec["activation"], spec["embed_dim"]
        self.embed = torch.nn.Sequential(
            FourierFeatures(embed_dim, spec["fourier_scale"]),
            torch.nn.Linear(embed_dim, embed_dim), getattr(torch.nn, activation)(),
            torch.nn.Linear(embed_dim, embed_dim), getattr(torch.nn, activation)())
        self.input = torch.nn.Linear(n_bands + n_context, width)
        self.blocks = torch.nn.ModuleList(Block(width, embed_dim, activation) for _ in range(depth))
        self.norm = torch.nn.LayerNorm(width)
        self.output = torch.nn.Linear(width, n_bands)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        e = self.embed(log_sigma)
        h = self.input(z)
        for block in self.blocks:
            h = block(h, e)
        return self.output(self.norm(h))
