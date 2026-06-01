"""
Shared model utilities: sinusoidal embeddings, activation functions, blocks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Activations
# ──────────────────────────────────────────────────────────────────────────────

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GEGLU(nn.Module):
    """Gated GELU activation for feedforward blocks."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional / time embedding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar timesteps to sinusoidal embeddings.

    Args:
        dim: output embedding dimension (must be even)
        max_period: controls frequency range
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timesteps
        Returns:
            (B, dim) sinusoidal embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Transformer feedforward block with GEGLU gating.

    Args:
        dim: input/output dimension
        mult: hidden dimension multiplier  
        dropout: dropout rate
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim * mult * 2  # ×2 for GEGLU split
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on time embedding.
    Applies scale/shift modulation: γ(t) * LayerNorm(x) + β(t)

    Args:
        dim: feature dimension
        cond_dim: conditioning (time) embedding dimension
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            Mish(),
            nn.Linear(cond_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)  or (B, D)
            cond: (B, cond_dim)
        """
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        if x.dim() == 3 and gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta
