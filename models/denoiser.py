"""
Cross-Attention Denoiser for Image-conditioned RNA-seq Diffusion.

The denoiser takes:
  - noisy RNA-seq embeddings (scGPT, 512-dim)
  - imaging features (ViT-L, 5120-dim) as conditioning context
  - diffusion timestep

And predicts the noise ε added to the RNA-seq embeddings.

Architecture:
  1. Project imaging features → model_dim, apply self-attention to create a
     rich context representation.
  2. Embed noisy RNA + sinusoidal time embedding → model_dim.
  3. Multiple cross-attention transformer blocks where RNA queries attend to
     imaging context.
  4. Project back to RNA embedding space and predict noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import (
    SinusoidalTimeEmbedding,
    Mish,
    FeedForward,
    AdaLayerNorm,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Block
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention layer: RNA query attends to imaging context.
    Uses adaptive layer norm for time-conditioning and pre-norm residual style.
    """

    def __init__(self, dim: int, num_heads: int, time_dim: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()

        # Time-conditioned norms
        self.norm_rna = AdaLayerNorm(dim, time_dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.norm_ff = AdaLayerNorm(dim, time_dim)

        # Cross-attention: Q from RNA, K/V from imaging
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention on RNA after cross-attention  
        self.self_attn_norm = AdaLayerNorm(dim, time_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        rna: torch.Tensor,       # (B, 1, D)
        context: torch.Tensor,    # (B, S, D)  imaging context
        time_emb: torch.Tensor,   # (B, time_dim)
    ) -> torch.Tensor:
        # Cross-attention
        rna_normed = self.norm_rna(rna, time_emb)
        ctx_normed = self.norm_ctx(context)
        rna = rna + self.cross_attn(rna_normed, ctx_normed, ctx_normed, need_weights=False)[0]

        # Self-attention (useful when we have multiple RNA tokens, but also adds
        # a residual self-refinement step)
        rna_normed = self.self_attn_norm(rna, time_emb)
        rna = rna + self.self_attn(rna_normed, rna_normed, rna_normed, need_weights=False)[0]

        # Feedforward
        rna = rna + self.ff(self.norm_ff(rna, time_emb))

        return rna


# ──────────────────────────────────────────────────────────────────────────────
# Imaging Context Encoder
# ──────────────────────────────────────────────────────────────────────────────

class ImagingEncoder(nn.Module):
    """
    Encodes a set of imaging cell features into context representations.
    Uses a small self-attention stack to let imaging cells attend to each other
    before being used as cross-attention context.
    """

    def __init__(self, img_dim: int, model_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(img_dim, model_dim),
            Mish(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features: (B, N, img_dim)
        Returns:
            (B, N, model_dim) context representations
        """
        x = self.proj(img_features)
        return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────────────
# Full Denoiser
# ──────────────────────────────────────────────────────────────────────────────

class Img2RNADenoiser(nn.Module):
    """
    Noise prediction network for image-conditioned RNA-seq diffusion.

    Args:
        img_dim:    input imaging feature dimension (5120)
        rna_dim:    RNA-seq embedding dimension (512)
        model_dim:  internal model dimension (1024)
        num_heads:  number of attention heads (8)
        num_layers: number of cross-attention blocks (6)
        time_dim:   time embedding dimension (256)
        ff_mult:    feedforward multiplier (4)
        dropout:    dropout rate (0.1)
    """

    def __init__(
        self,
        img_dim: int = 5120,
        rna_dim: int = 512,
        model_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        time_dim: int = 256,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.rna_dim = rna_dim

        # ── Time embedding ────────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # ── Imaging encoder ───────────────────────────────────────────────
        self.img_encoder = ImagingEncoder(
            img_dim=img_dim,
            model_dim=model_dim,
            num_heads=min(num_heads, 4),
            num_layers=2,
            dropout=dropout,
        )

        # ── RNA input projection ──────────────────────────────────────────
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, model_dim),
            Mish(),
            nn.LayerNorm(model_dim),
        )

        # ── Cross-attention transformer stack ─────────────────────────────
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=model_dim,
                num_heads=num_heads,
                time_dim=time_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Output projection: predict noise in RNA embedding space ──────
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            Mish(),
            nn.Linear(model_dim, rna_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stable diffusion training.

        Uses PyTorch defaults (kaiming) for all layers, with a small-scale
        init on the very last linear so the model starts by predicting
        near-zero noise while still allowing gradient flow.
        """
        # Small (not zero) init for the final output linear
        nn.init.normal_(self.out_proj[2].weight, std=1e-4)
        nn.init.zeros_(self.out_proj[2].bias)

    def forward(
        self,
        noisy_rna: torch.Tensor,    # (B, rna_dim) — noisy scGPT embedding
        img_features: torch.Tensor,  # (B, N, img_dim) — imaging features
        timestep: torch.Tensor,      # (B,) — diffusion timestep
    ) -> torch.Tensor:
        """
        Predict the noise component in the noisy RNA embedding.

        Returns:
            predicted_noise: (B, rna_dim)
        """
        # Time embedding
        t_emb = self.time_embed(timestep)  # (B, time_dim)

        # Encode imaging context
        context = self.img_encoder(img_features)  # (B, N, model_dim)

        # Project noisy RNA and add as a single token
        rna = self.rna_proj(noisy_rna).unsqueeze(1)  # (B, 1, model_dim)

        # Cross-attention layers
        for layer in self.layers:
            rna = layer(rna, context, t_emb)

        # Output projection
        rna = self.out_norm(rna.squeeze(1))  # (B, model_dim)
        noise_pred = self.out_proj(rna)       # (B, rna_dim)

        return noise_pred

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
