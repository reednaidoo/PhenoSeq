"""
Gaussian Diffusion for Image-conditioned RNA-seq Generation.

Implements the forward (noise) and reverse (denoise) diffusion processes
with support for linear and cosine beta schedules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Beta schedules
# ──────────────────────────────────────────────────────────────────────────────

def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
    (Nichol & Dhariwal, 2021).
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / num_steps) + s) / (1 + s) * (np.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


def get_beta_schedule(schedule: str, num_steps: int, **kwargs) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(num_steps, **kwargs)
    elif schedule == "cosine":
        return cosine_beta_schedule(num_steps)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian Diffusion
# ──────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for generating RNA-seq embeddings
    conditioned on imaging features.

    The model learns to predict the noise ε given:
      - noisy RNA embedding x_t
      - conditioning imaging features
      - timestep t

    Training loss:
        L = E[||ε - ε_θ(x_t, img, t)||²]

    Args:
        denoiser: noise prediction network (Img2RNADenoiser)
        num_steps: number of diffusion steps T
        schedule: beta schedule type ("linear" or "cosine")
        beta_start: start of linear schedule
        beta_end: end of linear schedule
        loss_type: "l2", "l1", or "huber"
    """

    def __init__(
        self,
        denoiser: nn.Module,
        num_steps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        rna_norm: Optional[dict] = None,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.num_steps = num_steps
        self.loss_type = loss_type

        # Compute noise schedule
        betas = get_beta_schedule(
            schedule, num_steps,
            beta_start=beta_start, beta_end=beta_end,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (moved to device automatically)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # Pre-compute useful quantities
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas).float())

        # Posterior q(x_{t-1} | x_t, x_0) parameters
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(torch.clamp(posterior_variance, min=1e-20)).float())
        self.register_buffer("posterior_mean_coef1",
                             (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())
        self.register_buffer("posterior_mean_coef2",
                             ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float())

        # RNA-seq normalization stats for un-normalizing generated samples
        if rna_norm is not None:
            self.register_buffer("rna_mean", torch.from_numpy(rna_norm['mean']).float())
            self.register_buffer("rna_std", torch.from_numpy(rna_norm['std']).float())
        else:
            self.rna_mean = None
            self.rna_std = None

    def unnormalize_rna(self, x: torch.Tensor) -> torch.Tensor:
        """Convert normalised diffusion output back to original RNA embedding scale."""
        if self.rna_mean is not None and self.rna_std is not None:
            return x * self.rna_std + self.rna_mean
        return x

    # ── Forward diffusion (add noise) ─────────────────────────────────────

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I).

        Args:
            x_start: (B, D) clean RNA embeddings
            t: (B,) timesteps
            noise: optional pre-sampled noise
        Returns:
            x_t: (B, D) noisy embeddings
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]       # (B,)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]  # (B,)

        # Reshape for broadcasting over feature dim
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    # ── Training loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        rna_embedding: torch.Tensor,   # (B, rna_dim) clean target
        img_features: torch.Tensor,    # (B, N, img_dim) conditioning
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the diffusion training loss.

        Randomly samples timesteps, adds noise, predicts noise, returns loss.

        Returns:
            dict with "loss" and other metrics
        """
        B = rna_embedding.shape[0]
        device = rna_embedding.device

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=device).long()

        # Sample noise
        if noise is None:
            noise = torch.randn_like(rna_embedding)

        # Add noise
        x_t = self.q_sample(rna_embedding, t, noise)

        # Predict noise
        noise_pred = self.denoiser(
            noisy_rna=x_t,
            img_features=img_features,
            timestep=t,
        )

        # Compute loss
        if self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return {
            "loss": loss,
            "mse": F.mse_loss(noise_pred, noise).detach(),
        }

    # ── Reverse diffusion (sampling) ──────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        img_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: sample x_{t-1} from p_θ(x_{t-1} | x_t).

        Args:
            x_t: (B, D) current noisy state
            t: scalar timestep
            img_features: (B, N, img_dim) conditioning
        Returns:
            x_{t-1}: (B, D) less noisy state
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        noise_pred = self.denoiser(
            noisy_rna=x_t,
            img_features=img_features,
            timestep=t_batch,
        )

        # Compute posterior mean
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        beta = self.betas[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip_alpha * (x_t - beta * noise_pred / sqrt_one_minus)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(
        self,
        img_features: torch.Tensor,   # (B, N, img_dim)
        shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Generate RNA-seq embeddings conditioned on imaging features.

        Args:
            img_features: (B, N, img_dim) conditioning features
            shape: output shape (B, rna_dim); inferred if None
        Returns:
            x_0: (B, rna_dim) generated RNA-seq embeddings
        """
        B = img_features.shape[0]
        device = img_features.device

        if shape is None:
            shape = (B, self.denoiser.rna_dim)

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Iterative denoising
        for t in reversed(range(self.num_steps)):
            x = self.p_sample(x, t, img_features)

        return self.unnormalize_rna(x)

    @torch.no_grad()
    def sample_ddim(
        self,
        img_features: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling for faster inference.

        Args:
            img_features: (B, N, img_dim) 
            num_inference_steps: number of denoising steps (< num_steps for speed)
            eta: controls stochasticity (0 = deterministic DDIM, 1 = DDPM)
            shape: output shape
        Returns:
            x_0: (B, rna_dim) generated embeddings
        """
        B = img_features.shape[0]
        device = img_features.device

        if shape is None:
            shape = (B, self.denoiser.rna_dim)

        # Create sub-sequence of timesteps
        step_size = self.num_steps // num_inference_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            noise_pred = self.denoiser(
                noisy_rna=x,
                img_features=img_features,
                timestep=t_batch,
            )

            # Predict x_0
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t]

            x_0_pred = (x - sqrt_one_minus_t * noise_pred) / sqrt_alpha_t

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * noise_pred
            x = torch.sqrt(alpha_t_prev) * x_0_pred + pred_dir

            if sigma > 0 and i < len(timesteps) - 1:
                x = x + sigma * torch.randn_like(x)

        return self.unnormalize_rna(x)
