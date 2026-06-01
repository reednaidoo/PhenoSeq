"""
PyTorch Lightning LightningModule for Img2RNA diffusion training.

Wraps the GaussianDiffusion model with:
  - Automatic optimizer / scheduler setup
  - Training / validation step definitions
  - EMA weight averaging
  - Periodic sample generation for qualitative monitoring
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.denoiser import Img2RNADenoiser
from models.diffusion import GaussianDiffusion


class EMA:
    """Exponential Moving Average of model parameters (Lightning-compatible)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(p.data, 1 - self.decay)

    def apply(self, model: nn.Module):
        self._backup = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class Img2RNALitModule(pl.LightningModule):
    """
    Lightning module for image-conditioned RNA-seq diffusion.

    Handles training, validation, EMA, and optional sample generation.
    Works with any Lightning logger (TensorBoard, wandb, CSV, etc.).
    """

    def __init__(
        self,
        # Model config
        img_dim: int = 5120,
        rna_dim: int = 512,
        model_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        time_dim: int = 256,
        ff_mult: int = 4,
        dropout: float = 0.1,
        # Diffusion config
        num_steps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        # Normalization stats
        rna_norm: Optional[dict] = None,
        # Training config
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        scheduler_type: str = "cosine",
        max_grad_norm: float = 1.0,
        ema_decay: float = 0.9999,
        # Sampling config (for validation visualisation)
        val_sample_every_n_epochs: int = 10,
        ddim_steps: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build denoiser
        self.denoiser = Img2RNADenoiser(
            img_dim=img_dim,
            rna_dim=rna_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            time_dim=time_dim,
            ff_mult=ff_mult,
            dropout=dropout,
        )

        # Build diffusion wrapper
        self.diffusion = GaussianDiffusion(
            denoiser=self.denoiser,
            num_steps=num_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            loss_type=loss_type,
            rna_norm=rna_norm,
        )

        # EMA (initialised in on_fit_start so it lives on correct device)
        self._ema: Optional[EMA] = None
        self._ema_decay = ema_decay

        # Cache for logging
        self._val_step_outputs: list[dict] = []

    # ── Lifecycle hooks ───────────────────────────────────────────────────

    def on_fit_start(self):
        self._ema = EMA(self.diffusion, decay=self._ema_decay)
        param_info = self.denoiser.count_parameters()
        self.log_dict({
            "model/total_params": float(param_info["total"]),
            "model/trainable_params": float(param_info["trainable"]),
        })

    # ── Training ──────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        img_features = batch["img_features"]   # (B, N, img_dim)
        rna_embedding = batch["rna_embedding"]  # (B, rna_dim)

        result = self.diffusion.compute_loss(rna_embedding, img_features)
        loss = result["loss"]

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", result["mse"], on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._ema is not None:
            self._ema.update(self.diffusion)

    # ── Validation ────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        img_features = batch["img_features"]
        rna_embedding = batch["rna_embedding"]

        result = self.diffusion.compute_loss(rna_embedding, img_features)

        self.log("val/loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse", result["mse"], on_step=False, on_epoch=True, sync_dist=True)

        self._val_step_outputs.append({
            "loss": result["loss"].detach(),
            "mse": result["mse"].detach(),
        })

        return result

    def on_validation_epoch_end(self):
        # Optional: generate samples for qualitative monitoring
        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.val_sample_every_n_epochs == 0
        ):
            self._log_generated_samples()

        self._val_step_outputs.clear()

    @torch.no_grad()
    def _log_generated_samples(self, n_samples: int = 8):
        """Generate a few samples and log statistics to the logger."""
        # Use EMA weights for sampling
        if self._ema is not None:
            self._ema.apply(self.diffusion)

        try:
            # Create dummy imaging conditioning (random from normal; in practice
            # you'd use real validation imaging features)
            dummy_img = torch.randn(
                n_samples,
                self.hparams.get("n_imaging_cells", 16),
                self.hparams.img_dim,
                device=self.device,
            )
            generated = self.diffusion.sample_ddim(
                dummy_img,
                num_inference_steps=self.hparams.ddim_steps,
            )

            self.log("val/generated_mean", generated.mean())
            self.log("val/generated_std", generated.std())
            self.log("val/generated_min", generated.min())
            self.log("val/generated_max", generated.max())

        finally:
            if self._ema is not None:
                self._ema.restore(self.diffusion)

    # ── Optimizer & scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        if self.hparams.scheduler_type == "cosine":
            # Total steps will be set by trainer
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.hparams.lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        if self.hparams.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.hparams.max_grad_norm)

    # ── Inference helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        img_features: torch.Tensor,
        use_ema: bool = True,
        ddim: bool = True,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate RNA-seq embeddings from imaging features.

        Args:
            img_features: (B, N, img_dim)
            use_ema: whether to use EMA weights
            ddim: use DDIM sampling (faster)
            ddim_steps: number of DDIM steps
        Returns:
            (B, rna_dim) generated embeddings
        """
        if use_ema and self._ema is not None:
            self._ema.apply(self.diffusion)

        try:
            if ddim:
                return self.diffusion.sample_ddim(img_features, num_inference_steps=ddim_steps)
            else:
                return self.diffusion.sample(img_features)
        finally:
            if use_ema and self._ema is not None:
                self._ema.restore(self.diffusion)
