"""
Training utilities: EMA, training loop, validation, checkpointing.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].sub_((1 - self.decay) * (self.shadow[name] - p))

    def apply(self, model: nn.Module):
        """Temporarily apply EMA weights to model."""
        self.backup = {name: p.clone() for name, p in model.named_parameters() if name in self.shadow}
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        del self.backup

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler with warmup
# ──────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """
    Create a learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        cfg: training config dict
        steps_per_epoch: number of steps per epoch
    """
    warmup_steps = cfg.get("warmup_epochs", 5) * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if cfg.get("scheduler", "cosine") == "cosine":
            import math
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: Optional[EMA],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: dict,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
        "config": cfg,
        # Normalization stats are stored as registered buffers in model.state_dict(),
        # but we also save them explicitly for easy standalone access.
        "rna_norm": {
            'mean': model.rna_mean.cpu().numpy() if hasattr(model, 'rna_mean') and model.rna_mean is not None else None,
            'std':  model.rna_std.cpu().numpy()  if hasattr(model, 'rna_std')  and model.rna_std  is not None else None,
        },
    }
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, ema=None):
    """Load training checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if ema and state.get("ema_state_dict"):
        ema.load_state_dict(state["ema_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {state['epoch']}, step {state['global_step']}")
    return state["epoch"], state["global_step"], state.get("best_val_loss", float("inf"))


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(diffusion_model, val_loader, device, max_batches: int = 50) -> dict:
    """
    Run validation and return average metrics.

    Args:
        diffusion_model: the full GaussianDiffusion model
        val_loader: validation DataLoader
        device: torch device
        max_batches: cap number of val batches for speed
    Returns:
        dict of averaged metrics
    """
    diffusion_model.eval()
    metrics_sum = {}
    n = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        img_features = batch["img_features"].to(device)
        rna_embedding = batch["rna_embedding"].to(device)

        result = diffusion_model.compute_loss(rna_embedding, img_features)

        for k, v in result.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v.item()
        n += 1

    diffusion_model.train()
    return {k: v / max(n, 1) for k, v in metrics_sum.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    diffusion_model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    device: torch.device,
    resume_path: Optional[str] = None,
):
    """
    Main training loop with logging, validation, EMA, and checkpointing.

    Args:
        diffusion_model: the full GaussianDiffusion model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        cfg: training configuration dict
        device: torch device
        resume_path: optional path to resume from checkpoint
    """
    train_cfg = cfg["training"]

    # Move model to device
    diffusion_model = diffusion_model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        betas=(0.9, 0.999),
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // train_cfg.get("grad_accum_steps", 1)
    scheduler = get_scheduler(optimizer, train_cfg, steps_per_epoch)

    # EMA
    ema = EMA(diffusion_model, decay=train_cfg.get("ema_decay", 0.9999))

    # Mixed precision
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Training config
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("val_every", 1)
    save_every = train_cfg.get("save_every", 10)

    # ── Logging setup ─────────────────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    use_wandb = log_cfg.get("use_wandb", False) and HAS_WANDB
    use_tb = log_cfg.get("use_tensorboard", True) and HAS_TB

    # Weights & Biases
    if use_wandb:
        wandb.init(
            project=log_cfg.get("wandb_project", "img2rna"),
            entity=log_cfg.get("wandb_entity", None),
            name=log_cfg.get("wandb_run_name", None),
            tags=log_cfg.get("wandb_tags", []),
            config=cfg,
            resume="allow",
        )
        wandb.watch(diffusion_model, log="gradients", log_freq=log_every)
        logger.info(f"wandb run: {wandb.run.url}")
    else:
        if log_cfg.get("use_wandb", False) and not HAS_WANDB:
            logger.warning("wandb requested but not installed — falling back to tensorboard")

    # Tensorboard
    writer = None
    log_dir = os.path.join(train_cfg.get("log_dir", "logs"), time.strftime("%Y%m%d_%H%M%S"))
    if use_tb:
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"Tensorboard logs: {log_dir}")

    # Checkpoint dir
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path, diffusion_model, optimizer, scheduler, ema
        )
        start_epoch += 1

    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"  Batch size: {train_cfg['batch_size']}")
    logger.info(f"  Grad accum: {grad_accum}")
    logger.info(f"  Effective batch: {train_cfg['batch_size'] * grad_accum}")
    logger.info(f"  Steps/epoch: {steps_per_epoch}")
    logger.info(f"  AMP: {use_amp}")

    param_info = diffusion_model.denoiser.count_parameters()
    logger.info(f"  Model params: {param_info['trainable']:,} trainable / {param_info['total']:,} total")

    # ── Diagnostic: verify normalization on first batch ─────────────────
    _diag_batch = next(iter(train_loader))
    _diag_rna = _diag_batch["rna_embedding"]
    _diag_img = _diag_batch["img_features"]
    logger.info(
        f"  [NORM CHECK] First batch RNA: mean={_diag_rna.mean():.4f}, std={_diag_rna.std():.4f}, "
        f"min={_diag_rna.min():.4f}, max={_diag_rna.max():.4f}"
    )
    logger.info(
        f"  [NORM CHECK] First batch IMG: mean={_diag_img.mean():.4f}, std={_diag_img.std():.4f}, "
        f"min={_diag_img.min():.4f}, max={_diag_img.max():.4f}"
    )
    if _diag_rna.std() < 0.1:
        logger.warning(
            "  [NORM CHECK] RNA std is very low — normalization may NOT be applied! "
            "Expected ~1.0 after standardization."
        )
    del _diag_batch, _diag_rna, _diag_img

    for epoch in range(start_epoch, train_cfg["epochs"]):
        diffusion_model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            img_features = batch["img_features"].to(device, non_blocking=True)
            rna_embedding = batch["rna_embedding"].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                result = diffusion_model.compute_loss(rna_embedding, img_features)
                loss = result["loss"] / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                ema.update(diffusion_model)
                global_step += 1

                # Logging
                epoch_loss += result["loss"].item()
                epoch_steps += 1

                pbar.set_postfix({
                    "loss": f"{result['loss'].item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                if global_step % log_every == 0:
                    step_metrics = {
                        "train/loss": result["loss"].item(),
                        "train/mse": result["mse"].item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    }
                    if writer:
                        for k, v in step_metrics.items():
                            writer.add_scalar(k, v, global_step)
                    if use_wandb:
                        wandb.log(step_metrics, step=global_step)

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} | Train loss: {avg_train_loss:.6f}")
        if writer:
            writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        if use_wandb:
            wandb.log({"epoch/train_loss": avg_train_loss, "epoch": epoch}, step=global_step)

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0:
            # Validate with EMA weights
            ema.apply(diffusion_model)
            val_metrics = validate(diffusion_model, val_loader, device)
            ema.restore(diffusion_model)

            val_loss = val_metrics["loss"]
            logger.info(f"Epoch {epoch} | Val loss: {val_loss:.6f} | Val MSE: {val_metrics['mse']:.6f}")

            if writer:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/mse", val_metrics["mse"], epoch)
            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/mse": val_metrics["mse"],
                    "epoch": epoch,
                }, step=global_step)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ema.apply(diffusion_model)
                save_checkpoint(
                    os.path.join(ckpt_dir, "best_model.pt"),
                    diffusion_model, optimizer, scheduler, ema,
                    epoch, global_step, best_val_loss, cfg,
                )
                ema.restore(diffusion_model)
                logger.info(f"  New best val loss: {best_val_loss:.6f}")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt"),
                diffusion_model, optimizer, scheduler, ema,
                epoch, global_step, best_val_loss, cfg,
            )

    # Final save
    save_checkpoint(
        os.path.join(ckpt_dir, "final_model.pt"),
        diffusion_model, optimizer, scheduler, ema,
        train_cfg["epochs"] - 1, global_step, best_val_loss, cfg,
    )

    if writer:
        writer.close()
    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
