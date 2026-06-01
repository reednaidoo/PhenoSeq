"""
Img2RNA: Train a conditional diffusion model to generate RNA-seq embeddings
from microscopy imaging features.

Usage:
    python train.py --config config.yaml [--key value ...]

CLI overrides follow dot notation:
    python train.py --config config.yaml \
        --training.batch_size 128 \
        --training.lr 5e-5 \
        --model.num_layers 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import random

import numpy as np
import torch
import yaml

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("img2rna")


# ──────────────────────────────────────────────────────────────────────────────
# Config utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply CLI overrides in dot notation.
    E.g., --training.lr 1e-4 → cfg["training"]["lr"] = 1e-4
    """
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        val = overrides[i + 1] if i + 1 < len(overrides) else None
        i += 2

        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})

        # Try to infer type
        if val is not None:
            # Try int
            try:
                val = int(val)
            except ValueError:
                # Try float
                try:
                    val = float(val)
                except ValueError:
                    # Try bool
                    if val.lower() in ("true", "false"):
                        val = val.lower() == "true"

        d[parts[-1]] = val

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Img2RNA diffusion model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args, unknown = parser.parse_known_args()

    # Load config
    cfg = load_config(args.config)
    if unknown:
        cfg = apply_overrides(cfg, unknown)

    # Seed
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        # logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    from data.dataset import create_dataloaders

    data_cfg = cfg["data"]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve(p: str) -> str:
        return os.path.normpath(os.path.join(script_dir, p))

    img_cache_dir = _resolve(data_cfg.get("img_cache_dir", "data/cache/imaging"))
    rna_cache_dir = _resolve(data_cfg.get("rna_cache_dir", "data/cache/rnaseq"))

    if not os.path.isdir(img_cache_dir) or not os.path.isdir(rna_cache_dir):
        logger.error(
            "Cache directories not found:\n"
            f"  {img_cache_dir}\n"
            f"  {rna_cache_dir}\n"
            "Run `python prepare_data.py` first to pre-extract per-sample .npz files."
        )
        raise SystemExit(1)

    logger.info("Building dataloaders...")
    train_loader, val_loader, data_info = create_dataloaders(
        img_cache_dir=img_cache_dir,
        rna_cache_dir=rna_cache_dir,
        n_imaging_cells=data_cfg.get("n_imaging_cells", 16),
        train_ratio=data_cfg.get("train_ratio", 0.85),
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        seed=seed,
        img_cache_size=data_cfg.get("img_cache_size", 32),
        preload_imaging=data_cfg.get("preload_imaging", True),
    )

    logger.info(f"Data info: {data_info}")

    # Normalization stats (computed from training set by create_dataloaders)
    rna_norm = data_info["rna_norm"]  # {'mean': ndarray, 'std': ndarray}
    img_norm = data_info["img_norm"]  # {'mean': ndarray, 'std': ndarray}

    # ── Model ─────────────────────────────────────────────────────────────
    from models.denoiser import Img2RNADenoiser
    from models.diffusion import GaussianDiffusion

    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]

    denoiser = Img2RNADenoiser(
        img_dim=data_info["img_dim"],
        rna_dim=data_info["rna_dim"],
        model_dim=model_cfg.get("model_dim", 1024),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        time_dim=model_cfg.get("time_dim", 256),
        ff_mult=model_cfg.get("ff_mult", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_steps=diff_cfg.get("num_steps", 1000),
        schedule=diff_cfg.get("schedule", "cosine"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        loss_type="l2",
        rna_norm=rna_norm,
    )

    param_info = denoiser.count_parameters()
    logger.info(f"Denoiser parameters: {param_info['trainable']:,} trainable / {param_info['total']:,} total")

    # ── Train ─────────────────────────────────────────────────────────────
    from utils.train_utils import train

    train(
        diffusion_model=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
