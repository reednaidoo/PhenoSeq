# PhenoSeq: Conditional Diffusion Model for Generating RNA-seq from Microscopy Imaging

A diffusion-based model that generates single-cell RNA-seq embeddings (scGPT) conditioned on microscopy imaging features (ViT-L).

## Overview

This project trains a Gaussian diffusion model that:
1. Takes **microscopy imaging features** (5,120-dim ViT-L embeddings from 5 fluorescence channels) as conditioning input
2. Generates **scRNA-seq embeddings** (512-dim scGPT cell embeddings) via iterative denoising

The model uses cross-attention to condition the denoising process on imaging features, enabling image-to-transcriptome translation at single-cell resolution.

## Data

The data comes from the [scGeneScope](https://huggingface.co/datasets/altoslabs/scGeneScope) dataset:

- **Imaging features**: `scGeneScope/data/features/imaging/imagenet/vit-l/` — `.h5ad` files with ViT-L embeddings (5 channels × 1,024 dims = 5,120 total)
- **RNA-seq features**: `scGeneScope/data/features/rnaseq/scgpt/` — `.h5ad` files with scGPT embeddings (512 dims)
- Samples are matched by `Sample_ID` at the well/sample level

## Project Structure

```
img2rna/
├── README.md
├── requirements.txt
├── config.yaml           # Training configuration
├── train.py              # Main training entry point
├── models/
│   ├── __init__.py
│   ├── diffusion.py      # Gaussian diffusion process
│   ├── denoiser.py       # Cross-attention denoiser network
│   └── model_utils.py    # Shared model utilities
├── data/
│   ├── __init__.py
│   └── dataset.py        # Paired imaging-RNA dataset & dataloader
└── utils/
    ├── __init__.py
    └── train_utils.py    # Training loop, logging, checkpointing
```

## Usage

### Training

```bash
cd img2rna
python train.py --config config.yaml
```

### Key Arguments (override via CLI)

```bash
python train.py --config config.yaml \
    --batch_size 256 \
    --lr 1e-4 \
    --epochs 200 \
    --diffusion_steps 1000 \
    --model_dim 1024 \
    --num_heads 8
```

## Architecture

The denoiser network uses a **cross-attention transformer** architecture:

1. **Imaging encoder**: Projects 5,120-dim imaging features → model dimension
2. **Noisy RNA encoder**: Projects noisy 512-dim scGPT + sinusoidal time embedding → model dimension  
3. **Cross-attention blocks**: RNA queries attend to imaging keys/values (multiple layers)
4. **RNA decoder**: Projects back to 512-dim scGPT space

The diffusion process adds Gaussian noise to scGPT embeddings over T steps, and the model learns to reverse this process conditioned on the corresponding imaging features.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- anndata
- h5py
- PyYAML
- tensorboard
- tqdm
