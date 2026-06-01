"""
Paired Imaging → RNA-seq Dataset

After running `prepare_data.py` once, per-sample arrays are stored as
compressed .npz files under data/cache/imaging/ and data/cache/rnaseq/.

Two loading modes (controlled by `preload_imaging` in config):

  preload_imaging=True  (default, recommended for high-RAM machines):
    All imaging .npz files are decompressed and loaded into RAM at startup
    (~1-2 min, ~15 GB).  __getitem__ is then pure numpy indexing — no I/O,
    no decompression, GPU runs at full utilisation.

  preload_imaging=False:
    Imaging loaded lazily per sample via an LRU cache (lower RAM, slower).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Discover available sample IDs from cache directory
# ──────────────────────────────────────────────────────────────────────────────

def _load_npz_dir(cache_dir: Path, sample_ids: list[str], label: str) -> dict[str, np.ndarray]:
    """Load all .npz files for the given sample IDs into RAM."""
    logger.info(f"Loading {label} into RAM from {cache_dir} ...")
    data: dict[str, np.ndarray] = {}
    for i, sid in enumerate(sample_ids):
        data[sid] = np.load(cache_dir / f"{sid}.npz")["X"]
        if (i + 1) % 20 == 0 or (i + 1) == len(sample_ids):
            mem_gb = sum(v.nbytes for v in data.values()) / 1e9
            logger.info(f"  {i+1}/{len(sample_ids)} samples loaded  ({mem_gb:.1f} GB)")
    total = sum(v.shape[0] for v in data.values())
    mem_gb = sum(v.nbytes for v in data.values()) / 1e9
    logger.info(f"  {label}: {total:,} cells, {mem_gb:.2f} GB in RAM")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imaging cache (only used when preload_imaging=False)
# ──────────────────────────────────────────────────────────────────────────────

class _ImgLRUCache:
    """Per-worker LRU cache backed by .npz files (lazy loading mode)."""

    def __init__(self, img_cache_dir: Path, cache_size: int = 32) -> None:
        self.img_cache_dir = img_cache_dir
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, sid: str) -> np.ndarray:
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        data = np.load(self.img_cache_dir / f"{sid}.npz")["X"]
        self._cache[sid] = data
        self._cache.move_to_end(sid)
        while len(self._cache) > self.cache_size:
            _, evicted = self._cache.popitem(last=False)
            del evicted
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class Img2RNADataset(Dataset):
    """
    Paired imaging → RNA-seq dataset.

    preload_imaging=True  (default):
      All imaging arrays are pre-loaded into RAM at init.  __getitem__ is
      pure numpy indexing — no disk I/O, GPU runs at full utilisation.

    preload_imaging=False:
      Imaging loaded lazily via an LRU cache (lower RAM, more I/O).

    Normalization:
      If rna_norm / img_norm dicts are provided (keys: 'mean', 'std'),
      embeddings are standardised to zero-mean, unit-variance in __getitem__.
      This is critical for correct diffusion SNR.
    """

    def __init__(
        self,
        rna_data: dict[str, np.ndarray],
        sample_ids: list[str],
        img_cache_dir: Path,
        img_cache_size: int = 32,
        n_imaging_cells: int = 16,
        mode: str = "train",
        seed: int = 42,
        preload_imaging: bool = True,
        # Pre-loaded imaging dict (passed in when preload_imaging=True)
        img_data: Optional[dict[str, np.ndarray]] = None,
        # Normalization stats (computed from training set)
        rna_norm: Optional[dict[str, np.ndarray]] = None,
        img_norm: Optional[dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self.rna_data = rna_data
        self.sample_ids = sample_ids
        self.img_cache_dir = Path(img_cache_dir)
        self.img_cache_size = img_cache_size
        self.n_imaging_cells = n_imaging_cells
        self.mode = mode
        self.preload_imaging = preload_imaging

        # Imaging storage
        self.img_data: Optional[dict[str, np.ndarray]] = img_data
        self._img_lru: Optional[_ImgLRUCache] = None

        # Normalization stats (numpy arrays, broadcastable)
        self.rna_norm = rna_norm   # {'mean': (D,), 'std': (D,)}
        self.img_norm = img_norm   # {'mean': (D,), 'std': (D,)}

        # Flat item list: (sample_id, rna_cell_idx)
        self._items: list[tuple[str, int]] = []
        for sid in sample_ids:
            for i in range(rna_data[sid].shape[0]):
                self._items.append((sid, i))

        # Single per-dataset RNG (each forked worker gets its own copy — safe)
        self._rng = np.random.default_rng(seed)

        mode_str = "preloaded" if preload_imaging else f"lazy (cache={img_cache_size})"
        logger.info(
            f"Img2RNADataset [{mode}]: {len(self._items):,} RNA cells, "
            f"{len(sample_ids)} samples, n_img_cells={n_imaging_cells}, "
            f"imaging={mode_str}"
        )

    def _get_img(self, sid: str) -> np.ndarray:
        if self.preload_imaging:
            return self.img_data[sid]
        # Lazy mode: per-process LRU cache, initialised on first access
        if self._img_lru is None:
            self._img_lru = _ImgLRUCache(self.img_cache_dir, self.img_cache_size)
        return self._img_lru.get(sid)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        sid, rna_idx = self._items[idx]

        rna_emb = self.rna_data[sid][rna_idx]

        img_all = self._get_img(sid)
        n_avail = img_all.shape[0]
        n = self.n_imaging_cells

        if self.mode == "train":
            indices = self._rng.integers(0, n_avail, size=n)
        else:
            indices = np.linspace(0, n_avail - 1, n, dtype=int)

        # Normalise to zero-mean, unit-variance if stats are provided
        rna_out = rna_emb.copy()
        if self.rna_norm is not None:
            rna_out = (rna_out - self.rna_norm['mean']) / self.rna_norm['std']

        img_out = img_all[indices].copy()
        if self.img_norm is not None:
            img_out = (img_out - self.img_norm['mean']) / self.img_norm['std']

        return {
            "img_features": torch.from_numpy(img_out),
            "rna_embedding": torch.from_numpy(rna_out),
        }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    img_cache_dir: str,
    rna_cache_dir: str,
    n_imaging_cells: int = 16,
    train_ratio: float = 0.85,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    img_cache_size: int = 32,
    preload_imaging: bool = True,
    # Legacy / unused
    img_paths: list[str] | None = None,
    rna_paths: list[str] | None = None,
    sample_id_col: str = "Sample_ID",
    rna_obsm_key: str = "scGPT",
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Build train and validation DataLoaders from pre-extracted .npz cache.

    preload_imaging=True (default):
      Loads all imaging data into RAM at startup (~1-2 min, ~15 GB).
      Training __getitem__ is then pure numpy — GPU runs at full utilisation.
    """
    img_cache_dir = Path(img_cache_dir)
    rna_cache_dir = Path(rna_cache_dir)

    # ── 1. Find overlapping samples ───────────────────────────────────────
    img_ids = {p.stem for p in img_cache_dir.glob("*.npz")}
    rna_ids = {p.stem for p in rna_cache_dir.glob("*.npz")}
    sample_ids = sorted(img_ids & rna_ids)
    logger.info(
        f"Cache: {len(img_ids)} imaging, {len(rna_ids)} RNA-seq, "
        f"{len(sample_ids)} overlapping"
    )
    if not sample_ids:
        raise FileNotFoundError(
            f"No overlapping .npz files in:\n  {img_cache_dir}\n  {rna_cache_dir}\n"
            "Run `python prepare_data.py` first."
        )

    # ── 2. Train/val split ────────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sample_ids))
    n_train   = max(1, int(len(sample_ids) * train_ratio))
    train_ids = sorted([sample_ids[i] for i in perm[:n_train]])
    val_ids   = sorted([sample_ids[i] for i in perm[n_train:]])
    logger.info(f"Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    # ── 3. Infer feature dims ────────────────────────────────────────────
    _s = train_ids[0]
    img_dim = np.load(img_cache_dir / f"{_s}.npz")["X"].shape[1]
    rna_dim = np.load(rna_cache_dir / f"{_s}.npz")["X"].shape[1]

    # ── 4. Load RNA into RAM (always, ~1-2 GB) ───────────────────────────
    all_ids  = sorted(set(train_ids) | set(val_ids))
    rna_data = _load_npz_dir(rna_cache_dir, all_ids, "RNA-seq")

    # ── 5. Optionally preload imaging into RAM ───────────────────────────
    img_data: Optional[dict[str, np.ndarray]] = None
    if preload_imaging:
        img_data = _load_npz_dir(img_cache_dir, all_ids, "imaging")

    # ── 5b. Compute normalization stats from TRAINING set only ───────────
    logger.info("Computing normalization statistics from training set...")
    # RNA-seq: per-dimension mean/std across all training cells
    rna_train_all = np.concatenate([rna_data[sid] for sid in train_ids], axis=0)
    rna_norm = {
        'mean': rna_train_all.mean(axis=0).astype(np.float32),
        'std':  rna_train_all.std(axis=0).astype(np.float32).clip(min=1e-6),
    }
    logger.info(
        f"  RNA-seq norm — mean: [{rna_norm['mean'].min():.4f}, {rna_norm['mean'].max():.4f}], "
        f"std: [{rna_norm['std'].min():.4f}, {rna_norm['std'].max():.4f}]"
    )
    del rna_train_all

    # Imaging: per-dimension mean/std across all training cells
    if preload_imaging:
        img_train_all = np.concatenate([img_data[sid] for sid in train_ids], axis=0)
    else:
        img_train_all = np.concatenate(
            [np.load(img_cache_dir / f"{sid}.npz")["X"] for sid in train_ids], axis=0
        )
    img_norm = {
        'mean': img_train_all.mean(axis=0).astype(np.float32),
        'std':  img_train_all.std(axis=0).astype(np.float32).clip(min=1e-6),
    }
    logger.info(
        f"  Imaging norm — mean: [{img_norm['mean'].min():.4f}, {img_norm['mean'].max():.4f}], "
        f"std: [{img_norm['std'].min():.4f}, {img_norm['std'].max():.4f}]"
    )
    del img_train_all

    # ── 6. Datasets ──────────────────────────────────────────────────────
    def _make_ds(ids: list[str], mode: str) -> Img2RNADataset:
        return Img2RNADataset(
            rna_data={sid: rna_data[sid] for sid in ids},
            sample_ids=ids,
            img_cache_dir=img_cache_dir,
            img_cache_size=img_cache_size,
            n_imaging_cells=n_imaging_cells,
            mode=mode,
            seed=seed,
            preload_imaging=preload_imaging,
            img_data={sid: img_data[sid] for sid in ids} if img_data else None,
            rna_norm=rna_norm,
            img_norm=img_norm,
        )

    train_ds = _make_ds(train_ids, "train")
    val_ds   = _make_ds(val_ids,   "val")

    # ── 7. DataLoaders ────────────────────────────────────────────────────
    # With preload_imaging=True, num_workers>0 is safe: forked workers share
    # the pre-loaded numpy arrays (Linux COW fork — no extra RAM copies as
    # long as arrays are only read-accessed).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    info = {
        "img_dim":         img_dim,
        "rna_dim":         rna_dim,
        "n_train_samples": len(train_ids),
        "n_val_samples":   len(val_ids),
        "n_train_cells":   len(train_ds),
        "n_val_cells":     len(val_ds),
        "rna_norm":        rna_norm,
        "img_norm":        img_norm,
    }
    return train_loader, val_loader, info



# ──────────────────────────────────────────────────────────────────────────────
# RNA-seq: load fully into RAM (~1-2 GB)
# ──────────────────────────────────────────────────────────────────────────────

