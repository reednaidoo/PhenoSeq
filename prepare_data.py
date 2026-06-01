"""
One-time data preparation: extract per-sample arrays from h5ad files.

The h5ad imaging and RNA-seq files store cells in random (non-contiguous)
order relative to Sample_ID.  Because the X matrix has no HDF5 chunking,
random-access by sample during training means reading nearly the entire file
on every sample access → extremely slow and high RAM.

This script reads each file ONCE, sequentially, then splits and saves per-
sample arrays as compressed .npz files.  Training then just calls np.load().

Usage:
    python prepare_data.py                        # uses paths from config.yaml
    python prepare_data.py --config config.yaml --out_dir data/cache
    python prepare_data.py --help

Output layout:
    <out_dir>/
        imaging/
            <Sample_ID>.npz   (key 'X', shape [n_cells, 5120])
        rnaseq/
            <Sample_ID>.npz   (key 'X', shape [n_cells, 512])

Time estimate: ~5-15 min total (reads each file once, sequentially).
RAM required: ~20 GB peak (one full file loaded at a time).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_categorical(f: h5py.File, key: str) -> np.ndarray:
    """Decode a stored categorical obs column into an array of strings."""
    grp = f[key]
    if isinstance(grp, h5py.Group) and "codes" in grp:
        cats = grp["categories"][:].astype(str)
        codes = grp["codes"][:]
        return cats[codes]
    # plain string array
    return f[key][:].astype(str)


def _build_per_sample_index(f: h5py.File, sample_id_col: str) -> dict[str, np.ndarray]:
    """Return {sample_id: sorted_row_indices} for all samples in the file."""
    obs_key = f"obs/{sample_id_col}"
    sample_ids = _read_categorical(f, obs_key)
    n = len(sample_ids)
    unique = np.unique(sample_ids)
    index: dict[str, np.ndarray] = {}
    for sid in unique:
        rows = np.where(sample_ids == sid)[0]
        index[sid] = np.sort(rows)          # sort for sequential HDF5 read
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_imaging(
    h5ad_paths: list[str],
    out_dir: Path,
    sample_id_col: str = "Sample_ID",
    force: bool = False,
) -> None:
    """Read each imaging h5ad once and write per-sample .npz files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in h5ad_paths:
        logger.info(f"Processing imaging file: {path}")
        t0 = time.time()

        with h5py.File(path, "r") as f:
            # ── 1. Build sample → row index ───────────────────────────────
            index = _build_per_sample_index(f, sample_id_col)
            n_cells, n_feat = f["X"].shape
            logger.info(
                f"  {n_cells:,} cells, {n_feat} dims, {len(index)} samples"
            )

            # ── 2. Load full X into RAM (one sequential read) ─────────────
            logger.info(f"  Reading X ({n_cells * n_feat * 4 / 1e9:.1f} GB) ...")
            X = f["X"][:]           # shape (n_cells, n_feat), float32

        # ── 3. Split and save per sample ──────────────────────────────────
        n_written = n_skipped = 0
        for sid, rows in index.items():
            fpath = out_dir / f"{sid}.npz"
            if fpath.exists() and not force:
                n_skipped += 1
                continue
            np.savez_compressed(fpath, X=X[rows].astype(np.float32))
            n_written += 1

        del X
        elapsed = time.time() - t0
        logger.info(
            f"  Written {n_written}, skipped {n_skipped} existing. "
            f"({elapsed:.0f}s)"
        )


def extract_rnaseq(
    h5ad_paths: list[str],
    out_dir: Path,
    sample_id_col: str = "Sample_ID",
    obsm_key: str = "scGPT",
    force: bool = False,
) -> None:
    """Read each RNA-seq h5ad once and write per-sample .npz files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in h5ad_paths:
        logger.info(f"Processing RNA-seq file: {path}")
        t0 = time.time()

        with h5py.File(path, "r") as f:
            # ── 1. Build sample → row index ───────────────────────────────
            index = _build_per_sample_index(f, sample_id_col)

            # ── 2. Find the obsm dataset path ─────────────────────────────
            obsm_path = f"obsm/{obsm_key}"
            if obsm_path not in f:
                # anndata sometimes nests obsm differently
                for k in f.get("obsm", {}).keys():
                    if k.lower() == obsm_key.lower():
                        obsm_path = f"obsm/{k}"
                        break
                else:
                    raise KeyError(
                        f"obsm key '{obsm_key}' not found in {path}. "
                        f"Available: {list(f.get('obsm', {}).keys())}"
                    )

            n_cells, n_feat = f[obsm_path].shape
            logger.info(
                f"  {n_cells:,} cells, {n_feat} embedding dims, {len(index)} samples"
            )

            # ── 3. Load full embedding matrix (one sequential read) ───────
            logger.info(
                f"  Reading obsm['{obsm_key}'] "
                f"({n_cells * n_feat * 4 / 1e9:.2f} GB) ..."
            )
            E = f[obsm_path][:]     # shape (n_cells, n_feat), float32

        # ── 4. Split and save per sample ──────────────────────────────────
        n_written = n_skipped = 0
        for sid, rows in index.items():
            fpath = out_dir / f"{sid}.npz"
            if fpath.exists() and not force:
                n_skipped += 1
                continue
            np.savez_compressed(fpath, X=E[rows].astype(np.float32))
            n_written += 1

        del E
        elapsed = time.time() - t0
        logger.info(
            f"  Written {n_written}, skipped {n_skipped} existing. "
            f"({elapsed:.0f}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pre-extract per-sample .npz files from h5ad.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--out_dir", default=None,
        help="Root output directory (default: data/cache next to config)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .npz files",
    )
    p.add_argument(
        "--imaging_only", action="store_true", help="Only process imaging files"
    )
    p.add_argument(
        "--rnaseq_only", action="store_true", help="Only process RNA-seq files"
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    img_paths  = data_cfg["imaging_paths"]
    rna_paths  = data_cfg["rnaseq_paths"]
    sid_col    = data_cfg.get("sample_id_col", "Sample_ID")
    obsm_key   = data_cfg.get("rnaseq_obsm_key", "scGPT")

    root = Path(args.out_dir) if args.out_dir else cfg_path.parent / "data" / "cache"
    logger.info(f"Output root: {root}")

    if not args.rnaseq_only:
        logger.info("=== Extracting imaging features ===")
        extract_imaging(
            img_paths,
            out_dir=root / "imaging",
            sample_id_col=sid_col,
            force=args.force,
        )

    if not args.imaging_only:
        logger.info("=== Extracting RNA-seq embeddings ===")
        extract_rnaseq(
            rna_paths,
            out_dir=root / "rnaseq",
            sample_id_col=sid_col,
            obsm_key=obsm_key,
            force=args.force,
        )

    logger.info("Done. You can now run train.py — it will load from .npz files.")


if __name__ == "__main__":
    main()
