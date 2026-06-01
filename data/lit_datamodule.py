"""
PyTorch Lightning LightningDataModule for Img2RNA.

Wraps the existing dataset / dataloader creation into a portable
LightningDataModule that can be used with any Lightning Trainer.
"""

from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import Img2RNADataset, create_dataloaders


class Img2RNADataModule(pl.LightningDataModule):
    """
    Lightning data module for paired imaging → RNA-seq data.

    Delegates heavy lifting to :func:`data.dataset.create_dataloaders`.
    """

    def __init__(
        self,
        # Data paths
        img_paths: list[str] | None = None,
        rna_paths: list[str] | None = None,
        # Column / key names
        sample_id_col: str = "Sample_ID",
        rna_obsm_key: str = "scGPT",
        # Dataset options
        n_imaging_cells: int = 16,
        train_ratio: float = 0.8,
        seed: int = 42,
        # DataLoader options
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._train_ds: Optional[Img2RNADataset] = None
        self._val_ds: Optional[Img2RNADataset] = None
        self._data_info: dict[str, Any] = {}

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        """Called on every process in DDP. Builds datasets once."""
        if self._train_ds is not None:
            return  # already set up

        result = create_dataloaders(
            img_paths=self.hparams.img_paths,
            rna_paths=self.hparams.rna_paths,
            sample_id_col=self.hparams.sample_id_col,
            rna_obsm_key=self.hparams.rna_obsm_key,
            n_imaging_cells=self.hparams.n_imaging_cells,
            train_ratio=self.hparams.train_ratio,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            seed=self.hparams.seed,
        )

        self._train_ds = result["train_loader"].dataset
        self._val_ds = result["val_loader"].dataset
        self._data_info = result["info"]

    @property
    def data_info(self) -> dict[str, Any]:
        return self._data_info

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
        )
