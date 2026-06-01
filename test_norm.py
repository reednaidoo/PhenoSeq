"""Quick test to verify normalization pipeline works correctly (lightweight)."""
import numpy as np
import torch
from pathlib import Path

rna_dir = Path('data/cache/rnaseq')
img_dir = Path('data/cache/imaging')

rna_files = sorted(rna_dir.glob('*.npz'))[:3]

rna_data = {f.stem: np.load(f)['X'] for f in rna_files}
rna_all = np.concatenate(list(rna_data.values()), axis=0)
rna_norm = {
    'mean': rna_all.mean(axis=0).astype(np.float32),
    'std':  rna_all.std(axis=0).astype(np.float32).clip(min=1e-6),
}
print(f"RNA raw:  mean={rna_all.mean():.4f}, std={rna_all.std():.4f}")
rna_normed = (rna_all - rna_norm['mean']) / rna_norm['std']
print(f"RNA norm: mean={rna_normed.mean():.4f}, std={rna_normed.std():.4f}")
print(f"  -> Should be ~0.0 mean and ~1.0 std")

from models.denoiser import Img2RNADenoiser
from models.diffusion import GaussianDiffusion

rna_dim = rna_all.shape[1]
denoiser = Img2RNADenoiser(img_dim=5120, rna_dim=rna_dim, model_dim=256, num_heads=4, num_layers=2)
diffusion = GaussianDiffusion(denoiser=denoiser, rna_norm=rna_norm)

print(f"\nDiffusion rna_mean registered: {diffusion.rna_mean is not None}")
print(f"Diffusion rna_std registered: {diffusion.rna_std is not None}")

raw = rna_all[:4].copy()
normed_t = (raw - rna_norm['mean']) / rna_norm['std']
unnormed = diffusion.unnormalize_rna(torch.from_numpy(normed_t)).numpy()
err = np.abs(raw - unnormed).max()
print(f"Roundtrip error: {err:.2e} (should be ~0)")

from data.dataset import Img2RNADataset
sample_ids = list(rna_data.keys())[:1]
ds = Img2RNADataset(
    rna_data={sid: rna_data[sid] for sid in sample_ids},
    sample_ids=sample_ids,
    img_cache_dir=img_dir,
    n_imaging_cells=4,
    mode='val',
    preload_imaging=False,
    rna_norm=rna_norm,
    img_norm=None,
)
item = ds[0]
print(f"\nDataset item rna: mean={item['rna_embedding'].mean():.4f}, std={item['rna_embedding'].std():.4f}")

print("\nAll tests passed!")
