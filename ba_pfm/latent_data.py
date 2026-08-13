"""Forbu14/imagenet-1k-latent loader.

Stored latents are RAW sd-vae-ft-ema outputs (empirical global std 4.63,
per-channel 3.97-5.41 measured on 12k samples of shard 0) — NOT pre-scaled.
Model space applies the SiT/DiT convention: z_model = 0.18215 * z_stored.

Parquet shards are converted once to a fp16 memmap (fast random access,
near-zero worker RAM). Latents are deterministic single encodings (center-crop
protocol, no stored flip variants): the only augmentation we apply is a
horizontal flip *of the latent feature map*, which for a conv VAE approximates
image-space flip; recorded as a protocol difference vs official SiT preprocessing.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

SCALE = 0.18215
LATENT_DIR = "data/latents"


def convert_shards(split="train", max_shards=None, latent_dir=LATENT_DIR):
    files = sorted(glob.glob(os.path.join(latent_dir, f"{split}-*.parquet")))
    if max_shards:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"no {split} parquet shards in {latent_dir}")
    tag = f"{split}_{len(files)}shards"
    z_path = os.path.join(latent_dir, f"{tag}_z.fp16.npy")
    y_path = os.path.join(latent_dir, f"{tag}_y.npy")
    meta_path = os.path.join(latent_dir, f"{tag}_meta.json")
    if os.path.exists(meta_path):
        return z_path, y_path, meta_path

    import pyarrow.parquet as pq
    counts = [pq.read_metadata(f).num_rows for f in files]
    total = sum(counts)
    z_mm = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float16,
                                     shape=(total, 4, 32, 32))
    y_all = np.zeros(total, dtype=np.int16)
    pos = 0
    for f, n in zip(files, counts):
        t = pq.read_table(f)
        z = np.array(t.column("latents").to_pylist(), dtype=np.float32)
        z_mm[pos:pos + n] = z.astype(np.float16)
        y_all[pos:pos + n] = np.array(t.column("label_latent").to_pylist(), dtype=np.int16)
        pos += n
        print(f"  converted {os.path.basename(f)} ({pos}/{total})")
    z_mm.flush()
    np.save(y_path, y_all)
    with open(meta_path, "w") as fh:
        json.dump({"files": files, "total": total}, fh)
    return z_path, y_path, meta_path


class LatentDataset(Dataset):
    def __init__(self, split="train", max_shards=None, flip=True, latent_dir=LATENT_DIR):
        z_path, y_path, _ = convert_shards(split, max_shards, latent_dir)
        self.z = np.load(z_path, mmap_mode="r")
        self.y = np.load(y_path)
        self.flip = flip

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        z = torch.from_numpy(self.z[i].astype(np.float32)) * SCALE
        if self.flip and torch.rand(()) < 0.5:
            z = torch.flip(z, dims=(2,))
        return z, int(self.y[i])
