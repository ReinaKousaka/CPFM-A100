"""Milestone evaluation for latent-256 checkpoints: FID-10k + sample grid.

  # once (downloads + decodes 10k validation latents as the reference set):
  python -m ba_pfm.eval_latent --make_ref
  # per milestone:
  python -m ba_pfm.eval_latent --run fm_base-s0 --ckpt ckpt_step050000.pt \
      --nfe 250 --cfg 1.0 --n 10000

Sampler: Euler on velocity, t 1->0; CFG on velocity (v_u + w*(v_c - v_u), w=1 is
unguided). Reference = decoded validation latents through the SAME sd-vae-ft-ema
decoder, so FID measures the model, not the VAE. Model NFE = nfe * (2 if w>1 else 1);
VAE decodes reported separately (1 per image, never counted as NFE).
"""

import argparse
import json
import os
import urllib.request

import torch
from torchvision.utils import save_image

from .latent_data import SCALE, LatentDataset, convert_shards
from .sit import build_sit
from .vae import FrozenVAE

RUNS = "runs/latent256"
REF_NAME = "latent256_valref"


@torch.no_grad()
def sample(model, vae, n, nfe, cfg, device, seed, folder, batch=64):
    """Generate n images, streaming each decoded batch straight to folder as
    PNGs (no RAM accumulation — review 2026-08-06). Returns the first 64 images
    for the preview grid."""
    grid = []
    g = torch.Generator("cpu").manual_seed(seed)
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, 4, 32, 32, generator=g).to(device)
        y = torch.randint(0, 1000, (b,), generator=g).to(device)
        y_null = torch.full_like(y, 1000)
        ts = torch.linspace(1.0, 0.0, nfe + 1)
        for j in range(nfe):
            t = torch.full((b,), float(ts[j]), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(z, t, y).float()
                if cfg > 1.0:
                    v_u = model(z, t, y_null).float()
                    v = v_u + cfg * (v - v_u)
            z = z + (float(ts[j + 1]) - float(ts[j])) * v
        imgs = vae.decode_model(z).clamp(-1, 1).cpu()
        for k in range(imgs.shape[0]):
            save_image((imgs[k] + 1) / 2, os.path.join(folder, f"{i + k:06d}.png"))
        n_grid = sum(g.shape[0] for g in grid)
        if n_grid < 64:
            grid.append(imgs[: 64 - n_grid])
    return torch.cat(grid)


def make_ref(device="cuda", n=10000):
    url = ("https://huggingface.co/datasets/Forbu14/imagenet-1k-latent/resolve/"
           "64d39472db7f/data/validation-00000-of-00002.parquet")
    dst = "data/latents/validation-00000-of-00002.parquet"
    if not os.path.exists(dst):
        print("[eval_latent] downloading validation shard ...")
        urllib.request.urlretrieve(url, dst)
    convert_shards("validation", max_shards=1)
    ds = LatentDataset("validation", max_shards=1, flip=False)
    vae = FrozenVAE(device=device, use_checkpoint=False)
    folder = os.path.join(RUNS, "fid_ref")
    os.makedirs(folder, exist_ok=True)
    n = min(n, len(ds))
    for i in range(0, n, 64):
        zs = torch.stack([ds[j][0] for j in range(i, min(i + 64, n))]).to(device)
        with torch.no_grad():
            imgs = vae.decode_model(zs).clamp(-1, 1)
        for k in range(imgs.shape[0]):
            save_image((imgs[k] + 1) / 2, os.path.join(folder, f"{i + k:06d}.png"))
    from cleanfid import fid
    fid.make_custom_stats(REF_NAME, folder, mode="clean")
    print(f"[eval_latent] reference stats '{REF_NAME}' from {n} decoded val latents")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make_ref", action="store_true")
    p.add_argument("--run")
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--nfe", type=int, default=250)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.make_ref:
        make_ref(args.device)
        return

    from cleanfid import fid
    run_dir = os.path.join(RUNS, args.run)
    ck = torch.load(os.path.join(run_dir, args.ckpt), map_location="cpu",
                    weights_only=True)
    model = build_sit("B").to(args.device)
    model.load_state_dict(ck["ema"])
    model.eval()
    vae = FrozenVAE(device=args.device, use_checkpoint=False)

    tag = f"{args.ckpt.replace('.pt', '')}_nfe{args.nfe}_cfg{args.cfg}"
    if args.n != 10000:  # n is part of the identity for non-default sizes
        tag += f"_n{args.n}"
    folder = os.path.join(run_dir, f"gen_{tag}")
    # never score a mixed folder: wipe any pre-existing contents (a rerun after
    # a crash or with different args must not blend old and new images)
    if os.path.isdir(folder) and os.listdir(folder):
        import shutil
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    grid = sample(model, vae, args.n, args.nfe, args.cfg, args.device,
                  args.seed, folder, args.batch)
    n_png = len([f for f in os.listdir(folder) if f.endswith(".png")])
    assert n_png == args.n, f"generated {n_png} != requested {args.n}"
    save_image((grid + 1) / 2, os.path.join(run_dir, f"grid_{tag}.png"), nrow=8)
    score = fid.compute_fid(folder, dataset_name=REF_NAME, mode="clean",
                            dataset_split="custom")
    rec = {"run": args.run, "ckpt": args.ckpt, "nfe": args.nfe, "cfg": args.cfg,
           "n": args.n, "seed": args.seed, "batch": args.batch, "fid": score,
           "model_nfe_true": args.nfe * (2 if args.cfg > 1.0 else 1),
           "vae_decodes_per_image": 1}
    with open(os.path.join(run_dir, f"fid_{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[eval_latent] {tag}: FID {score:.2f}")


if __name__ == "__main__":
    main()
