"""Improved precision & recall (Kynkaenniemi et al. 2019, k-NN radii) in
DINOv2-S CLS space for latent-256 gen folders vs the decoded-val reference.

  python -m ba_pfm.eval_prdc --run pfm_fixed_25k-s0 --gen gen_ckpt_step025000_nfe4_cfg1.0

Writes prdc_<gen>.json next to the folder. Registered 25k-gate metric:
recall within 10% relative of continued-FM at the winning budgets.
"""

import argparse
import json
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .perceptual import _IMNET_MEAN, _IMNET_STD

RUNS = "runs/latent256"
REF = os.path.join(RUNS, "fid_ref")


@torch.no_grad()
def folder_feats(folder, net, device, batch=64, n=None):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))[:n]
    feats = []
    for i in range(0, len(files), batch):
        imgs = torch.stack([
            TF.to_tensor(Image.open(os.path.join(folder, f)).convert("RGB"))
            for f in files[i:i + batch]]).to(device)
        imgs = TF.resize(imgs, [224, 224], antialias=True)
        imgs = (imgs - _IMNET_MEAN.to(device)) / _IMNET_STD.to(device)
        feats.append(net(imgs).float())
    return torch.cat(feats)


def knn_radii(feats, k=3):
    d = torch.cdist(feats, feats)
    d.fill_diagonal_(float("inf"))
    return d.topk(k, largest=False).values[:, -1]


def coverage(a, b, radii_b):
    d = torch.cdist(a, b)
    return (d <= radii_b[None, :]).any(dim=1).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--gen", required=True, help="gen folder name inside the run dir")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--knn", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from .perceptual import DINO_REPO
    net = torch.hub.load(DINO_REPO, "dinov2_vits14")
    net.eval().requires_grad_(False).to(args.device)

    gen_dir = os.path.join(RUNS, args.run, args.gen)
    gf = folder_feats(gen_dir, net, args.device, n=args.n)
    rf = folder_feats(REF, net, args.device, n=args.n)
    precision = coverage(gf, rf, knn_radii(rf, args.knn))
    recall = coverage(rf, gf, knn_radii(gf, args.knn))
    rec = {"run": args.run, "gen": args.gen, "n_gen": gf.shape[0],
           "n_ref": rf.shape[0], "knn": args.knn,
           "precision": precision, "recall": recall}
    out = os.path.join(RUNS, args.run, f"prdc_{args.gen}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[eval_prdc] {args.run}/{args.gen}: precision {precision:.3f} "
          f"recall {recall:.3f}")


if __name__ == "__main__":
    main()
