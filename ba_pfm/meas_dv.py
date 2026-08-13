"""Retroactive velocity-field drift measurement (Amendment 6).

D_v(ckpt) = E[ ||v_theta(x_t,t,y) - v_init(x_t,t,y)||^2 / (||v_init||^2 + eps) ]
per-sample ratios on fresh interpolation-path points (x_t from real latents,
t ~ U(0,1), true labels), teacher = frozen 200k EMA. Reported overall and in
4 t-buckets. Calibrates the Trust-PFM budget delta and tests directly whether
drift keeps growing through the overshoot while training FM loss stays flat.

  python -m ba_pfm.meas_dv --run pfm_fixed_25k-s0 --ckpt ckpt_step010000.pt
"""

import argparse
import json
import os

import torch

from .latent_data import LatentDataset
from .sit import build_sit

RUNS = "runs/latent256"
INIT = os.path.join(RUNS, "fm_base-s0", "ckpt_step200000.pt")


@torch.no_grad()
def measure(run, ckpt, device="cuda", n_batches=64, batch=32, seed=123):
    torch.manual_seed(seed)
    teacher = build_sit("B").to(device)
    teacher.load_state_dict(torch.load(INIT, map_location=device,
                                       weights_only=True)["ema"])
    teacher.eval()
    student = build_sit("B").to(device)
    student.load_state_dict(torch.load(os.path.join(RUNS, run, ckpt),
                                       map_location=device,
                                       weights_only=True)["ema"])
    student.eval()

    ds = LatentDataset("train", max_shards=2, flip=False)
    g = torch.Generator().manual_seed(seed)
    ratios, tvals = [], []
    for _ in range(n_batches):
        idx = torch.randint(0, len(ds), (batch,), generator=g)
        z0 = torch.stack([ds[int(i)][0] for i in idx]).to(device)
        y = torch.tensor([ds[int(i)][1] for i in idx]).to(device)
        t = torch.rand(batch, device=device)
        eps = torch.randn_like(z0)
        tb = t[:, None, None, None]
        zt = (1 - tb) * z0 + tb * eps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v0 = teacher(zt, t, y).float()
            v1 = student(zt, t, y).float()
        num = (v1 - v0).pow(2).flatten(1).sum(1)
        den = v0.pow(2).flatten(1).sum(1) + 1e-8
        ratios.append((num / den).cpu())
        tvals.append(t.cpu())
    r = torch.cat(ratios)
    t = torch.cat(tvals)
    buckets = {}
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        m = (t >= lo) & (t < hi)
        buckets[f"t[{lo},{hi})"] = r[m].mean().item()
    return {"run": run, "ckpt": ckpt, "n": int(r.numel()),
            "D_v": r.mean().item(), "D_v_median": r.median().item(),
            "D_v_by_t": buckets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    rec = measure(args.run, args.ckpt, args.device)
    out = os.path.join(RUNS, args.run, f"dv_{args.ckpt.replace('.pt','')}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[meas_dv] {args.run}/{args.ckpt}: D_v={rec['D_v']:.4f} "
          f"by_t={ {k: round(v,4) for k,v in rec['D_v_by_t'].items()} }")


if __name__ == "__main__":
    main()
