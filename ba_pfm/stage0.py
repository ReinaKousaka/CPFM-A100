"""Stage-0 sanity gates (PHASE1A_PLAN.md) — run BEFORE any base training.

  python -m ba_pfm.stage0 --step contract    # latent stats + decode grids (3 interpretations)
  python -m ba_pfm.stage0 --step integrity   # counts, labels, loader throughput
  python -m ba_pfm.stage0 --step numeric     # bf16/fp32, z0 formula, label dropout, EMA, resume
  python -m ba_pfm.stage0 --step overfit     # tiny-overfit gate (calls train_latent)
  python -m ba_pfm.stage0 --step bench       # 5-arm throughput/memory benchmark

Each step writes runs/latent256/stage0/<step>.json (+ pngs). No step trains a base.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .latent_data import SCALE, LatentDataset
from .perceptual import BAWeights, BLOCKS, LatentPerceptual, calibrate
from .sit import build_sit
from .vae import FrozenVAE

OUT = "runs/latent256/stage0"
PY = ".venv/bin/python"


def _save(step, payload):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{step}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[stage0] wrote {OUT}/{step}.json")


def contract(device="cuda"):
    ds = LatentDataset("train", max_shards=1, flip=False)
    n = min(12000, len(ds))
    zs = torch.stack([torch.from_numpy(ds.z[i].astype(np.float32)) for i in range(n)])
    stats = {
        "n_samples": n,
        "stored_global": {"mean": zs.mean().item(), "std": zs.std().item(),
                          "min": zs.min().item(), "max": zs.max().item()},
        "stored_per_channel": {
            f"ch{c}": {"mean": zs[:, c].mean().item(), "std": zs[:, c].std().item()}
            for c in range(4)},
        "interpretation": "stored latents are RAW E(x); z_model = 0.18215 * z_stored",
        "model_space_std_after_scale": (zs.std() * SCALE).item(),
    }
    vae = FrozenVAE(device=device, use_checkpoint=False)
    z16 = zs[:16].to(device)
    with torch.no_grad():
        img_correct = vae.decode_raw(z16)                    # raw interpretation
        img_wrongscale = vae.decode_raw(z16 / SCALE)         # "pre-scaled" interpretation
    save_image((img_correct.clamp(-1, 1) + 1) / 2, f"{OUT}/recon_correct_raw.png", nrow=4)
    save_image((img_wrongscale.clamp(-1, 1) + 1) / 2, f"{OUT}/recon_wrong_scale.png", nrow=4)
    stats["decoded_range_correct"] = [img_correct.min().item(), img_correct.max().item()]
    stats["decoded_range_wrong_scale"] = [img_wrongscale.min().item(), img_wrongscale.max().item()]
    del vae
    torch.cuda.empty_cache()
    vx = FrozenVAE(model_id="stabilityai/sdxl-vae", device=device, use_checkpoint=False)
    with torch.no_grad():
        img_wrongvae = vx.decode_raw(z16)                    # wrong-VAE check
    save_image((img_wrongvae.clamp(-1, 1) + 1) / 2, f"{OUT}/recon_wrong_vae_sdxl.png", nrow=4)
    stats["decoded_range_wrong_vae_sdxl"] = [img_wrongvae.min().item(), img_wrongvae.max().item()]
    _save("contract", stats)


def integrity():
    import pyarrow.parquet as pq
    import glob
    files = sorted(glob.glob("data/latents/train-*.parquet"))
    shard_rows = {os.path.basename(f): pq.read_metadata(f).num_rows for f in files}
    ds = LatentDataset("train", max_shards=len(files))
    y = ds.y
    uniq, cnt = np.unique(y, return_counts=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
    t0 = time.time()
    seen = 0
    for i, (z, yy) in enumerate(loader):
        seen += z.shape[0]
        if i == 40:
            break
    thr = seen / (time.time() - t0)
    _save("integrity", {
        "local_shards": shard_rows, "local_total": int(len(ds)),
        "card_claim_total_train": 1281167,
        "label_min": int(y.min()), "label_max": int(y.max()),
        "n_classes_present": int(len(uniq)),
        "class_count_min_max_mean": [int(cnt.min()), int(cnt.max()), float(cnt.mean())],
        "tensor": {"shape": list(ds.z.shape[1:]), "stored_dtype": "float16 memmap",
                   "loaded_dtype": "float32 * 0.18215"},
        "aug_note": "single deterministic latent per image; only latent-space hflip "
                    "applied (protocol difference vs SiT image-space aug, same for all arms)",
        "loader_samples_per_sec": thr,
    })


def numeric(device="cuda"):
    torch.manual_seed(0)
    rep = {}
    # z0 formula against analytic case
    z0 = torch.randn(8, 4, 32, 32)
    eps = torch.randn_like(z0)
    t = torch.rand(8)
    tb = t[:, None, None, None]
    zt = (1 - tb) * z0 + tb * eps
    v_true = eps - z0
    z0_hat = zt - tb * v_true
    rep["z0_formula_max_err"] = (z0_hat - z0).abs().max().item()
    # label dropout rate
    model = build_sit("B").to(device)
    y = torch.randint(0, 1000, (20000,), device=device)
    model.train()
    emb_null = model.y_embed.table.weight[1000]
    e = model.y_embed(y, train=True)
    frac = (e == emb_null).all(dim=-1).float().mean().item()
    rep["label_dropout_measured"] = frac
    # bf16 vs fp32 loss
    ds = LatentDataset("train", max_shards=1, flip=False)
    z = torch.stack([ds[i][0] for i in range(16)]).to(device)
    yy = torch.tensor([ds[i][1] for i in range(16)], device=device)
    tt = torch.full((16,), 0.5, device=device)
    ee = torch.randn_like(z)
    zt2 = 0.5 * z + 0.5 * ee
    model.eval()
    with torch.no_grad():
        v32 = model(zt2, tt, yy)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v16 = model(zt2, tt, yy)
    l32 = F.mse_loss(v32, ee - z).item()
    l16 = F.mse_loss(v16.float(), ee - z).item()
    rep["fm_loss_fp32"] = l32
    rep["fm_loss_bf16"] = l16
    rep["bf16_rel_diff"] = abs(l16 - l32) / max(l32, 1e-9)
    # EMA math on a probe entry
    from .train_latent import EMA
    ema = EMA(model, 0.9)
    k = next(iter(ema.shadow))
    before = ema.shadow[k].clone()
    with torch.no_grad():
        model.state_dict()[k].add_(1.0)
    ema.update(model)
    delta = (ema.shadow[k] - before).abs().mean().item()
    rep["ema_moves_toward_model"] = delta > 0
    # resume check (subprocess, tiny)
    subprocess.run([PY, "-m", "ba_pfm.train_latent", "--arm", "fm",
                    "--steps", "60", "--batch", "8", "--max_shards", "1",
                    "--ckpt_every", "50", "--tag", "_resumetest", "--overfit_n", "64"],
                   check=True, capture_output=True)
    # ckpt_final exists means the 60-step run completed; rerun should skip instantly
    r2 = subprocess.run([PY, "-m", "ba_pfm.train_latent", "--arm", "fm",
                         "--steps", "60", "--batch", "8", "--max_shards", "1",
                         "--ckpt_every", "50", "--tag", "_resumetest", "--overfit_n", "64"],
                        check=True, capture_output=True, text=True)
    rep["skip_if_done_works"] = "nothing to do" in r2.stdout
    _save("numeric", rep)


def overfit():
    # bapfm tiny-overfit: 512 samples, 2k steps
    subprocess.run([PY, "-m", "ba_pfm.train_latent", "--arm", "bapfm", "--lam_fm", "0.3",
                    "--steps", "2000", "--batch", "16",
                    "--max_shards", "1", "--overfit_n", "512", "--tag", "_overfit",
                    "--ckpt_every", "100000", "--log_every", "25"], check=True)
    # verify from the log: losses decrease, grad norms finite, weights sane
    path = "runs/latent256/bapfm_overfit-s0/log.jsonl"
    losses, fm_losses, weight_recs = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "loss" in r:
                losses.append(r["loss"])
                fm_losses.append(r["fm_loss"])
            if "grad_norms" in r:
                weight_recs.append(r)
    gn_last = weight_recs[-1]["grad_norms"] if weight_recs else {}
    rep = {
        "loss_first5_mean": float(np.mean(losses[:5])),
        "loss_last5_mean": float(np.mean(losses[-5:])),
        "fm_loss_first5_mean": float(np.mean(fm_losses[:5])),
        "fm_loss_last5_mean": float(np.mean(fm_losses[-5:])),
        "losses_all_finite": bool(np.isfinite(losses).all()),
        "grad_norms_last": gn_last,
        "grad_norms_finite_nonzero": bool(all(np.isfinite(v) and v > 0 for v in gn_last.values())),
        "n_eff_trajectory": [r["n_eff"] for r in weight_recs],
        "weights_last": weight_recs[-1]["weights"] if weight_recs else {},
    }
    # frozen-module check + prediction grid
    device = "cuda"
    vae = FrozenVAE(device=device)
    perc = LatentPerceptual(vae, device=device,
                            calib_path="runs/latent256/dino_calib.json")
    rep["vae_params_require_grad"] = any(p.requires_grad for p in vae.vae.parameters())
    rep["dino_params_require_grad"] = any(p.requires_grad for p in perc.dino.net.parameters())
    ck = torch.load("runs/latent256/bapfm_overfit-s0/ckpt_final.pt",
                    map_location=device, weights_only=True)
    model = build_sit("B").to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    ds = LatentDataset("train", max_shards=1, flip=False)
    z0 = torch.stack([ds[i][0] for i in range(8)]).to(device)
    y = torch.tensor([ds[i][1] for i in range(8)], device=device)
    t = torch.full((8,), 0.7, device=device)
    eps = torch.randn(z0.shape, device=device, generator=None)
    zt = 0.3 * z0 + 0.7 * eps
    with torch.no_grad():
        v = model(zt, t, y)
        z0_hat = zt - 0.7 * v
        grid = torch.cat([vae.decode_raw(z0 / SCALE), vae.decode_raw(z0_hat / SCALE)])
    save_image((grid.clamp(-1, 1) + 1) / 2, f"{OUT}/overfit_pred_t0.7.png", nrow=8)
    # balanced vs fixed gradient norm comparison on the same batch
    zh = z0.clone().requires_grad_(True)
    w, n_eff = None, None
    baw = BAWeights()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        norms, _ = perc.probe_grad_norms(zh.detach(), z0)
        baw.update(norms)
        w, n_eff = baw.weights()
        lb, _ = perc.loss(zh, z0, weights=w)
        (gb,) = torch.autograd.grad(lb, zh)
        zh2 = z0.clone().requires_grad_(True)
        lf, _ = perc.loss(zh2, z0, weights=None)
        (gf,) = torch.autograd.grad(lf, zh2)
    rep["balanced_grad_norm"] = gb.norm().item()
    rep["fixed_grad_norm"] = gf.norm().item()
    rep["n_eff_fresh"] = n_eff
    rep["clip_active"] = bool(any(abs(w[b] - 1 / (5 * len(BLOCKS))) < 1e-9
                                  or abs(w[b] - 5 / len(BLOCKS)) < 1e-9 for b in BLOCKS))
    _save("overfit", rep)


def bench(device="cuda", steps=300, warmup=50, batch=16):
    ds = LatentDataset("train", max_shards=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                         num_workers=4, drop_last=True,
                                         persistent_workers=True)
    model = build_sit("B").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vae = FrozenVAE(device=device)
    perc = LatentPerceptual(vae, device=device,
                            calib_path="runs/latent256/dino_calib.json")
    if not perc.calib:
        calibrate(perc, ds, "runs/latent256/dino_calib.json")
    baw = BAWeights()

    def one_arm(arm, lam_fm):
        gen = iter(loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_data = t_probe = 0.0
        t_start = None
        for i in range(steps + warmup):
            if i == warmup:
                torch.cuda.synchronize()
                t_start = time.time()
            td = time.time()
            try:
                z0, y = next(gen)
            except StopIteration:
                gen = iter(loader)
                z0, y = next(gen)
            z0, y = z0.to(device, non_blocking=True), y.to(device)
            if t_start is not None:
                t_data += time.time() - td
            t = torch.rand(batch, device=device)
            eps = torch.randn_like(z0)
            tb = t[:, None, None, None]
            zt = (1 - tb) * z0 + tb * eps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(zt, t, y)
                fm = F.mse_loss(v.float(), (eps - z0).float())
                if arm == "fm":
                    loss = fm
                else:
                    z0h = zt - tb * v
                    w = baw.weights()[0] if arm.startswith("bapfm") else None
                    pl, _ = perc.loss(z0h, z0, weights=w)
                    loss = pl + lam_fm * fm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if arm.startswith("bapfm") and i % 100 == 0:
                tp = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    norms, _ = perc.probe_grad_norms(z0h[:16].detach(), z0[:16])
                baw.update(norms)
                if t_start is not None:
                    t_probe += time.time() - tp
        torch.cuda.synchronize()
        wall = time.time() - t_start
        return {"sec_per_step": wall / steps, "samples_per_sec": batch * steps / wall,
                "peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                "dataloader_wait_frac": t_data / wall,
                "probe_refresh_total_sec": t_probe,
                "nan_seen": bool(not torch.isfinite(loss))}

    rep = {"batch": batch, "steps_measured": steps, "device": torch.cuda.get_device_name(0)}
    for arm, lam in [("fm", 0.0), ("pfm_fixed", 0.0), ("bapfm_l0", 0.0),
                     ("bapfm_f", 0.3), ("bapfm_b", 1.0)]:
        rep[arm] = one_arm(arm, lam)
        print(arm, rep[arm])
    _save("bench", rep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True,
                   choices=["contract", "integrity", "numeric", "overfit", "bench"])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"contract": contract, "integrity": integrity, "numeric": numeric,
     "overfit": overfit, "bench": bench}[args.step]()


if __name__ == "__main__":
    main()
