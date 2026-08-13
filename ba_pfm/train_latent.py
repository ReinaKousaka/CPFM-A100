"""Latent-space training for Phase-1A arms.

Arms: fm (continued FM control) | pfm_fixed (equal calibrated multi-layer)
      | bapfm (gradient-balanced multi + FM anchor via --lam_fm)

x_t = (1-t) z0 + t*eps, v_target = eps - z0, z0_hat = x_t - t*v_hat (SiT linear).
Logs per-block weights and N_eff every refresh; checkpoints every --ckpt_every
with resume, mirroring pfm_pilot.train.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .latent_data import LatentDataset
from .perceptual import BAWeights, BLOCKS, LatentPerceptual, calibrate
from .sit import build_sit
from .vae import FrozenVAE

RUNS = "runs/latent256"


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def infinite(loader):
    while True:
        yield from loader


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["fm", "pfm_fixed", "pfm_matched", "bapfm", "pfm_auto",
                            "pfm_trust"])
    # LAUNCH BLOCKER (Amendment 7): teacher.eval() disables CFG label dropout but
    # the student drops 10% labels in train mode — the D_v term sees mismatched
    # labels and diverges from offline meas_dv calibration. Fix with a shared
    # external dropout mask before any launch.
    # pfm_trust (Amendment 6, launch gated on pfm_auto P2 failure): equal
    # perceptual weights + dual-ascent lambda_v on the VELOCITY-FIELD drift
    # constraint D_v <= dv_delta, D_v = per-sample ||v - v_init||^2 /
    # (||v_init||^2 + eps) vs the frozen-init teacher, averaged across ALL
    # accumulation micro-batches per step (reviewer's controller fix). FM loss
    # logged for monitoring only — it is NOT part of the loss.
    p.add_argument("--dv_delta", type=float, default=None,
                   help="pfm_trust: velocity-drift budget (from meas_dv calibration)")
    # pfm_auto: drift-budgeted anchoring. Equal perceptual weights (isolates the
    # anchoring mechanism from balancing); lambda_FM is a DUAL VARIABLE updated
    # by dual ascent on the constraint fm_ema <= (1+eps)*fm0, where fm0 is the
    # frozen-init FM loss measured before training. lambda stays ~0 while the
    # constraint has slack (fixed-PFM-speed early gains) and rises only as
    # drift approaches the budget (prevents the registered overshoot).
    p.add_argument("--drift_eps", type=float, default=0.15,
                   help="pfm_auto: allowed relative FM-loss drift over init")
    p.add_argument("--lam_lr", type=float, default=0.02,
                   help="pfm_auto: dual-ascent step size on lambda_FM")
    p.add_argument("--lam_max", type=float, default=1.5,
                   help="pfm_auto: safety cap on lambda_FM")
    # pfm_matched: equal per-block weights, but the TOTAL perceptual loss is
    # rescaled (EMA of ||g_balanced||/||g_fixed|| from the probe) so its combined
    # gradient magnitude matches the balanced arm — the control separating
    # per-layer balancing from merely weakening the perceptual update.
    p.add_argument("--lam_fm", type=float, default=0.0, help="FM anchor weight (bapfm)")
    p.add_argument("--lam_perc", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--init", default=None,
                   help="checkpoint to fine-tune from (omit to train from scratch)")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=32, help="micro-batch per step")
    p.add_argument("--accum", type=int, default=1, help="grad accumulation")
    p.add_argument("--perc_frac", type=float, default=1.0,
                   help="fraction of micro-batch used for the perceptual term")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--refresh_every", type=int, default=100)
    p.add_argument("--probe_batch", type=int, default=16)
    p.add_argument("--max_shards", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--milestone_every", type=int, default=0,
                   help="save immutable ckpt_stepNNNNNN.pt at this interval")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overfit_n", type=int, default=0,
                   help="stage-0: restrict to first N samples (tiny-overfit gate)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    name = args.arm + args.tag + f"-s{args.seed}"
    out = os.path.join(RUNS, name)
    final_path = os.path.join(out, "ckpt_final.pt")
    last_path = os.path.join(out, "ckpt_last.pt")
    if os.path.exists(final_path):
        print(f"[train_latent] {final_path} exists — nothing to do")
        return
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ds = LatentDataset("train", max_shards=args.max_shards)
    if args.overfit_n:
        ds = torch.utils.data.Subset(ds, range(args.overfit_n))
    g = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4,
        pin_memory=True, generator=g, persistent_workers=True)
    batches = infinite(loader)

    model = build_sit("B").to(args.device)
    start_step = 1
    resume = os.path.exists(last_path)
    if resume:
        ck = torch.load(last_path, map_location=args.device, weights_only=True)
        model.load_state_dict(ck["model"])
    elif args.init:
        ck0 = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ck0["ema"])
        print(f"[train_latent] initialized from {args.init}")
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    perc, baw = None, None
    if args.arm != "fm":
        vae = FrozenVAE(device=args.device)
        calib_path = os.path.join(RUNS, "dino_calib.json")
        perc = LatentPerceptual(vae, device=args.device, calib_path=calib_path)
        if not perc.calib:
            print("[train_latent] calibrating block scales on decoded latents ...")
            calibrate(perc, ds if not args.overfit_n else ds.dataset, calib_path)
        # bapfm balances; pfm_matched measures balanced-vs-fixed scale (control);
        # pfm_fixed/pfm_auto carry a diagnostic-only BAWeights (probe logging
        # below) — weights NEVER enter the loss (w passed only when arm==bapfm).
        if args.arm in ("bapfm", "pfm_matched", "pfm_fixed", "pfm_auto",
                        "pfm_trust"):
            baw = BAWeights()
    match_scale = 1.0  # pfm_matched: EMA of ||g_balanced|| / ||g_fixed||
    lam_auto, fm0, fm_ema = 0.0, None, None  # pfm_auto dual-ascent state
    lam_v, dv_ema = 0.0, None                # pfm_trust dual-ascent state
    teacher = None
    if args.arm == "pfm_trust":
        assert args.dv_delta is not None and args.init, \
            "pfm_trust requires --dv_delta (calibrated) and --init"
        teacher = build_sit("B").to(args.device)
        teacher.load_state_dict(
            torch.load(args.init, map_location=args.device,
                       weights_only=True)["ema"])
        teacher.eval().requires_grad_(False)

    if resume:
        ema.shadow = {k: v.to(args.device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        if baw is not None and "baw" in ck:
            baw.load(json.loads(ck["baw"]))
        # restore the full stochastic trajectory (review 2026-08-06): match_scale
        # (pfm_matched would otherwise reset its control definition mid-run) and
        # the RNG streams behind t/eps/CFG-dropout/perc-subset (CUDA) and epoch
        # shuffles (loader generator). Worker prefetch position is not exactly
        # recoverable — data order after resume is same-distribution, not
        # sample-exact. Older checkpoints lack these keys (.get fallbacks).
        match_scale = ck.get("match_scale", match_scale)
        lam_auto = ck.get("lam_auto", lam_auto)
        fm0 = ck.get("fm0", fm0)
        fm_ema = ck.get("fm_ema", fm_ema)
        lam_v = ck.get("lam_v", lam_v)
        dv_ema = ck.get("dv_ema", dv_ema)
        if "rng_torch" in ck:
            torch.set_rng_state(ck["rng_torch"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
            g.set_state(ck["rng_loader"].cpu())
        start_step = ck["step"] + 1
        del ck
        print(f"[train_latent] resumed at step {start_step - 1}")

    if args.arm == "pfm_auto" and fm0 is None:
        # constraint reference: FM loss of the UNMODIFIED init, measured on 50
        # training batches before any update (consumes RNG/batches — part of
        # the registered arm protocol, identical for any seed)
        vals = []
        with torch.no_grad():
            for _ in range(50):
                z0m, ym = next(batches)
                z0m, ym = z0m.to(args.device), ym.to(args.device)
                tm = torch.rand(z0m.shape[0], device=args.device)
                em = torch.randn_like(z0m)
                zm = (1 - tm[:, None, None, None]) * z0m + tm[:, None, None, None] * em
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vm = model(zm, tm, ym)
                vals.append(F.mse_loss(vm.float(), (em - z0m).float()).item())
        fm0 = sum(vals) / len(vals)
        fm_ema = fm0
        print(f"[train_latent] pfm_auto fm0={fm0:.4f} budget=+{args.drift_eps:.0%} "
              f"lam_lr={args.lam_lr}")

    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    print(f"[train_latent] arm={args.arm} lam_fm={args.lam_fm} steps={args.steps} "
          f"micro-bs={args.batch} accum={args.accum} params={n_par:.1f}M -> {out}")

    log_f = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        dv_accum = 0.0
        for _ in range(args.accum):
            z0, y = next(batches)
            z0, y = z0.to(args.device, non_blocking=True), y.to(args.device)
            t = torch.rand(z0.shape[0], device=args.device)
            eps = torch.randn_like(z0)
            tb = t[:, None, None, None]
            zt = (1 - tb) * z0 + tb * eps

            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(zt, t, y)
                fm_loss = F.mse_loss(v.float(), (eps - z0).float())
                if args.arm == "fm":
                    loss = fm_loss
                else:
                    z0_hat = zt - tb * v
                    # perc_frac protocol: random unbiased subset each step (same
                    # RNG stream for all arms via --seed); subset MEAN is an
                    # unbiased estimator of the batch-mean perceptual loss — no
                    # inverse-probability compensation needed or applied.
                    k = max(1, int(round(args.perc_frac * z0.shape[0])))
                    sub = torch.randperm(z0.shape[0], device=z0.device)[:k]
                    w = baw.weights()[0] if args.arm == "bapfm" else None
                    p_loss, _ = perc.loss(z0_hat[sub], z0[sub], weights=w)
                    if args.arm == "pfm_matched":
                        p_loss = p_loss * match_scale
                    if args.arm == "pfm_trust":
                        with torch.no_grad():
                            v0 = teacher(zt, t, y).float()
                        num = (v.float() - v0).pow(2).flatten(1).sum(1)
                        den = v0.pow(2).flatten(1).sum(1) + 1e-8
                        dv_term = (num / den).mean()
                        dv_accum += dv_term.item()
                        loss = args.lam_perc * p_loss + lam_v * dv_term
                    else:
                        lam_eff = (lam_auto if args.arm == "pfm_auto"
                                   else args.lam_fm)
                        loss = args.lam_perc * p_loss + lam_eff * fm_loss
            (loss / args.accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        track_upr = step % 1000 == 0
        if track_upr:
            named = dict(model.named_parameters())
            probe_params = ["blocks.0.attn.in_proj_weight", "blocks.6.mlp.0.weight",
                            "x_embed.weight"]
            prev = {n: named[n].detach().clone() for n in probe_params}
        opt.step()
        ema.update(model)

        if args.arm == "pfm_auto":
            # dual ascent on the flow-drift constraint fm_ema <= (1+eps)*fm0:
            # lambda rises only when smoothed drift exceeds the budget, decays
            # toward 0 when there is slack (projected at 0).
            fm_ema = 0.99 * fm_ema + 0.01 * fm_loss.item()
            lam_auto = min(max(lam_auto + args.lam_lr *
                               (fm_ema / fm0 - 1.0 - args.drift_eps), 0.0),
                           args.lam_max)
        elif args.arm == "pfm_trust":
            dv_step = dv_accum / args.accum  # mean over ALL micro-batches
            dv_ema = dv_step if dv_ema is None else \
                0.99 * dv_ema + 0.01 * dv_step
            lam_v = min(max(lam_v + args.lam_lr * (dv_ema - args.dv_delta),
                            0.0), args.lam_max)
        if track_upr:
            named = dict(model.named_parameters())
            upr = {n: ((named[n] - prev[n]).norm()
                       / prev[n].norm().clamp_min(1e-12)).item() for n in probe_params}
            log_f.write(json.dumps({"step": step, "update_to_param": upr}) + "\n")

        if baw is not None and step % args.refresh_every == 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kp = min(args.probe_batch, z0.shape[0])
                grads, _ = perc.probe_grads(z0_hat[:kp].detach(), z0[:kp])
            norms = {b: g.norm().item() for b, g in grads.items()}
            baw.update(norms)
            w, n_eff = baw.weights()
            # diagnostics (all on the probe subgraph, no extra backward):
            g_bal = sum(w[b] * grads[b] for b in BLOCKS)
            g_fix = sum(grads[b] for b in BLOCKS) / len(BLOCKS)
            cos_bf = F.cosine_similarity(g_bal.flatten(), g_fix.flatten(), dim=0).item()
            # FM gradient at z0_hat is analytic: dL_FM/dz0_hat ∝ (z0_hat - z0)/t^2
            g_fm = ((z0_hat[:kp] - z0[:kp]).float()
                    / (tb[:kp].float() ** 2).clamp_min(1e-4))
            cos_pf = F.cosine_similarity(g_bal.flatten(), g_fm.flatten(), dim=0).item()
            cos_ff = F.cosine_similarity(g_fix.flatten(), g_fm.flatten(), dim=0).item()
            contrib = {str(b): (w[b] * grads[b].norm()).item() / max(g_bal.norm().item(), 1e-12)
                       for b in BLOCKS}
            # per-layer share of the EQUAL-weight gradient — the fixed arm's
            # actual optimization; documents deep-layer domination directly
            tot_norm = max(sum(norms.values()), 1e-12)
            contrib_fixed = {str(b): norms[b] / tot_norm for b in BLOCKS}
            s_z = g_bal.norm().item() / max(g_fix.norm().item(), 1e-12)
            s_theta = None
            if args.arm == "pfm_matched":
                # THETA-space ratio (Amendment 3 gate triggered 2026-08-08:
                # ratio_theta 0.849 vs ratio_z 0.624 at the frozen init — input
                # matching under-scales by ~27% in parameter space). Re-forward
                # a small probe slice through the model WITH grad and measure
                # both parameter-gradient norms; autograd.grad leaves .grad
                # untouched. kp_t kept small: this doubles the graph footprint.
                kp_t = min(8, z0.shape[0])
                mp = [q for q in model.parameters() if q.requires_grad]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v_p = model(zt[:kp_t], t[:kp_t], y[:kp_t])
                    z0h_p = zt[:kp_t] - tb[:kp_t] * v_p
                    bl_p = perc.block_losses(z0h_p, z0[:kp_t])
                    l_fix = torch.stack([bl_p[b] for b in BLOCKS]).mean()
                    l_bal = sum(w[b] * bl_p[b] for b in BLOCKS)
                gt_f = torch.autograd.grad(l_fix, mp, retain_graph=True)
                gt_b = torch.autograd.grad(l_bal, mp)
                tn = lambda gs: torch.sqrt(
                    sum(x.float().pow(2).sum() for x in gs)).item()
                s_theta = tn(gt_b) / max(tn(gt_f), 1e-12)
                del gt_f, gt_b
                match_scale = 0.9 * match_scale + 0.1 * s_theta
            log_f.write(json.dumps({
                "step": step, "grad_norms": norms,
                "weights": {str(b): w[b] for b in BLOCKS}, "n_eff": n_eff,
                "cos_balanced_fixed": cos_bf, "cos_perc_fm": cos_pf,
                "cos_fixed_fm": cos_ff,
                "layer_contrib": contrib, "layer_contrib_fixed": contrib_fixed,
                "probe_t": t[:kp].tolist(),
                "scale_ratio_z": s_z, "scale_ratio_theta": s_theta,
                "match_scale": match_scale, "perc_frac_realized": k / z0.shape[0],
            }) + "\n")

        if step % args.log_every == 0:
            rec = {"step": step, "loss": loss.item(), "fm_loss": fm_loss.item(),
                   "sec_per_step": (time.time() - t0) / (step - start_step + 1)}
            if args.arm == "pfm_auto":
                rec.update({"lam_auto": lam_auto, "fm_ema": fm_ema,
                            "drift": fm_ema / fm0 - 1.0})
            elif args.arm == "pfm_trust":
                rec.update({"lam_v": lam_v, "dv_ema": dv_ema,
                            "dv_by_t": {f"{lo}": (num / den)[
                                (t >= lo) & (t < lo + 0.25)].mean().item()
                                for lo in (0.0, 0.25, 0.5, 0.75)}})
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        # ckpt_last FIRST, milestone second: if the process dies between the two
        # saves, the resumable state is never older than the milestone (a kill
        # mid-ckpt_last after a completed milestone cost us the 10k optimizer
        # state on 2026-08-07).
        if step % args.ckpt_every == 0 and step < args.steps:
            payload = {"model": model.state_dict(), "ema": ema.shadow,
                       "opt": opt.state_dict(), "step": step,
                       "match_scale": match_scale,
                       "lam_auto": lam_auto, "fm0": fm0, "fm_ema": fm_ema,
                       "lam_v": lam_v, "dv_ema": dv_ema,
                       "rng_torch": torch.get_rng_state(),
                       "rng_cuda": torch.cuda.get_rng_state_all(),
                       "rng_loader": g.get_state()}
            if baw is not None:
                payload["baw"] = json.dumps(baw.state())
            torch.save(payload, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

        if args.milestone_every and step % args.milestone_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step},
                       os.path.join(out, f"ckpt_step{step:06d}.pt"))

    torch.save({"model": model.state_dict(), "ema": ema.shadow, "args": vars(args)},
               final_path)
    if os.path.exists(last_path):
        os.remove(last_path)
    log_f.close()
    print(f"[train_latent] done -> {final_path} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
