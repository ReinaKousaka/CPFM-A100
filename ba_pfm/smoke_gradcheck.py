"""Pre-A100 smoke test: fixed vs balanced vs matched perceptual gradients on the
SAME batch — norms + pairwise cosines (GPT directive, 2026-08-06).

Composition from one probe_grads call at real model predictions (frozen 200k EMA,
t ~ U(0,1)):
  g_fixed    = mean_b g_b                      (equal weights, fixed PFM)
  g_balanced = sum_b w_b g_b, w = one-shot inverse-norm weights (BAWeights fed
               this batch's norms once — the point-in-time balanced gradient)
  g_matched  = g_fixed * ||g_balanced|| / ||g_fixed||

Implementation invariants that MUST hold (else the matched control has a scale
bug): cos(matched, fixed) = 1, ||g_matched|| = ||g_balanced||. The informative
numbers are cos(balanced, fixed) and the per-block norm spread.
"""

import json
import os

import torch

from .latent_data import LatentDataset
from .perceptual import BAWeights, BLOCKS, LatentPerceptual
from .sit import build_sit
from .vae import FrozenVAE

OUT = "runs/latent256/smoke_gradcheck.json"
INIT = "runs/latent256/fm_base-s0/ckpt_step200000.pt"


def main(device="cuda", batch=16, seed=0, theta=False):
    torch.manual_seed(seed)
    ck = torch.load(INIT, map_location=device, weights_only=True)
    model = build_sit("B").to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    vae = FrozenVAE(device=device)
    perc = LatentPerceptual(vae, device=device,
                            calib_path="runs/latent256/dino_calib.json")
    assert perc.calib, "run calibration first (any perceptual arm does it)"

    ds = LatentDataset("train", max_shards=1, flip=False)
    z0 = torch.stack([ds[i][0] for i in range(batch)]).to(device)
    y = torch.tensor([ds[i][1] for i in range(batch)]).to(device)
    t = torch.rand(batch, device=device)
    tb = t[:, None, None, None]
    zt = (1 - tb) * z0 + tb * torch.randn_like(z0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z0_hat = zt - tb * model(zt, t, y)
    z0_hat = z0_hat.float().detach()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        grads, _ = perc.probe_grads(z0_hat, z0)

    norms = {b: grads[b].norm().item() for b in BLOCKS}
    baw = BAWeights()
    baw.update(norms)
    w, n_eff = baw.weights()

    g_fix = sum(grads[b] for b in BLOCKS) / len(BLOCKS)
    g_bal = sum(w[b] * grads[b] for b in BLOCKS)
    g_mat = g_fix * (g_bal.norm() / g_fix.norm().clamp_min(1e-12))

    cos = lambda a, b: torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0).item()
    rec = {
        "t": t.tolist(),
        "block_grad_norms": {str(b): norms[b] for b in BLOCKS},
        "norm_spread_max_over_min": max(norms.values()) / min(norms.values()),
        "weights": {str(b): w[b] for b in BLOCKS}, "n_eff": n_eff,
        "norm_fixed": g_fix.norm().item(), "norm_balanced": g_bal.norm().item(),
        "norm_matched": g_mat.norm().item(),
        "cos_balanced_fixed": cos(g_bal, g_fix),
        "cos_matched_fixed": cos(g_mat, g_fix),
        "cos_matched_balanced": cos(g_mat, g_bal),
        "invariants_ok": bool(abs(cos(g_mat, g_fix) - 1.0) < 1e-4
                              and abs(g_mat.norm().item() - g_bal.norm().item())
                              / max(g_bal.norm().item(), 1e-12) < 1e-4),
    }
    if theta:
        # Review 2026-08-06 point: matching ||grad wrt z0_hat|| does not imply
        # matching ||grad wrt theta|| (the parameter Jacobian can transform the
        # fixed and balanced directions differently). Measure both parameter-
        # gradient norms on the SAME batch/graph: if ratio_theta ~= ratio_z the
        # input-gradient-matched control is a valid proxy; if not, the control
        # must be redefined in theta space before the A100 arms launch.
        model_params = [p for p in model.parameters() if p.requires_grad]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(zt, t, y)
            z0h = zt - tb * v
            bl = perc.block_losses(z0h, z0)
            l_fix = torch.stack([bl[b] for b in BLOCKS]).mean()
            l_bal = sum(w[b] * bl[b] for b in BLOCKS)
        gt_fix = torch.autograd.grad(l_fix, model_params, retain_graph=True)
        gt_bal = torch.autograd.grad(l_bal, model_params)
        tnorm = lambda gs: torch.sqrt(
            sum(g.float().pow(2).sum() for g in gs)).item()
        rec["theta_norm_fixed"] = tnorm(gt_fix)
        rec["theta_norm_balanced"] = tnorm(gt_bal)
        rec["ratio_theta"] = rec["theta_norm_balanced"] / rec["theta_norm_fixed"]
        rec["ratio_z"] = rec["norm_balanced"] / rec["norm_fixed"]
        rec["theta_cos_balanced_fixed"] = torch.nn.functional.cosine_similarity(
            torch.cat([g.float().flatten() for g in gt_bal]),
            torch.cat([g.float().flatten() for g in gt_fix]), dim=0).item()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))
    assert rec["invariants_ok"], "matched-control scale bug!"


if __name__ == "__main__":
    import sys
    main(theta="--theta" in sys.argv)
