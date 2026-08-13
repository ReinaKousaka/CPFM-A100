"""BA-PFM perceptual supervision in latent space.

Pipeline: z0_hat --FrozenVAE.decode_model--> image --resize 224--> DINOv2-S blocks.
Per-block losses are value-calibrated (divided by mean block distance between
random decoded real pairs, cached per dataset). On top of calibration, BA-PFM
applies inverse EMA-gradient-norm weights with a c=5 ratio clip and stop-grad
(PHASE1A_PLAN.md). fixed-PFM mode = equal calibrated weights (no balancing).
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = [0, 1, 2, 4, 5, 6, 9, 10, 11]

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoBlocks(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.net.eval().requires_grad_(False).to(device)
        self.mean = _IMNET_MEAN.to(device)
        self.std = _IMNET_STD.to(device)

    def _prep(self, img):
        img = (img.clamp(-1, 1) + 1) / 2
        img = F.interpolate(img, size=(224, 224), mode="bicubic",
                            align_corners=False, antialias=True)
        return (img - self.mean) / self.std

    def features(self, img, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return self.net.get_intermediate_layers(self._prep(img), n=BLOCKS, norm=True)


class BAWeights:
    """EMA of per-block gradient norms -> stop-grad inverse weights, ratio-clipped."""

    def __init__(self, blocks=BLOCKS, c=5.0, beta=0.99, eps=1e-8):
        self.blocks = blocks
        self.c, self.beta, self.eps = c, beta, eps
        self.g_ema = {b: None for b in blocks}

    def update(self, grad_norms):
        for b, g in grad_norms.items():
            self.g_ema[b] = g if self.g_ema[b] is None else \
                self.beta * self.g_ema[b] + (1 - self.beta) * g

    def weights(self):
        L = len(self.blocks)
        if any(self.g_ema[b] is None for b in self.blocks):
            w = {b: 1.0 / L for b in self.blocks}   # uniform until first update
            return w, float(L)
        inv = {b: 1.0 / (self.g_ema[b] + self.eps) for b in self.blocks}
        lo, hi = 1.0 / (self.c * L), self.c / L
        # exact bounded-simplex weights: w_b = clip(inv_b * tau, lo, hi) with tau
        # bisected so sum(w) = 1 (monotone + continuous, hence exact; feasible
        # since lo*L <= 1 <= hi*L). Replaces clip-then-renormalize, which can
        # leave the declared [lo, hi] interval — 2026-08-06 review found a
        # renormalized weight at 0.758 > hi = c/L = 0.556.
        t_lo = lo / max(inv.values())   # everything at/below lo -> sum < 1
        t_hi = hi / min(inv.values())   # everything at/above hi -> sum > 1
        for _ in range(100):
            tau = 0.5 * (t_lo + t_hi)
            s = sum(min(max(inv[b] * tau, lo), hi) for b in self.blocks)
            if s < 1.0:
                t_lo = tau
            else:
                t_hi = tau
        w = {b: min(max(inv[b] * tau, lo), hi) for b in self.blocks}
        s = sum(w.values())             # exact to ~1e-14 after bisection
        w = {b: v / s for b, v in w.items()}
        n_eff = 1.0 / sum(v * v for v in w.values())
        return w, n_eff

    def state(self):
        return {"g_ema": self.g_ema}

    def load(self, st):
        self.g_ema = {int(k): v for k, v in st["g_ema"].items()}


class LatentPerceptual(nn.Module):
    def __init__(self, vae, device="cuda", calib_path=None):
        super().__init__()
        self.vae = vae
        self.dino = DinoBlocks(device)
        self.device = device
        self.calib = {}
        if calib_path and os.path.exists(calib_path):
            with open(calib_path) as f:
                self.calib = {int(k): v for k, v in json.load(f).items()}

    def block_losses(self, z0_hat, z0_target):
        """dict block -> scalar calibrated MSE. Single DINO forward per side."""
        img_pred = self.vae.decode_model(z0_hat)
        with torch.no_grad():
            img_tgt = self.vae.decode_model(z0_target)
        fp = self.dino.features(img_pred, grad=torch.is_grad_enabled())
        ft = self.dino.features(img_tgt, grad=False)
        out = {}
        for b, a, t in zip(BLOCKS, fp, ft):
            d = ((a - t.detach()) ** 2).mean()
            out[b] = d / self.calib[b] if self.calib else d
        return out

    def loss(self, z0_hat, z0_target, weights=None):
        bl = self.block_losses(z0_hat, z0_target)
        if weights is None:                       # fixed PFM: equal weights
            return torch.stack(list(bl.values())).mean(), bl
        total = sum(weights[b] * bl[b] for b in BLOCKS)
        return total, bl

    def probe_grads(self, z0_hat_detached, z0_target):
        """Per-block grad VECTORS wrt z0_hat from ONE decode+DINO graph via
        autograd.grad (no parameter .grad accumulation)."""
        z = z0_hat_detached.detach().requires_grad_(True)
        bl = self.block_losses(z, z0_target)
        grads = {}
        for i, b in enumerate(BLOCKS):
            (g,) = torch.autograd.grad(bl[b], z, retain_graph=i < len(BLOCKS) - 1)
            grads[b] = g.detach().float()
        return grads, {b: v.item() for b, v in bl.items()}

    def probe_grad_norms(self, z0_hat_detached, z0_target):
        grads, bl = self.probe_grads(z0_hat_detached, z0_target)
        return {b: g.norm().item() for b, g in grads.items()}, bl


def calibrate(perc, dataset, path, n_pairs=256, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[: 2 * n_pairs]
    sums = {b: 0.0 for b in BLOCKS}
    n = 0
    with torch.no_grad():
        for i in range(0, 2 * n_pairs, 2 * batch):
            ia = idx[i : i + batch].tolist()
            ib = idx[i + batch : i + 2 * batch].tolist()
            m = min(len(ia), len(ib))
            if m == 0:
                break
            za = torch.stack([dataset[j][0] for j in ia[:m]]).to(perc.device)
            zb = torch.stack([dataset[j][0] for j in ib[:m]]).to(perc.device)
            ia_img = perc.vae.decode_model(za)
            ib_img = perc.vae.decode_model(zb)
            fa = perc.dino.features(ia_img, grad=False)
            fb = perc.dino.features(ib_img, grad=False)
            for b, x, y in zip(BLOCKS, fa, fb):
                sums[b] += ((x - y) ** 2).mean().item() * m
            n += m
    calib = {b: sums[b] / n for b in BLOCKS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    perc.calib = calib
    return calib
