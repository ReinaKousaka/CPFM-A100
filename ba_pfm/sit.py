"""Compact SiT/DiT-B/2 for latent ImageNet-256 (4x32x32, patch 2, adaLN-Zero).

Faithful to the DiT architecture (Peebles & Xie) as used by SiT: fixed 2D sincos
positional embedding, adaLN-Zero blocks, label embedding with CFG dropout.
Time convention (SiT linear interpolant): x_t = (1-t) z0 + t * eps, target
velocity v = eps - z0, so z0_hat = x_t - t * v_hat.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_2d(embed_dim, grid_size):
    def _1d(dim, pos):
        omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
        omega = 1.0 / 10000 ** omega
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid = np.meshgrid(np.arange(grid_size, dtype=np.float64),
                       np.arange(grid_size, dtype=np.float64))  # (x, y)
    emb = np.concatenate([_1d(embed_dim // 2, grid[1]), _1d(embed_dim // 2, grid[0])], axis=1)
    return torch.from_numpy(emb).float()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden, dropout_prob):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, hidden)  # index num_classes = null
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, y, train):
        if train and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mh = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, mh), nn.GELU(approximate="tanh"),
                                 nn.Linear(mh, hidden))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch * patch * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SiT(nn.Module):
    def __init__(self, input_size=32, patch=2, in_ch=4, hidden=768, depth=12,
                 heads=12, num_classes=1000, label_dropout=0.1):
        super().__init__()
        self.patch = patch
        self.in_ch = in_ch
        self.grid = input_size // patch
        self.x_embed = nn.Conv2d(in_ch, hidden, patch, stride=patch)
        self.register_buffer("pos_embed", sincos_2d(hidden, self.grid), persistent=False)
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(num_classes, hidden, label_dropout)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.final = FinalLayer(hidden, patch, in_ch)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)

    def unpatchify(self, x):
        B, N, _ = x.shape
        p, c, g = self.patch, self.in_ch, self.grid
        x = x.reshape(B, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, c, g * p, g * p)

    def forward(self, z, t, y):
        x = self.x_embed(z).flatten(2).transpose(1, 2) + self.pos_embed[None]
        c = self.t_embed(t) + self.y_embed(y, self.training)
        for blk in self.blocks:
            x = blk(x, c)
        return self.unpatchify(self.final(x, c))


def build_sit(size="B"):
    cfg = {"B": dict(hidden=768, depth=12, heads=12),
           "L": dict(hidden=1024, depth=24, heads=16),
           "XL": dict(hidden=1152, depth=28, heads=16)}[size]
    return SiT(**cfg)
