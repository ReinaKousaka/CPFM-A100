"""Frozen SD-VAE decoder for the training-loop perceptual path.

Contract (all of Phase-1): stabilityai/sd-vae-ft-ema, z_model = 0.18215 * E(x).
decode_model() maps model-space latents to images in ~[-1,1]; gradients flow to
the input latent, never to VAE parameters. Optional activation checkpointing
halves decoder activation memory at ~1.3x decode compute.
"""

import torch
import torch.nn as nn

SCALE = 0.18215


class FrozenVAE(nn.Module):
    def __init__(self, model_id="stabilityai/sd-vae-ft-ema", device="cuda",
                 use_checkpoint=True):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.to(device).eval().requires_grad_(False)
        self.model_id = model_id
        if use_checkpoint:
            # segmented (per-block) checkpointing caps backward peak memory; a
            # whole-decoder checkpoint wrapper does NOT (it re-materializes the
            # full graph on backward). diffusers applies it only in train mode;
            # the decoder is GroupNorm/SiLU-only (no BN/dropout), so train-mode
            # with frozen params is numerically identical to eval.
            self.vae.enable_gradient_checkpointing()
            self.vae.decoder.train()

    def _decode(self, z_raw):
        return self.vae.decode(z_raw).sample

    def decode_model(self, z_model):
        """model-space latent -> image ~[-1,1]; differentiable w.r.t. z_model."""
        return self._decode(z_model / SCALE)

    @torch.no_grad()
    def decode_raw(self, z_raw):
        return self._decode(z_raw)
