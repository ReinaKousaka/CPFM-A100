# Frozen Phase-1A protocol (condensed; authoritative version: paper2/PHASE1A_PLAN.md)

## Identity (verify, never regenerate)
- Init: `runs/latent256/fm_base-s0/ckpt_step200000.pt`,
  sha256 `8743fd1dddaf413d75e6a9cce96707b5b5b1a12b84a21716a4fd67ff9206ae02`
  (custom DiT-B/2 rectified flow, 200k steps, batch 256, seed 0).
- Data: HF `Forbu14/imagenet-1k-latent` rev `64d39472db7f` (RAW latents; code
  applies x0.18215). VAE: `stabilityai/sd-vae-ft-ema` rev `f04b2c4b9831`.
- DINO calibration: `assets/dino_calib.json` — install as
  `runs/latent256/dino_calib.json` (bootstrap does this). DO NOT recalibrate.

## Arm convention (identical for every arm)
Init from milestone EMA weights; optimizer RESET (fresh AdamW lr 1e-4 wd 0, no
warmup); data-order seed = arm seed; global batch 256 as **micro-batch 16 x
accum 16 — geometry is FROZEN protocol identity** (the C-PFM controller
estimator and stochastic pairing depend on it; larger micro-batches only in
Phase-1B, where the controller will be revised to accumulation-averaged FM
loss and re-validated); perc_frac 0.5; milestones every 5k + resumable ckpt
every CKPT_EVERY; eval on EMA weights.

## Registered arms
| arm (code)   | meaning                                   | flags               |
|--------------|-------------------------------------------|---------------------|
| fm           | continued-FM control                      | —                   |
| pfm_fixed    | plain PFM baseline (equal weights, no FM) | —                   |
| pfm_auto     | **C-PFM (method)** dual-ascent lambda     | EPS=0.15 (or 0.10)  |
| bapfm        | gradient-balanced + static anchor         | LAMFM=0.0 (balanced)|
| pfm_matched  | input-grad-matched control (theta probe)  | STEPS=10000         |
| pfm_trust    | velocity trust region — **LAUNCH BLOCKED**| (label-dropout bug) |

## A100 priority (Amendment 7)
1. C-PFM seed-1 confirmation (selected eps).  2. balanced `bapfm LAMFM=0.0`
staged to 10k (stop if NFE{2,4,8} all clearly worse than fixed AND none improved
from 5k; else 15k; 25k only if competitive).  3. theta-matched to 10k.
4. One balanced+C-PFM composition ONLY if balanced shows signal.  5. BA-B last.
eps sweep is FROZEN at {inf, 0.15, 0.10} — no new eps values.

## Gates (Amendment 4)
Primary: matched-budget comparison at 5k/10k/15k/20k/25k, all arms.
Secondary: best-achieved + its location. Mechanism: K*(s) migration + peak
width. Pass rule vs fixed PFM: win at >=1 matched budget at usable NFE.

## Reference numbers (4090, seed 0, FID-10k CFG 1.0 vs decoded-val ref)
fixed  NFE4: 66.1 / **41.7** / 43.8 / 47.7 / 49.7   (5k/10k/15k/20k/25k)
fixed  NFE8: 53.9 / 55.4 / 57.6 / 60.0 / 61.6
C-PFM(0.15) NFE4: 73.0 / 49.1 / 44.6 / 46.0 / 49.3
C-PFM(0.15) NFE8: 55.8 / 46.6 / 44.9 / 45.2 / 46.8
cont-FM: flat at all few-step budgets. NFE-250@25k: fixed 74.6, C-PFM 71.6,
init 64.9. NOTE: these FIDs are internal (10k samples, val-decoded reference)
— comparable across arms here, NOT to published numbers.

## Cost anchors
4090 (bs16x16): 5.93 s/step -> 25k arm ~= 41 h + ~3 h evals. A100 runs KEEP
bs16x16 (frozen geometry); expect similar-or-better s/step. The bootstrap
probe's larger geometries are Phase-1B planning info only.
