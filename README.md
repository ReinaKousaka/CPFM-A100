# cpfm — C-PFM (Constrained Perceptual Flow Matching) portable run package

Self-contained package to reproduce/extend the Phase-1A screen on a fresh GPU
machine (A100 target). Everything needed is here except two large artifacts
(the frozen init checkpoint and the dataset, fetched in steps 2-3).

## Quick start on the A100 machine

```bash
# 1. get the package — this repo only; the A100 machine never needs paper2
git clone git@github.com:ReinaKousaka/CPFM-A100.git && cd CPFM-A100
#    (read-only alternative without SSH keys: https://github.com/ReinaKousaka/CPFM-A100.git)

# 2. receive the frozen init (1.0 GB). Nothing to do on this machine — from
#    the 4090 box someone runs:  bash scripts/push_init.sh <user@this-a100-host>
#    (push_init.sh lives in the private paper2 repo on the 4090 box, not here)
#    which rsyncs the checkpoint into runs/latent256/fm_base-s0/ here and
#    verifies its sha256 (8743fd1d...). Any other transfer works too, as long
#    as the file lands at runs/latent256/fm_base-s0/ckpt_step200000.pt —
#    bootstrap re-verifies the sha and refuses to run on mismatch.

# 3. bootstrap: venv, deps, calibration install, 25 GB data fetch + conversion,
#    FID reference, regression test, micro-batch scaling probe (~1-3 h,
#    network-dominated). Needs ~300 GB free disk and outbound HTTPS.
bash scripts/bootstrap.sh

# 4. read the scaling probe output (last lines of runs/bootstrap.log) for
#    headroom info. Phase-1A arms MUST keep the protocol geometry BATCH=16
#    ACCUM=16 — the C-PFM controller estimator depends on micro-batch size, so
#    changing geometry changes the method (reviewer, 2026-08-13). Larger
#    micro-batches are for Phase-1B only.

# 5. launch arms (one per GPU; set CUDA_VISIBLE_DEVICES per card). Examples:
# C-PFM seed-1 confirmation: EPS must be the SELECTED value (frozen after the
# eps sensitivity run on the 4090 — do not launch before selection is frozen):
CUDA_VISIBLE_DEVICES=0 ARM=pfm_auto TAG=_25k EPS=0.15 SEED=1 \
  nohup bash scripts/run_arm.sh > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 ARM=bapfm TAG=_l0 LAMFM=0.0 STOP_AFTER=10000 \
  nohup bash scripts/run_arm.sh > /dev/null 2>&1 &

# 6. per running arm, start its gate-eval watcher. Sharing the training GPU
#    is OK at the mandatory 16x16 geometry on 80GB cards; otherwise use a
#    separate GPU. Staged arms need LAST_MS (watcher exits at that milestone):
CUDA_VISIBLE_DEVICES=0 RUN=pfm_auto_25k-s1 nohup bash scripts/watch_arm.sh > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 RUN=bapfm_l0-s0 LAST_MS=010000 nohup bash scripts/watch_arm.sh > /dev/null 2>&1 &
```

Progress: `tail runs/latent256/<run>/log.jsonl` (training) and
`ls runs/latent256/<run>/fid_*.json` (gate results as they land).

## Collecting results

Everything small lives under `runs/latent256/<run>/`: `fid_*.json`,
`prdc_*.json`, `dv_*.json`, `grid_*.png`, `log.jsonl`, plus `runs/*.log`.
Send those back (tar or git push if this clone has write access):

```bash
tar czf results_$(hostname)_$(date +%m%d).tgz \
  runs/latent256/*/fid_*.json runs/latent256/*/prdc_*.json \
  runs/latent256/*/dv_*.json runs/latent256/*/grid_*.png \
  runs/latent256/*/log.jsonl runs/latent256/*/args.json runs/*.log
```

Do NOT ship `gen_*` folders (10k PNGs each) unless recall/precision needs
recomputing elsewhere — `ba_pfm.eval_prdc` can compute it on-site instead:
`.venv/bin/python -m ba_pfm.eval_prdc --run <run> --gen gen_ckpt_step025000_nfe4_cfg1.0`

This repo is a submodule of `paper2` (fresh paper2 checkouts:
`git clone --recurse-submodules`).

## Layout

```
ba_pfm/          training/eval code (self-contained python package)
assets/          frozen DINO calibration (protocol identity — never regenerate)
scripts/         bootstrap.sh | run_arm.sh | watch_arm.sh
tests/           BAWeights bounded-simplex regression test
PROTOCOL.md      frozen protocol, arm table, gates, reference numbers
requirements.txt
```

## Ground rules (see PROTOCOL.md for the full registered protocol)

- The init checkpoint and `assets/dino_calib.json` are FROZEN identity — the
  sha-check and `cp -n` in bootstrap enforce this; never regenerate either.
- All arms: optimizer reset at init, lr 1e-4, perc_frac 0.5, global batch 256
  (keep 256 whatever micro-batch geometry the probe picks), 5k milestones.
- `pfm_trust` is launch-blocked (documented label-dropout mismatch) — do not
  run it without the fix.
- eps values are frozen to {0.15, 0.10}; no new sweeps.
- FIDs here are internal-reference numbers — comparable across arms in this
  package only, not to published results.
