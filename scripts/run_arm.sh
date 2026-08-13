#!/usr/bin/env bash
# Parameterized arm launcher. Usage examples (from the cpfm/ package root):
#   ARM=fm        TAG=_cont25k                     bash scripts/run_arm.sh
#   ARM=pfm_fixed TAG=_25k                         bash scripts/run_arm.sh
#   ARM=pfm_auto  TAG=_25k    EPS=0.15             bash scripts/run_arm.sh   # C-PFM
#   ARM=bapfm     TAG=_l0     LAMFM=0.0            bash scripts/run_arm.sh   # balanced
#   ARM=pfm_matched TAG=_10k  STEPS=10000          bash scripts/run_arm.sh
#   SEED=1 ... for replications. BATCH=16 ACCUM=16 is FROZEN protocol geometry
#   for all Phase-1A arms (do not change; the probe informs Phase-1B only).
#   Staged arms: STEPS=25000 STOP_AFTER=10000 pauses cleanly at 10k keeping the
#   resumable checkpoint; relaunch with a higher STOP_AFTER (or none) extends.
# flock-guarded, sha-verified init, resumable (ckpt_last), 5 retries.
set -u
cd "$(dirname "$0")/.."
ARM=${ARM:?set ARM}; TAG=${TAG:?set TAG}; SEED=${SEED:-0}
STEPS=${STEPS:-25000}; BATCH=${BATCH:-16}; ACCUM=${ACCUM:-16}
CKPT_EVERY=${CKPT_EVERY:-5000}   # resumable-ckpt interval; use 500-1000 on preemptible slots
STOP_AFTER=${STOP_AFTER:-0}      # staged pause step (0 = run to STEPS)
EPS=${EPS:-}; LAMFM=${LAMFM:-}
NAME="$ARM$TAG-s$SEED"
LOG="runs/$NAME.log"
INIT=runs/latent256/fm_base-s0/ckpt_step200000.pt
exec 9> "runs/.$NAME.lock"
flock -n 9 || { echo "$NAME already running"; exit 0; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "8743fd1dddaf413d75e6a9cce96707b5b5b1a12b84a21716a4fd67ff9206ae02  $INIT" \
  | sha256sum -c - >> "$LOG" 2>&1 || { echo "[$NAME] INIT SHA MISMATCH" >> "$LOG"; exit 1; }

EXTRA=""
[ -n "$EPS" ] && EXTRA="$EXTRA --drift_eps $EPS"
[ -n "$LAMFM" ] && EXTRA="$EXTRA --lam_fm $LAMFM"
[ "$STOP_AFTER" != "0" ] && EXTRA="$EXTRA --stop_after $STOP_AFTER"

for ATTEMPT in 1 2 3 4 5; do
  [ -f "runs/.$NAME.paused" ] && { echo "[$NAME] paused by gate decision" >> "$LOG"; exit 0; }
  echo "[$NAME $(date +%H:%M)] attempt $ATTEMPT" >> "$LOG"
  if .venv/bin/python -m ba_pfm.train_latent --arm "$ARM" --tag "$TAG" --init "$INIT" \
       --steps "$STEPS" --batch "$BATCH" --accum "$ACCUM" --perc_frac 0.5 \
       --lr 1e-4 --seed "$SEED" --milestone_every 5000 --ckpt_every "$CKPT_EVERY" \
       $EXTRA >> "$LOG" 2>&1; then
    echo "[$NAME $(date +%H:%M)] COMPLETED" >> "$LOG"; exit 0
  fi
  echo "[$NAME $(date +%H:%M)] attempt $ATTEMPT failed; retry in 60s" >> "$LOG"
  sleep 60
done
echo "[$NAME $(date +%H:%M)] giving up after 5 attempts" >> "$LOG"; exit 1
