#!/usr/bin/env bash
# Gate-eval watcher for one arm.
#   RUN=pfm_auto_25k-s1 bash scripts/watch_arm.sh                # full 25k arm
#   RUN=bapfm_l0-s0 LAST_MS=010000 bash scripts/watch_arm.sh    # staged arm
# Few-step FID-10k (NFE 1/2/4/8, CFG 1.0) at each 5k milestone up to LAST_MS;
# NFE-250 CFG {1.0,1.5} references at LAST_MS. Each eval gets 3 attempts; exits
# nonzero listing failures (never a silent "done"). Idempotent.
# GPU guidance: sharing the training GPU is safe ONLY at the 16x16 protocol
# geometry on an 80GB card; otherwise use a separate GPU or run after training.
set -u
cd "$(dirname "$0")/.."
RUN=${RUN:?set RUN (run dir name under runs/latent256)}
LAST_MS=${LAST_MS:-025000}
D="runs/latent256/$RUN"
LOG="runs/$RUN.watch.log"
exec 8> "runs/.$RUN.watch.lock"
flock -n 8 || { echo "watcher already running"; exit 0; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FAILED=""

ev () { local TAG="${1%.pt}_nfe${2}_cfg${3}"
  [ -f "$D/fid_$TAG.json" ] && return 0
  for TRY in 1 2 3; do
    echo "[watch $(date +%H:%M)] $RUN $1 nfe=$2 cfg=$3 (try $TRY)" >> "$LOG"
    .venv/bin/python -m ba_pfm.eval_latent --run "$RUN" --ckpt "$1" --nfe "$2" \
      --cfg "$3" --n 10000 --batch 16 >> "$LOG" 2>&1
    [ -f "$D/fid_$TAG.json" ] && return 0
    sleep 60
  done
  FAILED="$FAILED $TAG"; return 1
}

MS=005000
while [ "$MS" -le "$LAST_MS" ]; do
  until [ -f "$D/ckpt_step$MS.pt" ]; do sleep 300; done
  for NFE in 1 2 4 8; do ev "ckpt_step$MS.pt" "$NFE" 1.0 || true; done
  MS=$(printf '%06d' $((10#$MS + 5000)))
done
ev "ckpt_step$LAST_MS.pt" 250 1.0 || true
ev "ckpt_step$LAST_MS.pt" 250 1.5 || true

if [ -n "$FAILED" ]; then
  echo "[watch $(date +%H:%M)] $RUN FINISHED WITH FAILURES:$FAILED" >> "$LOG"
  echo "FAILED evals:$FAILED"; exit 1
fi
echo "[watch $(date +%H:%M)] $RUN done (through $LAST_MS)" >> "$LOG"
