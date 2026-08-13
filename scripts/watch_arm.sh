#!/usr/bin/env bash
# Gate-eval watcher for one arm. Usage:  RUN=pfm_auto_25k-s0 bash scripts/watch_arm.sh
# Few-step FID-10k (NFE 1/2/4/8, CFG 1.0) at every 5k milestone; NFE-250
# CFG {1.0,1.5} references at 25k. Idempotent (skips existing fid jsons).
# No git operations — collect results per README step 6.
set -u
cd "$(dirname "$0")/.."
RUN=${RUN:?set RUN (run dir name under runs/latent256)}
D="runs/latent256/$RUN"
LOG="runs/$RUN.watch.log"
exec 8> "runs/.$RUN.watch.lock"
flock -n 8 || { echo "watcher already running"; exit 0; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ev () { local TAG="${1%.pt}_nfe${2}_cfg${3}"
  [ -f "$D/fid_$TAG.json" ] && return 0
  echo "[watch $(date +%H:%M)] $RUN $1 nfe=$2 cfg=$3" >> "$LOG"
  .venv/bin/python -m ba_pfm.eval_latent --run "$RUN" --ckpt "$1" --nfe "$2" \
    --cfg "$3" --n 10000 --batch 16 >> "$LOG" 2>&1
}
for MS in 005000 010000 015000 020000 025000; do
  until [ -f "$D/ckpt_step$MS.pt" ]; do sleep 300; done
  for NFE in 1 2 4 8; do ev "ckpt_step$MS.pt" "$NFE" 1.0; done
done
ev ckpt_step025000.pt 250 1.0
ev ckpt_step025000.pt 250 1.5
echo "[watch $(date +%H:%M)] $RUN done" >> "$LOG"
