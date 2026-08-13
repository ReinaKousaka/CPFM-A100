#!/usr/bin/env bash
# One-shot A100 bootstrap. Run from the cpfm/ package root AFTER placing the
# frozen init (see README step 2). Does: venv+deps, sha-verify init, install
# the frozen DINO calibration, 44-shard data fetch (pinned revision, resumable)
# + fp16 memmap conversion, FID reference build, method regression test, and a
# micro-batch scaling probe (16/32/64) that prints measured s/step.
set -e
cd "$(dirname "$0")/.."
LOG=runs/bootstrap.log
mkdir -p runs/latent256 data/latents
echo "[boot $(date +%H:%M)] start" | tee -a "$LOG"

# 0. frozen init must be byte-identical to the registered checkpoint
echo "8743fd1dddaf413d75e6a9cce96707b5b5b1a12b84a21716a4fd67ff9206ae02  runs/latent256/fm_base-s0/ckpt_step200000.pt" \
  | sha256sum -c - | tee -a "$LOG"

# 1. env
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); \
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '|', \
round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GiB')" | tee -a "$LOG"

# 2. frozen calibration (protocol identity — do NOT recalibrate on A100).
#    Hash-checked on BOTH sides: a stale destination must never silently win.
CALIB_SHA=f7a23ef6f563b9b09725aa653d8226e7ce5b3788b35c2a7444bc74a1114b6709
echo "$CALIB_SHA  assets/dino_calib.json" | sha256sum -c - | tee -a "$LOG"
cp -f assets/dino_calib.json runs/latent256/dino_calib.json
echo "$CALIB_SHA  runs/latent256/dino_calib.json" | sha256sum -c - | tee -a "$LOG"

# 3. data: 44 train shards at the pinned dataset revision, then fp16 memmaps
for i in $(seq -w 0 43); do
  F="data/latents/train-000$i-of-00044.parquet"
  [ -f "$F" ] || {
    echo "[boot $(date +%H:%M)] fetching shard $i" | tee -a "$LOG"
    curl -sSL --fail --retry 5 -C - \
      "https://huggingface.co/datasets/Forbu14/imagenet-1k-latent/resolve/64d39472db7f/data/train-000$i-of-00044.parquet" \
      -o "$F.part" && mv "$F.part" "$F"
  }
done
# validate BEFORE conversion: exactly 44 readable shards, registered row total
.venv/bin/python - << 'PYEOF' 2>&1 | tee -a "$LOG"
import glob, sys
import pyarrow.parquet as pq
files = sorted(glob.glob("data/latents/train-*.parquet"))
assert len(files) == 44, f"expected 44 shards, found {len(files)}"
rows = 0
for f in files:
    rows += pq.read_metadata(f).num_rows   # raises on corrupt/partial files
assert rows == 1281167, f"expected 1,281,167 rows, got {rows}"
print(f"shard validation OK: 44 files, {rows} rows")
PYEOF
.venv/bin/python -c "from ba_pfm.latent_data import convert_shards; convert_shards('train')" 2>&1 | tail -2 | tee -a "$LOG"
# post-conversion memmap validation (shape vs meta; loader re-checks every run)
.venv/bin/python -c "from ba_pfm.latent_data import LatentDataset; d=LatentDataset('train'); print('memmap validation OK:', len(d), 'samples')" 2>&1 | tail -1 | tee -a "$LOG"

# 4. FID reference (validation shard decoded through the pinned VAE)
[ -f runs/latent256/.valref_done ] || {
  .venv/bin/python -m ba_pfm.eval_latent --make_ref 2>&1 | tail -2 | tee -a "$LOG"
  touch runs/latent256/.valref_done
}

# 5. method-code regression test (bounded-simplex weights)
.venv/bin/python -m tests.test_baweights | tee -a "$LOG"

# 6. scaling probe: measured s/step at micro-bs 16/32/64; 150 steps so the
#    step-100 diagnostic refresh (the true peak-memory path) is exercised.
#    NOTE (protocol): Phase-1A arms MUST run 16x16 regardless — the probe
#    informs Phase-1B geometry and headroom only. Keep >=15% memory headroom.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for BS in 16 32 64; do
  AC=$((256 / BS))
  echo "[boot $(date +%H:%M)] probe micro-bs=$BS accum=$AC" | tee -a "$LOG"
  rm -rf "runs/latent256/pfm_fixed_probe${BS}-s0"
  .venv/bin/python -m ba_pfm.train_latent --arm pfm_fixed --tag "_probe$BS" \
    --init runs/latent256/fm_base-s0/ckpt_step200000.pt \
    --steps 150 --batch "$BS" --accum "$AC" --perc_frac 0.5 --lr 1e-4 --seed 0 \
    --ckpt_every 1000 --milestone_every 0 --log_every 10 >> "$LOG" 2>&1 \
    && grep '"step": 150' "runs/latent256/pfm_fixed_probe${BS}-s0/log.jsonl" | tail -1 | tee -a "$LOG" \
    || echo "[boot] micro-bs=$BS FAILED (likely OOM)" | tee -a "$LOG"
done
echo "[boot $(date +%H:%M)] BOOTSTRAP COMPLETE — send runs/bootstrap.log back" | tee -a "$LOG"
