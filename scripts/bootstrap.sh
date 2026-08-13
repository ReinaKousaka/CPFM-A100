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

# 2. frozen calibration (protocol identity — do NOT recalibrate on A100)
cp -n assets/dino_calib.json runs/latent256/dino_calib.json

# 3. data: 44 train shards at the pinned dataset revision, then fp16 memmaps
for i in $(seq -w 0 43); do
  F="data/latents/train-000$i-of-00044.parquet"
  [ -f "$F" ] || {
    echo "[boot $(date +%H:%M)] fetching shard $i" | tee -a "$LOG"
    curl -sL --retry 5 -C - \
      "https://huggingface.co/datasets/Forbu14/imagenet-1k-latent/resolve/64d39472db7f/data/train-000$i-of-00044.parquet" \
      -o "$F.part" && mv "$F.part" "$F"
  }
done
.venv/bin/python -c "from ba_pfm.latent_data import convert_shards; convert_shards('train')" 2>&1 | tail -2 | tee -a "$LOG"

# 4. FID reference (validation shard decoded through the pinned VAE)
[ -f runs/latent256/.valref_done ] || {
  .venv/bin/python -m ba_pfm.eval_latent --make_ref 2>&1 | tail -2 | tee -a "$LOG"
  touch runs/latent256/.valref_done
}

# 5. method-code regression test (bounded-simplex weights)
.venv/bin/python -m tests.test_baweights | tee -a "$LOG"

# 6. scaling probe: measured s/step at micro-bs 16/32/64 (bs64 OOM on 40GB is
#    itself the answer; probe continues)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for BS in 16 32 64; do
  AC=$((256 / BS))
  echo "[boot $(date +%H:%M)] probe micro-bs=$BS accum=$AC" | tee -a "$LOG"
  rm -rf "runs/latent256/pfm_fixed_probe${BS}-s0"
  .venv/bin/python -m ba_pfm.train_latent --arm pfm_fixed --tag "_probe$BS" \
    --init runs/latent256/fm_base-s0/ckpt_step200000.pt \
    --steps 60 --batch "$BS" --accum "$AC" --perc_frac 0.5 --lr 1e-4 --seed 0 \
    --ckpt_every 1000 --milestone_every 0 --log_every 10 >> "$LOG" 2>&1 \
    && grep '"step": 60' "runs/latent256/pfm_fixed_probe${BS}-s0/log.jsonl" | tail -1 | tee -a "$LOG" \
    || echo "[boot] micro-bs=$BS FAILED (likely OOM)" | tee -a "$LOG"
done
echo "[boot $(date +%H:%M)] BOOTSTRAP COMPLETE — send runs/bootstrap.log back" | tee -a "$LOG"
