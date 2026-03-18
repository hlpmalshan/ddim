#!/usr/bin/env bash
set -euo pipefail

# Configure GPU selection externally if needed: CUDA_VISIBLE_DEVICES=1 ./run_transient_r_precision.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

CKPT="ddim_cifar10/logs/ddim_iso_0.3/ckpt_508300.pth"
OUTDIR_FULL="Transient_R_Precision/transient_R_precision_cifar10_32_0.3_t0_to_t150"
OUTDIR_SMOKE="Transient_R_Precision/transient_R_precision_cifar10_32_0.3_t0_to_t150_smoke"

SEED="${SEED:-1234}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda:0}"
TIMESTAMPS="${TIMESTAMPS:-auto}"

mkdir -p "${OUTDIR_FULL}"
mkdir -p "${OUTDIR_SMOKE}"

# Smoke test (fast sanity check)
python3 scripts/transient_r_precision_cifar10_ddim.py \
  --ckpt "${CKPT}" \
  --outdir "${OUTDIR_SMOKE}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --timestamps "${TIMESTAMPS}" \
  --num_images_per_t 1000 \
  --resume true

# Full run (50k samples per timestamp; resume-safe)
python3 scripts/transient_r_precision_cifar10_ddim.py \
  --ckpt "${CKPT}" \
  --outdir "${OUTDIR_FULL}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --timestamps "${TIMESTAMPS}" \
  --num_images_per_t 50000 \
  --resume true
