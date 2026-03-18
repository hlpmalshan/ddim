#!/usr/bin/env bash
set -euo pipefail

PY=NLL_check.py

# choose GPUs (example: 6 GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5"}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-12}
SEED=${SEED:-123}
CLIP_DENOISED=${CLIP_DENOISED:---clip_denoised}

IMAGE_DIR="ddim_celeba/datasets/celeba/celeba/prepocessed_imgs"

CONFIG="configs/celeba.yml"
DDPM_CKPT="ddim_celeba/logs/ddim_iso_0.0/ckpt_500000.pth"
ISO_CKPT="ddim_celeba/logs/ddim_iso_0.3/ckpt_500000.pth"

OUT_ROOT="./bpd_eval_results/celeba"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${PY}" \
  --config "${CONFIG}" \
  --ckpt "${DDPM_CKPT}" \
  --image_dir "${IMAGE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${CLIP_DENOISED} \
  --out_dir "${OUT_ROOT}/ddpm"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${PY}" \
  --config "${CONFIG}" \
  --ckpt "${ISO_CKPT}" \
  --image_dir "${IMAGE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${CLIP_DENOISED} \
  --out_dir "${OUT_ROOT}/iso"


# celeba_ddpm_bpd : 2.7184
# celeba_iso_bpd : 2.7255