#!/usr/bin/env bash
set -euo pipefail

PY=NLL_check.py

# choose GPUs (example: 6 GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6,7"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-12}
SEED=${SEED:-123}
CLIP_DENOISED=${CLIP_DENOISED:---clip_denoised}

IMAGE_DIR="ddim_celebahq/datasets/celebahq256/celebahq256_imgs/train"

CONFIG="configs/celebahq.yml"
DDPM_CKPT="ddim_celebahq/logs/celebahq_iso_0.0/ckpt_500000.pth"
ISO_CKPT="ddim_celebahq/logs/celebahq_iso_0.3/ckpt_500000.pth"

OUT_ROOT="./bpd_eval_results/celebahq"

# torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${PY}" \
#   --config "${CONFIG}" \
#   --ckpt "${DDPM_CKPT}" \
#   --image_dir "${IMAGE_DIR}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --seed "${SEED}" \
#   ${CLIP_DENOISED} \
#   --out_dir "${OUT_ROOT}/ddpm"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${PY}" \
  --config "${CONFIG}" \
  --ckpt "${ISO_CKPT}" \
  --image_dir "${IMAGE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${CLIP_DENOISED} \
  --out_dir "${OUT_ROOT}/iso"


# celebahq_ddpm_bpd : 2.7184
# celebahq_iso_bpd : 2.5713