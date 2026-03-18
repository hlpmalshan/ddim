#!/usr/bin/env bash
set -euo pipefail

PY=NLL_check.py

# choose GPUs (example: 6 GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,4"}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-123}
CLIP_DENOISED=${CLIP_DENOISED:---clip_denoised}

IMAGE_DIR="ddim_cifar100/datasets/cifar100/train"

CONFIG="configs/cifar100.yml"
DDPM_CKPT="ddim_cifar100/logs/ddim_iso_0.0/ckpt_406640.pth"
ISO_CKPT="ddim_cifar100/logs/ddim_iso_0.3/ckpt_406640.pth"

OUT_ROOT="bpd_eval_results/cifar100"

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


#cifar100 ddpm bpd : 3.7467
#cifar100 ddpm bpd : 3.7484
