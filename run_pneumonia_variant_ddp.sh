#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example: train on 4 GPUs. Adjust CUDA_VISIBLE_DEVICES and --normal_variant as needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Use a non-default port to avoid collisions with other torchrun jobs.
export MASTER_PORT="${MASTER_PORT:-29506}"

torchrun --nproc_per_node=8 --master_port 29506 "${repo_dir}/pneumonia_variant_train_ddp.py" \
  --train_dir "${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/train" \
  --val_dir "${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/val" \
  --test_dir "${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/test" \
  --normal_variant "NORMAL_ISO_100" \
  --model "resnet50" \
  --epochs 100 \
  --batch_size 32 \
  --train_mode "head" \
  --output_dir "${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/outputs_iso_100"
