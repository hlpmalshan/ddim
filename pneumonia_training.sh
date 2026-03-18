#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Train Pneumonia diffusion model (256x256, grayscale, simple loss)

BASE_CONFIG="pneumonia_hq.yml"
EXP_ROOT="ddim_pneumonia"
GPU_ID=0

# Multi-GPU (DDP) toggle and settings
DISTRIBUTED=${DISTRIBUTED:-true}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

run_main() {
  if [ "$DISTRIBUTED" = true ]; then
    if [ -n "$GPUS" ]; then
      export CUDA_VISIBLE_DEVICES="$GPUS"
      if [ "${NPROC_PER_NODE}" -le 0 ]; then
        IFS=',' read -r -a _gpu_array <<< "$GPUS"
        NPROC_PER_NODE=${#_gpu_array[@]}
      fi
    fi
    if [ "${NPROC_PER_NODE}" -le 0 ]; then
      echo "[WARN] NPROC_PER_NODE not set; defaulting to 1"
      NPROC_PER_NODE=1
    fi
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
    export DIST_BACKEND
    torchrun --standalone \
      --master_port="$MASTER_PORT" \
      --nproc_per_node="$NPROC_PER_NODE" \
      main.py --distributed "$@"
  else
    python main.py "$@" --gpu "$GPU_ID"
  fi
}

DOC="ddim_subset_normal_0.0"
run_main \
  --config "$BASE_CONFIG" \
  --exp "$EXP_ROOT" \
  --doc "$DOC" \
  --reg 0.0 \
  --ni

echo "Pneumonia diffusion training completed."
