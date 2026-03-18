#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Sample from the pneumonia diffusion model

BASE_CONFIG="pneumonia_hq.yml"
EXP_ROOT="ddim_pneumonia"
DOC="ddim_pneumonia_0.3"

# Sampling params
TIMESTEPS=${TIMESTEPS:-1000}
ETA=${ETA:-0.0}
IMAGE_FOLDER=${IMAGE_FOLDER:-"pneumonia_samples_0.3"}

# Single-GPU / DDP
GPU_ID=${GPU_ID:-0}
DISTRIBUTED=${DISTRIBUTED:-true}
GPUS=${GPUS:-6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
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
    [ "${NPROC_PER_NODE}" -le 0 ] && NPROC_PER_NODE=1
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

run_main \
  --config "$BASE_CONFIG" \
  --exp "$EXP_ROOT" \
  --doc "$DOC" \
  --reg 0.3 \
  --sample \
  --timesteps "$TIMESTEPS" \
  --eta "$ETA" \
  --ni \
  --fid \
  -i "$IMAGE_FOLDER"

echo "Pneumonia sampling completed."
