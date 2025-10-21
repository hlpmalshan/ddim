#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Train CelebA only (no sampling/evaluation)

# -------- Configurable knobs --------
# Regularization values to sweep
reg_values=(0.0 0.3)

# Base config to use for training (relative to configs/)
BASE_CONFIG="celeba.yml"

# Experiment root (where logs/ and datasets/ live)
EXP_ROOT="ddim_celeba"

# Single-GPU settings (used when DISTRIBUTED=false)
GPU_ID=0

# Multi-GPU (DDP) toggle and settings
DISTRIBUTED=${DISTRIBUTED:-false}
GPUS=${GPUS:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-0}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

# Training params
TIMESTEPS=${TIMESTEPS:-1000}
ETA=${ETA:-1}

# Derived dirs
LOGS_DIR="$EXP_ROOT/logs"

# Launcher helper (switches between single GPU and DDP)
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

for reg in "${reg_values[@]}"; do
  echo "=== reg=$reg ==="
  DOC="ddim_iso_${reg}"
  REG_LOG_DIR="$LOGS_DIR/$DOC"

  echo "[Train] CelebA, reg=$reg"
    run_main \
      --config "$BASE_CONFIG" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --resume_training \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni
done

echo "CelebA training completed."

