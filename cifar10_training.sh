#!/bin/bash
set -euo pipefail

# Train CIFAR-10 only (no sampling/evaluation)
# Example: RESUME=true DISTRIBUTED=true GPUS=0,1 bash cifar10_training.sh

# Resolve main.py and configs relative to this script.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# -------- Configurable knobs --------
# Regularization values to sweep
reg_values=(0.0)

# Base config to use for training (relative to configs/)
BASE_CONFIG=${BASE_CONFIG:-cifar10.yml}

# Experiment root (where logs/ and datasets/ live)
EXP_ROOT=${EXP_ROOT:-ddim_cifar10}

# Single-GPU settings (used when DISTRIBUTED=false)
GPU_ID=${GPU_ID:-1}

# Multi-GPU (DDP) toggle and settings
DISTRIBUTED=${DISTRIBUTED:-false}
GPUS=${GPUS:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-0}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

# Training params
TIMESTEPS=${TIMESTEPS:-1000}
ETA=${ETA:-1}
RESUME=${RESUME:-true}

# Per-timestep regularization mode (only supported for iso model) otherwise select batch mode
REG_MODE=${REG_MODE:-per_timestep} 
TIMESTEP_GROUPS=${TIMESTEP_GROUPS:-4}

for flag in DISTRIBUTED RESUME; do
  if [[ "${!flag}" != true && "${!flag}" != false ]]; then
    echo "[ERROR] $flag must be true or false." >&2
    exit 1
  fi
done
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NPROC_PER_NODE must be a nonnegative integer." >&2
  exit 1
fi

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
      main.py --distributed --dist_backend "$DIST_BACKEND" "$@"
  else
    python main.py "$@" --gpu "$GPU_ID"
  fi
}

for reg in "${reg_values[@]}"; do
  echo "=== reg=$reg ==="
  # DOC="ddim_iso_${reg}"
  if [ "$REG_MODE" = "per_timestep" ]; then
  DOC="ddim_iso_${REG_MODE}_g${TIMESTEP_GROUPS}_${reg}"
  else
    DOC="ddim_iso_${reg}"
  fi
  REG_LOG_DIR="$LOGS_DIR/$DOC"
  echo "[Train] CIFAR10, reg=$reg"

  CKPT="$REG_LOG_DIR/ckpt.pth"
  have_ckpt=false
  if [ -f "$CKPT" ]; then
    have_ckpt=true
  fi

  resume_arg=()
  if [ "$RESUME" = true ] && [ "$have_ckpt" = true ]; then
    resume_arg=(--resume_training)
  fi

  # Train if RESUME=true (and ckpt exists) OR no checkpoint yet
  if [ "$RESUME" = true ] || [ "$have_ckpt" = false ]; then
    run_main \
      --config "$BASE_CONFIG" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --reg_mode "$REG_MODE" \
      --timestep_groups "$TIMESTEP_GROUPS" \
      "${resume_arg[@]}" \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni
  else
    echo "Checkpoint exists ($CKPT) and RESUME=false — skipping."
  fi

done

echo "CIFAR10 training script finished."
