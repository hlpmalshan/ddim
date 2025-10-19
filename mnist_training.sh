# To train with specific digits (e.g., 0, 1, 2):
# RESUME=true DISTRIBUTED=true NPROC_PER_NODE=2 GPUS=0,1 SELECTED_DIGITS=0,1,2 bash mnist_training.sh

# If you run without SELECTED_DIGITS:
# RESUME=true DISTRIBUTED=true NPROC_PER_NODE=2 GPUS=0,1 bash mnist_training.sh

#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Train CelebA only (no sampling/evaluation)

# -------- Configurable knobs --------
# Regularization values to sweep
reg_values=(0.0 0.3)

# Base config to use for training (relative to configs/)
BASE_CONFIG="mnist.yml"

# Experiment root (where logs/ and datasets/ live)
EXP_ROOT="ddim_mnist"

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
RESUME=${RESUME:-false}

# Selected digits passed via command line (e.g., "0,1,2")
SELECTED_DIGITS=${SELECTED_DIGITS:-}

# Validate SELECTED_DIGITS
if [ -n "$SELECTED_DIGITS" ]; then
  # Convert comma-separated digits to a sorted, underscore-separated string for folder naming
  IFS=',' read -r -a digits_array <<< "$SELECTED_DIGITS"
  sorted_digits=($(echo "${digits_array[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' '))
  digits_str=$(IFS=_; echo "${sorted_digits[*]}")
  # Update EXP_ROOT to include selected digits
  EXP_ROOT="ddim_mnist_digits_${digits_str}"
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
      main.py --distributed "$@"
  else
    python main.py "$@" --gpu "$GPU_ID"
  fi
}

# Function to create temporary YAML config with updated selected_digits
# make_cfg_with_digits() {
#   local digits="$1" in_rel="$2" out_rel="$3"
#   awk -v digits="[$digits]" '
#     BEGIN { in_data=0 }
#     /^[[:space:]]*data:[[:space:]]*$/ { in_data=1; print; next }
#     /^[^[:space:]]/ { if (in_data) in_data=0 }
#     {
#       if (in_data && $1 ~ /^selected_digits:/) {
#         sub(/selected_digits:[[:space:]]*\[[^]]*\]/, "selected_digits: " digits)
#       }
#       print
#     }
#     END {
#       if (!in_data && digits != "[]") {
#         print "data:"
#         print "  selected_digits: " digits
#       }
#     }
#   ' "$in_rel" > "$out_rel"
# }

# make_cfg_with_digits() {
#   local digits="$1" in_rel="$2" out_rel="$3"
#   cp "$in_rel" "$out_rel"
#   if command -v yq >/dev/null 2>&1; then
#     yq eval ".data.selected_digits = [$digits]" -i "$out_rel"
#   else
#     if grep -q "^  selected_digits:" "$out_rel"; then
#       sed -i "s|^  selected_digits:.*|  selected_digits: [$digits]|" "$out_rel"
#     else
#       sed -i "/^data:/a\  selected_digits: [$digits]" "$out_rel"
#     fi
#   fi
# }

# make_cfg_with_digits() {
#   local digits="$1" in_rel="$2" out_rel="$3"
#   # Copy the input YAML to preserve all fields
#   cp "$in_rel" "$out_rel"
#   # Update selected_digits in-place, ensuring correct indentation
#   sed -i "/^[[:space:]]*data:/,/^[[:space:]]*[a-zA-Z]/ s/^[[:space:]]*selected_digits:.*/  selected_digits: [$digits]/" "$out_rel"
# }

for reg in "${reg_values[@]}"; do
  echo "=== reg=$reg ==="
  DOC="ddim_iso_${reg}"
  REG_LOG_DIR="$LOGS_DIR/$DOC"

  # Create log directories if they don't exist
  mkdir -p "$LOGS_DIR" "$REG_LOG_DIR"

  # # Create temporary YAML config
  # TEMP_CONFIG="temp_mnist_${reg}_${digits_str:-all}.yml"
  # make_cfg_with_digits "$SELECTED_DIGITS" "configs/$BASE_CONFIG" "configs/$TEMP_CONFIG"

  # echo "Base YAML content (configs/$BASE_CONFIG):"
  # cat "configs/$BASE_CONFIG"

  # echo "Temporary YAML content for reg=$reg (configs/$TEMP_CONFIG):"
  # cat "configs/$TEMP_CONFIG"

  # # Check if the temporary YAML was created successfully
  # if [ ! -f "configs/$TEMP_CONFIG" ]; then
  #   echo "Error: Failed to create temporary config 'configs/$TEMP_CONFIG'."
  #   exit 1
  # fi

  echo "[Train] MNIST, reg=$reg, digits=${SELECTED_DIGITS:-all}"
  # If no checkpoints yet, run training; otherwise skip
  # if [ "$RESUME"=true ] ||  ! compgen -G "$REG_LOG_DIR/ckpt_*.pth" > /dev/null; then
  if ! compgen -G "$REG_LOG_DIR/ckpt_*.pth" > /dev/null; then
    run_main \
      --config "$BASE_CONFIG" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --resume_training \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni
  else
    echo "Checkpoints already present in $REG_LOG_DIR and RESUME=false — skipping training."
  fi

  # Clean up temporary YAML file
  # [ -f "configs/$TEMP_CONFIG" ] && rm "configs/$TEMP_CONFIG"
done

echo "MNIST training completed."
