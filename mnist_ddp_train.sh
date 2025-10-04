# DISTRIBUTED=true GPUS=0,1 NPROC_PER_NODE=2 ./mnist_train_and_eval_all_ckpts.sh


#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Train MNIST, then for each saved checkpoint:
#  - sample images
#  - evaluate (FID/IS/PRDC)
#  - append results to CSV per checkpoint

# -------- Configurable knobs --------
# Regularization values to sweep
reg_values=(0.9)

# Base config to use for training/sampling (relative to configs/)
BASE_CONFIG="mnist.yml"

# Experiment roots (match your existing layout)
EXP_ROOT="ddim_mnist"      # where logs/ and image_samples/ live
DATA_ROOT="ddim_mnist"     # where datasets/ live for evaluation

# Single-GPU settings (used when DISTRIBUTED=false)
GPU_ID=0

# Multi-GPU (DDP) toggle and settings
DISTRIBUTED=${DISTRIBUTED:-false}
GPUS=${GPUS:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-0}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

# Training params (mirrors your other script)
TIMESTEPS=1000
ETA=1

# MNIST paths for evaluation
IMAGE_PATH="$DATA_ROOT/datasets/mnist/MNIST/raw/train-images-idx3-ubyte"
LABEL_PATH="$DATA_ROOT/datasets/mnist/MNIST/raw/train-labels-idx1-ubyte"

# Derived dirs
LOGS_DIR="$EXP_ROOT/logs"
GEN_BASE="$EXP_ROOT/image_samples"

# Create helper to write a temp config under configs/ with sampling.ckpt_id set
make_cfg_with_ckpt() {
  local step="$1" in_rel="$2" out_rel="$3"
  # All paths are relative to repo root, but we write under configs/
  awk -v step="$step" '
    BEGIN{in_sampling=0}
    /^[[:space:]]*sampling:[[:space:]]*$/ {in_sampling=1; print; next}
    /^[^[:space:]]/ { if (in_sampling) in_sampling=0 }
    {
      if (in_sampling && $1 ~ /^ckpt_id:/) {
        sub(/ckpt_id:[[:space:]]*[0-9]+/, "ckpt_id: " step)
      }
      print
    }
  ' "configs/${in_rel}" > "configs/${out_rel}"
}

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

# Parse numeric fields from evaluation text output
parse_eval_to_csv_row() {
  local txt="$1"
  local reg="$2"
  local step="$3"

  local fid is_mean is_std precision recall density coverage
  fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
  # Inception Score: <mean> ± <std>
  local is_line
  is_line=$(awk -F": " '/^Inception Score:/ {print $2}' "$txt" | head -n1 || true)
  is_mean=$(awk '{print $1}' <<<"$is_line" || true)
  is_std=$(awk '{print $3}'  <<<"$is_line" || true)
  precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
  recall=$(awk -F": "    '/^Recall:/ {print $2}' "$txt"    | head -n1 || true)
  density=$(awk -F": "   '/^Density:/ {print $2}' "$txt"   | head -n1 || true)
  coverage=$(awk -F": "  '/^Coverage:/ {print $2}' "$txt"  | head -n1 || true)

  # Default to NA if empty
  fid=${fid:-NA}; is_mean=${is_mean:-NA}; is_std=${is_std:-NA}
  precision=${precision:-NA}; recall=${recall:-NA}; density=${density:-NA}; coverage=${coverage:-NA}

  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$reg" "$step" "$fid" "$is_mean" "$is_std" "$precision" "$recall" "$density" "$coverage"
}

for reg in "${reg_values[@]}"; do
  echo "=== reg=$reg ==="
  DOC="ddim_iso_${reg}"
  REG_LOG_DIR="$LOGS_DIR/$DOC"
  REG_GEN_DIR="$GEN_BASE"

  echo "[Train] reg=$reg"
  # If no checkpoints yet, run training; otherwise skip
  if ! compgen -G "$REG_LOG_DIR/ckpt_*.pth" > /dev/null; then
    run_main \
      --config "$BASE_CONFIG" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni
  else
    echo "Checkpoints already present in $REG_LOG_DIR — skipping training."
  fi

  # Discover all available checkpoints
  mapfile -t ckpts < <(ls -1 "$REG_LOG_DIR"/ckpt_*.pth 2>/dev/null | sort -V)
  if [ ${#ckpts[@]} -eq 0 ]; then
    echo "No checkpoints found in $REG_LOG_DIR — nothing to evaluate."
    continue
  fi

  CSV_OUT="$REG_LOG_DIR/eval_summary.csv"
  if [ ! -f "$CSV_OUT" ]; then
    echo "reg,step,fid,is_mean,is_std,precision,recall,density,coverage" > "$CSV_OUT"
  fi

  for ckpt_path in "${ckpts[@]}"; do
    step=$(basename "$ckpt_path" | sed -E 's/^ckpt_([0-9]+)\.pth$/\1/')
    echo "--- Sampling & Evaluating: reg=$reg, step=$step ---"

    # Unique image output folder per checkpoint (under EXP_ROOT/image_samples)
    IDIR="ddim_iso_${reg}_s${step}"
    GEN_DIR="$REG_GEN_DIR/$IDIR"
    mkdir -p "$GEN_DIR"

    # Create a temporary config with the desired ckpt id inside configs/
    TMP_CFG_NAME="_tmp_mnist_ckpt_${step}.yml"
    make_cfg_with_ckpt "$step" "$BASE_CONFIG" "$TMP_CFG_NAME"

    # Run sampling for this checkpoint id
    run_main \
      --config "$TMP_CFG_NAME" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --sample --fid \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni \
      -i "$IDIR"

    # Evaluate and tee to a per-ckpt txt, then append a CSV row
    EVAL_TXT="$REG_LOG_DIR/evals/eval_ckpt_${step}.txt"
    mkdir -p "$(dirname "$EVAL_TXT")"
    python evaluation_mnist.py \
      --image_path "$IMAGE_PATH" \
      --label_path "$LABEL_PATH" \
      --gen_dir "$GEN_DIR" | tee "$EVAL_TXT"

    parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"

    # Clean temp config
    rm -f "configs/$TMP_CFG_NAME"
  done

  echo "Saved CSV: $CSV_OUT"
  echo "----------------------------------------"
done

echo "All training, sampling, and evaluations completed."
