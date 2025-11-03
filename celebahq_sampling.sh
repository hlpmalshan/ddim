#!/bin/bash
set -euo pipefail
shopt -s nullglob

# ---------- Config ----------
reg_values=(0.3)
CKPT_STEPS=(500000)  # leave empty () to auto-discover

BASE_CONFIG="celebahq.yml"   # under configs/

# Match your path pattern:
EXP_ROOT="ddim_celebahq"
DATA_ROOT="$EXP_ROOT"
REAL_DIR="$DATA_ROOT/datasets/celebahq/celebahq256/celebahq256_imgs"

LOGS_DIR="$EXP_ROOT/logs"
GEN_BASE="$EXP_ROOT/image_samples"
EVAL_SCRIPT="celebahq_evaluation.py"   # prints FID + PRDC

# Sampling params
TIMESTEPS=${TIMESTEPS:-1000}
ETA=${ETA:-1}

# DDP / single-GPU
GPU_ID=${GPU_ID:-0}
DISTRIBUTED=${DISTRIBUTED:-false}
GPUS=${GPUS:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-0}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

# Eval params
EVAL_RES=${EVAL_RES:-256}
MAX_IMAGES=${MAX_IMAGES:-27000}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-8}
NEAREST_K=${NEAREST_K:-5}
EVAL_CPU=${EVAL_CPU:-false}
EVAL_SEED=${EVAL_SEED:-123}

# ---------- Helpers ----------
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

make_cfg_with_ckpt() {
  local step="$1" in_rel="$2" out_rel="$3"
  awk -v step="$step" '
    BEGIN{in_sampling=0}
    /^[[:space:]]*sampling:[[:space:]]*$/ {in_sampling=1; print; next}
    /^[^[:space:]]/ { if (in_sampling) in_sampling=0 }
    {
      if (in_sampling && $1 ~ /^ckpt_id:/) sub(/ckpt_id:[[:space:]]*[0-9]+/, "ckpt_id: " step)
      print
    }
  ' "configs/${in_rel}" > "configs/${out_rel}"
}

discover_steps_for_reg() {
  local reg="$1"
  local dir="$LOGS_DIR/celebahq_iso_${reg}"
  ls -1 "$dir"/ckpt_*.pth 2>/dev/null \
    | sed -E 's@.*/ckpt_([0-9]+)\.pth@\1@' \
    | sort -V || true
}

ckpt_path_for() {
  local reg="$1" step="$2"
  echo "$LOGS_DIR/celebahq_iso_${reg}/ckpt_${step}.pth"
}

parse_eval_to_csv_row() {
  local txt="$1" reg="$2" step="$3"
  local fid precision recall density coverage
  fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
  precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
  recall=$(awk -F": "    '/^Recall:/ {print $2}'    "$txt" | head -n1 || true)
  density=$(awk -F": "   '/^Density:/ {print $2}'   "$txt" | head -n1 || true)
  coverage=$(awk -F": "  '/^Coverage:/ {print $2}'  "$txt" | head -n1 || true)
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "${reg}" "${step}" "${fid:-NA}" "${precision:-NA}" "${recall:-NA}" "${density:-NA}" "${coverage:-NA}"
}

# ---------- Build checkpoint-first order ----------
declare -a STEPS
if [ ${#CKPT_STEPS[@]} -gt 0 ]; then
  STEPS=("${CKPT_STEPS[@]}")
else
  tmp_steps=()
  for reg in "${reg_values[@]}"; do
    while IFS= read -r s; do tmp_steps+=("$s"); done < <(discover_steps_for_reg "$reg")
  done
  mapfile -t STEPS < <(printf "%s\n" "${tmp_steps[@]}" | sort -V | awk '!seen[$0]++')
fi

[ ${#STEPS[@]} -gt 0 ] || { echo "[INFO] No checkpoints to process."; exit 0; }

# ---------- Main (by step, then by reg) ----------
for step in "${STEPS[@]}"; do
  echo "=== STEP $step ==="
  for reg in "${reg_values[@]}"; do
    REG_LOG_DIR="$LOGS_DIR/celebahq_iso_${reg}"
    REG_GEN_DIR="$GEN_BASE"
    CKPT_PATH="$(ckpt_path_for "$reg" "$step")"

    if [ ! -f "$CKPT_PATH" ]; then
      echo "[SKIP] reg=$reg step=$step (missing $CKPT_PATH)"
      continue
    fi

    echo "--- Sampling & Evaluating: reg=$reg, step=$step ---"
    IDIR="celebahq_iso_${reg}_s${step}"
    GEN_DIR="$REG_GEN_DIR/$IDIR"

    TMP_CFG_NAME="_tmp_celebahq_ckpt_${step}.yml"
    make_cfg_with_ckpt "$step" "$BASE_CONFIG" "$TMP_CFG_NAME"

    # Sampling (assumes Python creates needed dirs)
    run_main \
      --config "$TMP_CFG_NAME" \
      --exp "$EXP_ROOT" \
      --doc "celebahq_iso_${reg}" \
      --reg "$reg" \
      --sample --fid \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni \
      -i "$IDIR"

    # Evaluation (assumes eval/log dirs already exist)
    EVAL_TXT="$REG_LOG_DIR/logs/eval_ckpt_${step}.txt"
    python "$EVAL_SCRIPT" \
      --real_dir "$REAL_DIR" \
      --gen_dir  "$GEN_DIR" \
      --resolution "$EVAL_RES" \
      --max_images "$MAX_IMAGES" \
      --batch_size "$BATCH_SIZE" \
      --workers "$WORKERS" \
      --nearest_k "$NEAREST_K" \
      ${EVAL_CPU:+--cpu} \
      --seed "$EVAL_SEED" | tee "$EVAL_TXT"

    CSV_OUT="$REG_LOG_DIR/eval_summary.csv"
    # Do NOT create file/dir; append only if file exists.
    if [ -f "$CSV_OUT" ]; then
      parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"
    else
      # If you want to strictly avoid creating files, comment the next two lines.
      echo "reg,step,fid,precision,recall,density,coverage" > "$CSV_OUT"
      parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"
    fi

    rm -f "configs/$TMP_CFG_NAME"
  done
done

echo "CelebA-HQ sampling & evaluation done."
