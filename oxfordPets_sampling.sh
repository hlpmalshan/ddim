#!/bin/bash
set -euo pipefail
shopt -s nullglob

# -------- Config --------
# Regularization values to sweep (must match your training runs)
reg_values=(0.0 0.3)

# EXACT checkpoint steps to process per run.
# e.g., CKPT_STEPS=(120000 200000); leave empty () to auto-discover all.
CKPT_STEPS=( 200560 175490 150420 125350 100280 75210 50140 25070 225630 250700 275770 300840 325910 350980 376050 401120 426190 451260 476330 501400 )

# Base config (relative to configs/)
BASE_CONFIG="oxfordIIITpet.yml"

# Experiment roots
EXP_ROOT="ddim_ox_pet"          # where logs/ and image_samples/ live
DATA_ROOT="ddim_ox_pet"         # where datasets live

# Real images dir for FID/PRDC (adjust to your dataset layout)
REAL_DIR="$DATA_ROOT/ddim_ox_pet/datasets/oxford_pets/oxford-iiit-pet/images"

# Output / logs
LOGS_DIR="$EXP_ROOT/logs"
GEN_BASE="$EXP_ROOT/image_samples"
EVAL_SCRIPT="evaluation.py"

# Sampling params
TIMESTEPS=${TIMESTEPS:-1000}
ETA=${ETA:-1}

# Single-GPU / DDP
GPU_ID=${GPU_ID:-0}
DISTRIBUTED=${DISTRIBUTED:-false}
GPUS=${GPUS:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-0}
MASTER_PORT=${MASTER_PORT:-29501}
DIST_BACKEND=${DIST_BACKEND:-nccl}

# -------- Helpers --------
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
      if (in_sampling && $1 ~ /^ckpt_id:/) {
        sub(/ckpt_id:[[:space:]]*[0-9]+/, "ckpt_id: " step)
      }
      print
    }
  ' "configs/${in_rel}" > "configs/${out_rel}"
}

discover_ckpts() {
  local dir="$1"
  if [ ${#CKPT_STEPS[@]} -gt 0 ]; then
    for s in "${CKPT_STEPS[@]}"; do
      local p="$dir/ckpt_${s}.pth"
      [ -f "$p" ] && echo "$p"
    done
  else
    ls -1 "$dir"/ckpt_*.pth 2>/dev/null | sort -V || true
  fi
}

parse_eval_to_csv_row() {
  local txt="$1" reg="$2" step="$3"
  local fid is_mean is_std precision recall density coverage
  fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
  is_line=$(awk -F": " '/^Inception Score:/ {print $2}' "$txt" | head -n1 || true)
  is_mean=$(awk '{print $1}' <<<"$is_line" || true)
  is_std=$(awk '{print $3}'  <<<"$is_line" || true)
  precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
  recall=$(awk -F": "    '/^Recall:/ {print $2}' "$txt"    | head -n1 || true)
  density=$(awk -F": "   '/^Density:/ {print $2}' "$txt"   | head -n1 || true)
  coverage=$(awk -F": "  '/^Coverage:/ {print $2}' "$txt"  | head -n1 || true)
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${reg}" "${step}" "${fid:-NA}" "${is_mean:-NA}" "${is_std:-NA}" \
    "${precision:-NA}" "${recall:-NA}" "${density:-NA}" "${coverage:-NA}"
}

# -------- Main --------
for reg in "${reg_values[@]}"; do
  DOC="ddim_iso_${reg}"
  REG_LOG_DIR="$LOGS_DIR/$DOC"
  REG_GEN_DIR="$GEN_BASE"
  mkdir -p "$REG_GEN_DIR"

  mapfile -t ckpts < <(discover_ckpts "$REG_LOG_DIR")
  if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[INFO] No checkpoints to process for reg=$reg in $REG_LOG_DIR"
    continue
  fi

  CSV_OUT="$REG_LOG_DIR/eval_summary.csv"
  [ -f "$CSV_OUT" ] || echo "reg,step,fid,is_mean,is_std,precision,recall,density,coverage" > "$CSV_OUT"

  for ckpt_path in "${ckpts[@]}"; do
    step=$(basename "$ckpt_path" | sed -E 's/^ckpt_([0-9]+)\.pth$/\1/')
    echo "--- Sampling & Evaluating: reg=$reg, step=$step ---"

    IDIR="ox_pet_iso_${reg}_s${step}"
    GEN_DIR="$REG_GEN_DIR/$IDIR"
    mkdir -p "$GEN_DIR"
    find "$GEN_DIR" -maxdepth 1 -type f -name '*.png' -delete

    TMP_CFG_NAME="_tmp_ox_pet_ckpt_${step}.yml"
    make_cfg_with_ckpt "$step" "$BASE_CONFIG" "$TMP_CFG_NAME"

    # Sampling to EXP_ROOT/image_samples/<IDIR>
    run_main \
      --config "$TMP_CFG_NAME" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --sample --fid \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni \
      -i "$IDIR"

    # Evaluation (non-destructive; uses matched subset internally)
    EVAL_TXT="$REG_LOG_DIR/evals/eval_ckpt_${step}.txt"
    mkdir -p "$(dirname "$EVAL_TXT")"
    python "$EVAL_SCRIPT" \
      --real_dir "$REAL_DIR" \
      --gen_dir  "$GEN_DIR" | tee "$EVAL_TXT"

    parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"

    rm -f "configs/$TMP_CFG_NAME"
  done
  echo "Saved CSV: $CSV_OUT"
done

echo "OxfordPet sampling & evaluation done."
