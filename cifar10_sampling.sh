#!/bin/bash
set -euo pipefail
shopt -s nullglob

# -------- Config --------
# Regularization values to sweep (must match your training runs)
reg_values=(0.0 0.3)

# EXACT checkpoint steps to process per run.
# e.g., CKPT_STEPS=(120000 200000); leave empty () to auto-discover all.
# CKPT_STEPS=( 508300 482885 457470 432055 406640 381225 355810 330395 304980 279565 254150 228735 203320 177905 152490 127075 101660 76245 50830 25415 )
CKPT_STEPS=( 508300 406640 279565 152490 25415 )

# Base config (relative to configs/)
BASE_CONFIG="cifar10.yml"

# Experiment roots
EXP_ROOT="ddim_cifar10"          # where logs/ and image_samples/ live
DATA_ROOT="ddim_cifar10"         # where datasets live

# Real images dir for FID/PRDC (adjust to your dataset layout)
REAL_DIR="$DATA_ROOT/datasets/cifar10/train"

# Image size
IMAGE_SIZE=32
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

make_cfg_with_ckpt() {
  local step="$1" in_rel="$2" out_rel="$3"
  awk -v step="$step" '
    BEGIN { in_sampling=0; saw_ckpt=0 }
    /^[[:space:]]*sampling:[[:space:]]*$/ { in_sampling=1; print; next }
    in_sampling && /^[[:space:]]*[^[:space:]]/ && !/^[[:space:]]*ckpt_id:/ {
      if (!saw_ckpt) print "    ckpt_id: " step;
      in_sampling=0; saw_ckpt=1
    }
    {
      if (in_sampling && $1 ~ /^ckpt_id:/) {
        sub(/ckpt_id:[[:space:]]*[0-9]+/, "ckpt_id: " step); saw_ckpt=1
      }
      print
    }
  ' "configs/${in_rel}" > "configs/${out_rel}"
}

discover_steps_for_reg() {
  local reg="$1"
  local dir="$LOGS_DIR/ddim_iso_${reg}"
  ls -1 "$dir"/ckpt_*.pth 2>/dev/null \
    | sed -E 's@.*/ckpt_([0-9]+)\.pth@\1@' \
    | sort -V || true
}

ckpt_path_for() {
  local reg="$1" step="$2"
  echo "$LOGS_DIR/ddim_iso_${reg}/ckpt_${step}.pth"
}

# parse_eval_to_csv_row() {
#   local txt="$1" reg="$2" step="$3"
#   local fid precision recall density coverage
#   fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
#   precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
#   recall=$(awk -F": "    '/^Recall:/ {print $2}'    "$txt" | head -n1 || true)
#   density=$(awk -F": "   '/^Density:/ {print $2}'   "$txt" | head -n1 || true)
#   coverage=$(awk -F": "  '/^Coverage:/ {print $2}'  "$txt" | head -n1 || true)
#   printf "%s,%s,%s,%s,%s,%s,%s\n" \
#     "${reg}" "${step}" "${fid:-NA}" "${precision:-NA}" "${recall:-NA}" "${density:-NA}" "${coverage:-NA}"
# }

parse_eval_to_csv_row() {
  local txt="$1" reg="$2" step="$3"
  local fid is_mean is_std precision recall density coverage
  fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
  precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
  recall=$(awk -F": "    '/^Recall:/ {print $2}' "$txt"    | head -n1 || true)
  density=$(awk -F": "   '/^Density:/ {print $2}' "$txt"   | head -n1 || true)
  coverage=$(awk -F": "  '/^Coverage:/ {print $2}' "$txt"  | head -n1 || true)
  is_line=$(awk -F": " '/^Inception Score:/ {print $2}' "$txt" | head -n1 || true)
  is_mean=$(awk '{print $1}' <<<"$is_line" || true)
  is_std=$(awk '{print $3}'  <<<"$is_line" || true)
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${reg}" "${step}" "${fid:-NA}" "${precision:-NA}" "${recall:-NA}" "${density:-NA}" "${coverage:-NA}" "${is_mean:-NA}" "${is_std:-NA}"
}

# -------- Build checkpoint-first order --------
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

# -------- Main (by step, then by reg) --------
for step in "${STEPS[@]}"; do
  echo "=== STEP $step ==="
  for reg in "${reg_values[@]}"; do
    DOC="ddim_iso_${reg}"
    REG_LOG_DIR="$LOGS_DIR/$DOC"
    REG_GEN_DIR="$GEN_BASE"

    CKPT_PATH="$(ckpt_path_for "$reg" "$step")"
    if [ ! -f "$CKPT_PATH" ]; then
      echo "[SKIP] reg=$reg step=$step (missing $CKPT_PATH)"
      continue
    fi

    echo "--- Sampling & Evaluating: reg=$reg, step=$step ---"
    IDIR="cifar10_iso_${reg}_s${step}"
    GEN_DIR="$REG_GEN_DIR/$IDIR"

    TMP_CFG_NAME="_tmp_cifar10_ckpt_${step}.yml"
    make_cfg_with_ckpt "$step" "$BASE_CONFIG" "$TMP_CFG_NAME"

    # Sampling (assumes Python creates needed dirs)
    run_main \
      --config "$TMP_CFG_NAME" \
      --exp "$EXP_ROOT" \
      --doc "$DOC" \
      --reg "$reg" \
      --sample --fid \
      --timesteps "$TIMESTEPS" --eta "$ETA" --ni \
      -i "$IDIR"

    # Evaluation (assumes eval/log dirs exist already)
    EVAL_TXT="$REG_LOG_DIR/eval_ckpt_${step}.txt"
    python "$EVAL_SCRIPT" \
      --image_size "$IMAGE_SIZE" \
      --real_dir "$REAL_PROCCED_DIR" \
      --gen_dir  "$GEN_DIR" | tee "$EVAL_TXT"

    CSV_OUT="$REG_LOG_DIR/eval_summary.csv"
    if [ -f "$CSV_OUT" ]; then
      parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"
    else
      echo "reg,step,fid,precision,recall,density,coverage,is_mean,is_std" > "$CSV_OUT"
      parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"
    fi

    rm -f "configs/$TMP_CFG_NAME"
  done
done

echo "CIFAR10 sampling & evaluation done (checkpoint-first, no mkdir)."

# discover_ckpts() {
#   local dir="$1"
#   if [ ${#CKPT_STEPS[@]} -gt 0 ]; then
#     for s in "${CKPT_STEPS[@]}"; do
#       local p="$dir/ckpt_${s}.pth"
#       [ -f "$p" ] && echo "$p"
#     done
#   else
#     ls -1 "$dir"/ckpt_*.pth 2>/dev/null | sort -V || true
#   fi
# }

# parse_eval_to_csv_row() {
#   local txt="$1" reg="$2" step="$3"
#   local fid is_mean is_std precision recall density coverage
#   fid=$(awk -F": " '/^FID:/ {print $2}' "$txt" | head -n1 || true)
#   is_line=$(awk -F": " '/^Inception Score:/ {print $2}' "$txt" | head -n1 || true)
#   is_mean=$(awk '{print $1}' <<<"$is_line" || true)
#   is_std=$(awk '{print $3}'  <<<"$is_line" || true)
#   precision=$(awk -F": " '/^Precision:/ {print $2}' "$txt" | head -n1 || true)
#   recall=$(awk -F": "    '/^Recall:/ {print $2}' "$txt"    | head -n1 || true)
#   density=$(awk -F": "   '/^Density:/ {print $2}' "$txt"   | head -n1 || true)
#   coverage=$(awk -F": "  '/^Coverage:/ {print $2}' "$txt"  | head -n1 || true)
#   printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
#     "${reg}" "${step}" "${fid:-NA}" "${is_mean:-NA}" "${is_std:-NA}" \
#     "${precision:-NA}" "${recall:-NA}" "${density:-NA}" "${coverage:-NA}"
# }

# # -------- Main --------
# for reg in "${reg_values[@]}"; do
#   DOC="ddim_iso_${reg}"
#   REG_LOG_DIR="$LOGS_DIR/$DOC"
#   REG_GEN_DIR="$GEN_BASE"
# #  mkdir -p "$REG_GEN_DIR"

#   mapfile -t ckpts < <(discover_ckpts "$REG_LOG_DIR")
#   if [ ${#ckpts[@]} -eq 0 ]; then
#     echo "[INFO] No checkpoints to process for reg=$reg in $REG_LOG_DIR"
#     continue
#   fi

#   CSV_OUT="$REG_LOG_DIR/eval_summary.csv"
#   [ -f "$CSV_OUT" ] || echo "reg,step,fid,is_mean,is_std,precision,recall,density,coverage" > "$CSV_OUT"

#   for ckpt_path in "${ckpts[@]}"; do
#     step=$(basename "$ckpt_path" | sed -E 's/^ckpt_([0-9]+)\.pth$/\1/')
#     echo "--- Sampling & Evaluating: reg=$reg, step=$step ---"

#     IDIR="cifar10_iso_${reg}_s${step}"
#     GEN_DIR="$REG_GEN_DIR/$IDIR"
#     # mkdir -p "$GEN_DIR"
#     # find "$GEN_DIR" -maxdepth 1 -type f -name '*.png' -delete

#     TMP_CFG_NAME="_tmp_cifar10_ckpt_${step}.yml"
#     make_cfg_with_ckpt "$step" "$BASE_CONFIG" "$TMP_CFG_NAME"

#     # Sampling to EXP_ROOT/image_samples/<IDIR>
#     run_main \
#       --config "$TMP_CFG_NAME" \
#       --exp "$EXP_ROOT" \
#       --doc "$DOC" \
#       --reg "$reg" \
#       --sample --fid \
#       --timesteps "$TIMESTEPS" --eta "$ETA" --ni \
#       -i "$IDIR"

#     # Evaluation (non-destructive; uses matched subset internally)
#     EVAL_TXT="$REG_LOG_DIR/evals/eval_ckpt_${step}.txt"
#     mkdir -p "$(dirname "$EVAL_TXT")"
#     python "$EVAL_SCRIPT" \
#       --image_size "$IMAGE_SIZE" \
#       --real_dir "$REAL_DIR" \
#       --gen_dir  "$GEN_DIR" | tee "$EVAL_TXT"

#     parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"

#     rm -f "configs/$TMP_CFG_NAME"
#   done
#   echo "Saved CSV: $CSV_OUT"
# done

# echo "CIFAR10 sampling & evaluation done."
