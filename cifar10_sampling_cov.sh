#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Sampling + predicted-noise covariance from the SAME reverse trajectories.
# For each checkpoint, transient_covarince_cifar10_ddpm_save.py runs NUM_IMAGES
# full 1000-step DDPM trajectories, records the covariance of eps_theta at 25
# timesteps, and saves each trajectory's final image. FID/PRDC are then computed
# on those images. (cifar10_sampling.sh is the images-only version.)

# -------- Config --------
# Regularization values to sweep (must match your training runs)
reg_values=(0.01)

# EXACT checkpoint steps to process per run.
# e.g., CKPT_STEPS=(120000 200000); leave empty () to auto-discover all
# (~1.7 GB of covariance matrices per checkpoint, so prefer a short list).
CKPT_STEPS=(800000)

# Base config (relative to configs/)
BASE_CONFIG="cifar10.yml"

# Experiment roots
EXP_ROOT="ddim_cifar10"          # where logs/ and image_samples/ live
DATA_ROOT="ddim_cifar10"         # where datasets live

# Real images dir for FID/PRDC (adjust to your dataset layout)
REAL_DIR="$DATA_ROOT/datasets/cifar10/train"

# Fail before sampling when the real-image export is missing.
if [ ! -d "$REAL_DIR" ]; then
  echo "[ERROR] Missing real images: $REAL_DIR" >&2
  echo "Run python cifar10_batch_to_image.py before sampling/evaluation." >&2
  exit 1
fi

# Image size
IMAGE_SIZE=32
# Output / logs
LOGS_DIR="$EXP_ROOT/logs"
GEN_BASE="$EXP_ROOT/image_samples"
EVAL_SCRIPT="evaluation.py"
FEATURE_EVAL_SCRIPT="evaluation_feature_space.py"
COV_SCRIPT="transient_covarince_cifar10_ddpm_save.py"

# Sampling params (single GPU)
NUM_IMAGES=${NUM_IMAGES:-50000}
SAMPLE_BATCH=${SAMPLE_BATCH:-256}
GPU_ID=${GPU_ID:-1}

# -------- Helpers --------
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

    CKPT_PATH="$(ckpt_path_for "$reg" "$step")"
    if [ ! -f "$CKPT_PATH" ]; then
      echo "[SKIP] reg=$reg step=$step (missing $CKPT_PATH)"
      continue
    fi

    # Separate names from cifar10_sampling.sh so the two never mix.
    TAG="_n${NUM_IMAGES}"
    GEN_DIR="$GEN_BASE/cifar10_iso_${reg}_s${step}_cov${TAG}"
    COV_OUT="$REG_LOG_DIR/covariance_ckpt_${step}${TAG}"
    #GEN_DIR="$GEN_BASE/cifar10_iso_${reg}_s${step}_cov"
    #COV_OUT="$REG_LOG_DIR/covariance_ckpt_${step}"

    # 1) Sampling + covariance in one pass. Skipped when already complete; an
    #    interrupted run restarts this checkpoint from scratch (images included).
    if [ ! -f "$COV_OUT/covariance_summary.csv" ]; then
      echo "--- Sampling + covariance: reg=$reg, step=$step ---"
      if ! python "$COV_SCRIPT" \
          --ckpt "$CKPT_PATH" \
          --config "configs/$BASE_CONFIG" \
          --outdir "$COV_OUT" \
          --save_images_dir "$GEN_DIR" \
          --num_images_per_t "$NUM_IMAGES" \
          --batch_size "$SAMPLE_BATCH" \
          --label "reg=$reg" \
          --device "cuda:$GPU_ID" \
          --overwrite; then
        echo "[WARN] Sampling + covariance failed for reg=$reg step=$step; skipping its evaluation."
        continue
      fi
    fi

    # 2) Evaluation on the images from step 1 (written to .tmp first, so an
    #    interrupted evaluation is redone on the next run).
    EVAL_TXT="$REG_LOG_DIR/eval_cov_ckpt_${step}${TAG}.txt"
    if [ ! -f "$EVAL_TXT" ]; then
      echo "--- Evaluating: reg=$reg, step=$step ---"
      python "$EVAL_SCRIPT" \
        --image_size "$IMAGE_SIZE" \
        --real_dir "$REAL_DIR" \
        --gen_dir  "$GEN_DIR" | tee "$EVAL_TXT.tmp"
      mv "$EVAL_TXT.tmp" "$EVAL_TXT"

      CSV_OUT="$REG_LOG_DIR/eval_cov${TAG}_summary.csv"
      if [ ! -f "$CSV_OUT" ]; then
        echo "reg,step,fid,precision,recall,density,coverage,is_mean,is_std" > "$CSV_OUT"
      fi
      parse_eval_to_csv_row "$EVAL_TXT" "$reg" "$step" >> "$CSV_OUT"
    fi

    FEATURE_EVAL_TXT="$REG_LOG_DIR/eval_feature_space_cov_ckpt_${step}${TAG}.txt"
    if [ ! -f "$FEATURE_EVAL_TXT" ]; then
      python "$FEATURE_EVAL_SCRIPT" \
        --image_size "$IMAGE_SIZE" \
        --real_dir "$REAL_DIR" \
        --gen_dir "$GEN_DIR" | tee "$FEATURE_EVAL_TXT.tmp"
      mv "$FEATURE_EVAL_TXT.tmp" "$FEATURE_EVAL_TXT"

      FEATURE_CSV_OUT="$REG_LOG_DIR/eval_feature_space_cov${TAG}_summary.csv"
      if [ ! -f "$FEATURE_CSV_OUT" ]; then
        echo "reg,step,fid,precision,recall,density,coverage" > "$FEATURE_CSV_OUT"
      fi
      # Retain FID and PRDC fields; this evaluator does not calculate IS.
      parse_eval_to_csv_row "$FEATURE_EVAL_TXT" "$reg" "$step" \
        | cut -d, -f1-7 >> "$FEATURE_CSV_OUT"
    fi
  done
done

echo "CIFAR10 sampling + covariance + evaluation done."
