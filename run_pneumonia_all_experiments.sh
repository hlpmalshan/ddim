#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

train_dir="${TRAIN_DIR:-${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/train}"
val_dir="${VAL_DIR:-${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/val}"
test_dir="${TEST_DIR:-${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/test}"

out_root="${OUT_ROOT:-${repo_dir}/ddim_pneumonia/datasets/Pneumonia_dataset/experiment_outputs}"
summary_dir="${SUMMARY_DIR:-${out_root}}"

model="${MODEL:-resnet50}"
train_mode="${TRAIN_MODE:-head}"
epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-64}"
img_size="${IMG_SIZE:-224}"
lr="${LR:-1e-4}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
workers="${WORKERS:-10}"
threshold="${THRESHOLD:-0.5}"
focal_gamma="${FOCAL_GAMMA:-2.0}"

seed_list="${SEEDS:-42,43,44,45,46}"
include_focal="${INCLUDE_FOCAL:-false}"

DISTRIBUTED="${DISTRIBUTED:-true}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29506}"

AMP="${AMP:-false}"
AMP_FLAG=()
if [ "$AMP" = "true" ]; then
  AMP_FLAG+=(--amp)
fi

mkdir -p "$out_root"

run_train() {
  local condition="$1"
  local normal_variant="$2"
  local loss_type="$3"
  local seed="$4"
  local out_dir="${out_root}/${condition}/seed_${seed}"

  mkdir -p "$out_dir"
  echo
  echo "=== condition=${condition} seed=${seed} variant=${normal_variant} loss=${loss_type} ==="

  if [ "$DISTRIBUTED" = "true" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    torchrun \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_port="$MASTER_PORT" \
      "${repo_dir}/pneumonia_variant_train_ddp.py" \
      --train_dir "$train_dir" \
      --val_dir "$val_dir" \
      --test_dir "$test_dir" \
      --normal_variant "$normal_variant" \
      --model "$model" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --img_size "$img_size" \
      --lr "$lr" \
      --weight_decay "$weight_decay" \
      --num_workers "$workers" \
      --train_mode "$train_mode" \
      --loss_type "$loss_type" \
      --focal_gamma "$focal_gamma" \
      --decision_threshold "$threshold" \
      --seed "$seed" \
      --output_dir "$out_dir" \
      "${AMP_FLAG[@]}"
  else
    python "${repo_dir}/pneumonia_variant_train_ddp.py" \
      --train_dir "$train_dir" \
      --val_dir "$val_dir" \
      --test_dir "$test_dir" \
      --normal_variant "$normal_variant" \
      --model "$model" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --img_size "$img_size" \
      --lr "$lr" \
      --weight_decay "$weight_decay" \
      --num_workers "$workers" \
      --train_mode "$train_mode" \
      --loss_type "$loss_type" \
      --focal_gamma "$focal_gamma" \
      --decision_threshold "$threshold" \
      --seed "$seed" \
      --output_dir "$out_dir" \
      "${AMP_FLAG[@]}"
  fi
}

IFS=',' read -r -a seeds <<< "$seed_list"

for seed in "${seeds[@]}"; do
  run_train "real_imbalanced" "NORMAL" "ce" "$seed"
  run_train "real_imbalanced_weighted" "NORMAL" "weighted_ce" "$seed"
  run_train "real_ddpm_synth_balanced" "NORMAL_DDPM_100" "ce" "$seed"
  run_train "real_iso_synth_balanced" "NORMAL_ISO_100" "ce" "$seed"
  if [ "$include_focal" = "true" ]; then
    run_train "real_imbalanced_focal" "NORMAL" "focal" "$seed"
  fi
done

python "${repo_dir}/aggregate_pneumonia_metrics.py" \
  --root "$out_root" \
  --output_dir "$summary_dir"

echo
echo "All experiments finished."
