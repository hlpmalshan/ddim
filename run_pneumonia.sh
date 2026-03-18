#!/usr/bin/env bash
set -euo pipefail

# Force GPU 6 for this run unless overridden by the caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="${repo_dir}/Pneumonia_dataset/outputs"
mkdir -p "${out_dir}"

python "${repo_dir}/pneumonia_train.py" \
  --data_root "${repo_dir}/Pneumonia_dataset/chest_xray" \
  --output_dir "${out_dir}" \
  2>&1 | tee "${out_dir}/train.log"
