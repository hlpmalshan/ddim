#!/usr/bin/env bash

REAL_DIR="ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/train/NORMAL"
SYN_DIR="ddim_pneumonia/image_samples/pneumonia_samples_0.3"
OUT_DIR="ddim_pneumonia/datasets/Pneumonia_dataset/chest_xray_preprocessed/train/NORMAL_ISO_100"

mkdir -p "$OUT_DIR"
rsync -a --exclude='.*' "$REAL_DIR/" "$OUT_DIR/"
N_REAL=2534
N_SYN=$((N_REAL * 100 / 100))
find "$SYN_DIR" -type f -print0 \
  | sort -z \
  | head -z -n "$N_SYN" \
  | while IFS= read -r -d '' f; do
      cp -n "$f" "$OUT_DIR/"
    done