#!/usr/bin/env python3
# multiscale_rgb_hist_figure.py
# Usage:
#   python multiscale_rgb_hist_figure.py /path/to/img.png --out out.png
# Optional:
#   python multiscale_rgb_hist_figure.py img.jpg --out out.png --bins 256 --resample lanczos

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

def to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.asarray(pil_img, dtype=np.uint8)

def resize_rgb(arr: np.ndarray, size: int, method: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((size, size), method), dtype=np.uint8)

def compute_histograms(img_rgb: np.ndarray, bins: int):
    bin_edges = np.linspace(0, 256, bins + 1, dtype=np.float64)
    h = {}
    for i, name in enumerate(("R", "G", "B")):
        counts, _ = np.histogram(img_rgb[:, :, i].ravel(), bins=bin_edges)
        probs = counts.astype(np.float64) / counts.sum()  # normalize to probability
        h[name] = probs
    return h, bin_edges

def plot_hist(ax, probs, bin_edges, title):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    ax.plot(centers, probs, linewidth=1)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Probability")  # updated label
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.7)


def main():
    p = argparse.ArgumentParser(description="5-row figure: original image, then RGB histograms at 256,128,64,32.")
    p.add_argument("image", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--resample", type=str, default="lanczos", choices=RESAMPLE_MAP.keys())
    args = p.parse_args()

    # Load image
    img = Image.open(args.image)
    img_rgb = to_rgb_array(img)

    # Prepare sizes
    sizes = [256, 128, 64, 32]

    # If original is not 256x256, still show original as-is on row 1,
    # and compute a 256x256 version for the 256 row.
    resample = RESAMPLE_MAP[args.resample]
    scaled = {s: resize_rgb(img_rgb, s, resample) for s in sizes}

    # Compute histograms
    hists = {}
    for s in sizes:
        hists[s], bin_edges = compute_histograms(scaled[s], args.bins)

    # Figure: 5 rows x 3 cols. Row 0 spans all columns for original image.
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(nrows=5, ncols=3, figure=fig, height_ratios=[1.2, 1, 1, 1, 1])

    # Row 0: original image spanning all 3 columns
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_rgb)
    H, W = img_rgb.shape[:2]
    ax_img.set_title(f"Original image ({W}×{H})", fontsize=12)
    ax_img.axis("off")

    # Rows 1–4: RGB histograms for 256, 128, 64, 32
    labels = ["Red", "Green", "Blue"]
    for row_idx, s in enumerate(sizes, start=1):
        for col_idx, ch in enumerate(("R", "G", "B")):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            plot_hist(ax, hists[s][ch], bin_edges, f"{s}×{s} {labels[col_idx]}")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
