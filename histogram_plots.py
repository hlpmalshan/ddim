#!/usr/bin/env python3
# rgb_histogram_figure.py
# Usage:
#   python rgb_histogram_figure.py /path/to/image.jpg --out figure_name.png
# Optional:
#   python rgb_histogram_figure.py img.png --out out.png --bins 256

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def compute_histograms(img_rgb: np.ndarray, bins: int = 256):
    # img_rgb: H×W×3, uint8
    hist = {}
    bin_edges = np.linspace(0, 256, bins + 1, dtype=np.float64)
    for i, name in enumerate(("R", "G", "B")):
        channel = img_rgb[:, :, i].ravel()
        counts, _ = np.histogram(channel, bins=bin_edges)
        hist[name] = counts
    return hist, bin_edges

def plot_image_and_hists(img_rgb: np.ndarray, hist: dict, bin_edges: np.ndarray, out_path: Path):
    # Prepare figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_img = axs[0, 0]
    ax_r   = axs[0, 1]
    ax_b   = axs[1, 0]
    ax_g   = axs[1, 1]

    # Original image
    ax_img.imshow(img_rgb)
    ax_img.set_title("Original")
    ax_img.axis("off")

    # Histogram plotting helper
    def plot_hist(ax, counts, color_name, title):
        # Use bin centers for line plot
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        ax.plot(centers, counts, linewidth=1)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, linewidth=0.3, alpha=0.7)

    plot_hist(ax_r, hist["R"], "r", "Red histogram")
    plot_hist(ax_b, hist["B"], "b", "Blue histogram")
    plot_hist(ax_g, hist["G"], "g", "Green histogram")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Create a 2x2 figure: image + RGB histograms.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--out", required=True, type=Path, help="Output figure filename, e.g., result.png")
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins (default: 256)")
    args = parser.parse_args()

    # Load image and ensure RGB
    img = Image.open(args.image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_rgb = np.array(img, dtype=np.uint8)

    # Compute histograms
    hist, bin_edges = compute_histograms(img_rgb, bins=args.bins)

    # Plot and save
    plot_image_and_hists(img_rgb, hist, bin_edges, args.out)

if __name__ == "__main__":
    main()
