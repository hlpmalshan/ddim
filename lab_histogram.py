#!/usr/bin/env python3
# Layout:
#   Row 1 → 3 columns: L* PDF | a* PDF | b* PDF
#   Row 2 → 2 columns: Original image | a*–b* 2D PDF (clipped)

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from skimage.color import rgb2lab

# ---------- helpers ----------
def to_rgb_u8(x):
    if x.mode != "RGB":
        x = x.convert("RGB")
    return np.asarray(x, dtype=np.uint8)

def rgb2lab_np(x):
    return rgb2lab(x.astype(np.float32) / 255.0)

def hist1d_prob(x, bins, r):
    c, e = np.histogram(x, bins=bins, range=r)
    p = c / c.sum() if c.sum() > 0 else c
    centers = 0.5 * (e[:-1] + e[1:])
    return centers, p

def hist2d_prob(x, y, bins, rx, ry):
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[rx, ry])
    P = H / H.sum() if H.sum() > 0 else H
    return P.T, xe, ye  # P indexed [b*, a*]

def plot_1d(ax, centers, probs, title, xlabel):
    ax.plot(centers, probs, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.grid(True, linewidth=0.3, alpha=0.7)

def plot_img(ax, im):
    ax.imshow(im)
    ax.set_title("Original")
    ax.axis("off")

def plot_ab_clipped(ax, P, xa, yb, title, lo, hi):
    a_idx = np.where((xa[:-1] >= lo) & (xa[1:] <= hi))[0]
    b_idx = np.where((yb[:-1] >= lo) & (yb[1:] <= hi))[0]
    if len(a_idx) == 0 or len(b_idx) == 0:
        raise ValueError("Clip window outside histogram range.")
    Pw = P[b_idx][:, a_idx]
    xa_w = xa[a_idx[0]:a_idx[-1] + 2]
    yb_w = yb[b_idx[0]:b_idx[-1] + 2]

    im = ax.imshow(
        Pw,
        origin="lower",
        extent=[xa_w[0], xa_w[-1], yb_w[0], yb_w[-1]],
        aspect="equal"
    )
    ax.set_title(title)
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Probability")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Lab PDFs with 2-row layout (3 plots top, 2 plots bottom).")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins-l", type=int, default=256)
    ap.add_argument("--bins-ab", type=int, default=256)
    ap.add_argument("--clip", type=float, nargs=2, default=[-25.0, 25.0], metavar=("MIN", "MAX"))
    args = ap.parse_args()

    # Load and convert
    rgb = to_rgb_u8(Image.open(args.image))
    lab = rgb2lab_np(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Ranges
    rL = (0.0, 100.0)
    ra = (-128.0, 128.0)
    rb = (-128.0, 128.0)

    # Histograms as probabilities
    cL, pL = hist1d_prob(L.ravel(), args.bins_l, rL)
    ca, pa = hist1d_prob(a.ravel(), args.bins_ab, ra)
    cb, pb = hist1d_prob(b.ravel(), args.bins_ab, rb)
    Pab, xa, yb = hist2d_prob(a.ravel(), b.ravel(), args.bins_ab, ra, rb)

    lo, hi = args.clip

    # Figure with nested grids:
    # Outer 2x1 → Row1 (3 cols), Row2 (2 cols)
    fig = plt.figure(figsize=(18, 10))
    gs_outer = GridSpec(2, 1, height_ratios=[1.0, 1.2], figure=fig)

    # Row 1: 3 columns (L*, a*, b*)
    gs_row1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0])
    ax_L = fig.add_subplot(gs_row1[0, 0]); plot_1d(ax_L, cL, pL, "L* PDF", "L*")
    ax_a = fig.add_subplot(gs_row1[0, 1]); plot_1d(ax_a, ca, pa, "a* PDF", "a*")
    ax_b = fig.add_subplot(gs_row1[0, 2]); plot_1d(ax_b, cb, pb, "b* PDF", "b*")

    # Row 2: 2 columns (Image | AB 2D PDF)
    gs_row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1], wspace=0.15)
    ax_img = fig.add_subplot(gs_row2[0, 0]); plot_img(ax_img, rgb)
    ax_ab  = fig.add_subplot(gs_row2[0, 1]); plot_ab_clipped(ax_ab, Pab, xa, yb, "a*–b* PDF (clipped)", lo, hi)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
