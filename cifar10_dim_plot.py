#!/usr/bin/env python3
# cifar_embed_from_filenames.py

import argparse, re, random, numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CIFAR10_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def load_image(path: Path, size: int | None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    return arr.reshape(-1).astype(np.float32) / 255.0

def infer_class_id_from_filename(path: Path) -> Optional[int]:
    # Example: train1_12_7_quarter_horse_s_000672 -> digits = [1,12,7,000672] -> class id = 7 (3rd number)
    digits = re.findall(r"\d+", path.stem)
    if len(digits) < 3:
        return None
    return int(digits[2])

def label_from_id(cls: int, map_names: bool) -> str:
    if map_names and 0 <= cls <= 9:
        return CIFAR10_NAMES[cls]
    return f"{cls}"

def sample_indices(n_total: int, n_keep: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    return idx[:n_keep]

def scatter_2d(ax, xy: np.ndarray, labels: List[str], title: str, annotate: bool):
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    colors = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.7, c=[colors[l] for l in labels], linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if annotate:
        for u in uniq:
            mask = np.array([l == u for l in labels])
            if not mask.any():
                continue
            c = xy[mask].mean(axis=0)
            ax.text(c[0], c[1], u, fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    if len(uniq) <= 12:
        handles = [plt.Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=colors[u], markersize=6, label=u)
                   for u in uniq]
        ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="best")

def parse_classes(s: Optional[str]) -> Optional[Set[int]]:
    if not s:
        return None
    toks = [t for t in re.split(r"[,\s]+", s.strip()) if t]
    return set(int(t) for t in toks)

def main():
    ap = argparse.ArgumentParser(description="PCA and UMAP 2D embeddings; class parsed as the 3rd number in filename.")
    ap.add_argument("input_dir", type=str, help="Folder with images.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for plots.")
    ap.add_argument("--max_samples", type=int, default=10000, help="Max images to use.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument("--resize", type=int, default=32, help="Resize to square. 0 keeps original.")
    ap.add_argument("--umap_n_neighbors", type=int, default=10)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--map_cifar10_names", action="store_true", help="Map labels 0–9 to CIFAR-10 names.")
    ap.add_argument("--annotate", action="store_true", help="Write class names at cluster centroids.")
    ap.add_argument("--classes", type=str, default=None,
                    help='Filter to specific class IDs. Space or comma separated, e.g. "0 1 7" or "0,1,7".')
    args = ap.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_images(in_root)
    if not files:
        raise SystemExit(f"No images found under {in_root}")

    n = min(len(files), args.max_samples)
    files = [files[i] for i in sample_indices(len(files), n, args.seed)]
    size = None if args.resize == 0 else int(args.resize)
    wanted = parse_classes(args.classes)

    X_list, cls_ids = [], []
    for fp in files:
        try:
            cid = infer_class_id_from_filename(fp)
            if cid is None:
                continue
            if wanted is not None and cid not in wanted:
                continue
            X_list.append(load_image(fp, size))
            cls_ids.append(cid)
        except Exception:
            continue

    if not X_list:
        raise SystemExit("No images passed the filtering/reading stage.")

    X = np.stack(X_list, axis=0)
    labels = [label_from_id(cid, args.map_cifar10_names) for cid in cls_ids]

    # filename tag
    if wanted is None:
        tag = "cls-all"
    else:
        tag = "cls-" + "_".join(str(c) for c in sorted(wanted))

    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    xlab = f"PC1 ({evr[0]*100:.1f}% var)"
    ylab = f"PC2 ({evr[1]*100:.1f}% var)"
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    scatter_2d(ax, X_pca, labels, "PCA (2D)", annotate=args.annotate)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    pca_path = out_root / f"cifar_embed_pca2d_{tag}.png"
    fig.tight_layout(); fig.savefig(pca_path); plt.close(fig)
    print(f"Saved: {pca_path}")

    # UMAP
    if umap is None:
        print("umap-learn not installed. Skipping UMAP.")
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=args.seed,
            verbose=False,
        )
        X_umap = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        scatter_2d(ax, X_umap, labels, "UMAP (2D)", annotate=args.annotate)
        umap_path = out_root / f"cifar_embed_umap2d_{tag}.png"
        fig.tight_layout(); fig.savefig(umap_path); plt.close(fig)
        print(f"Saved: {umap_path}")

if __name__ == "__main__":
    main()
