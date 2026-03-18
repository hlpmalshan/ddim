#!/usr/bin/env python3
"""
CelebA Transient Sweep Evaluator
================================

Purpose
-------
Evaluate transient diffusion samples saved at intermediate timesteps (t_*) against a
fixed REAL CelebA 64×64 set, producing per-timestep metrics + artifacts.

Inputs
------
- REAL images directory (64×64 recommended; any size accepted)
- Generated root directory containing timestep subfolders: t_0, t_100, ..., t_1000

Outputs (written under --out_dir)
--------------------------------
- CSV: `metrics_by_timestep.csv`
  Columns: timestep, n_images, fid, is_mean, is_std, precision, recall, density, coverage
- NPY: `metrics_matrix.npy`
  Dict with {"timesteps": [...], "metrics": np.ndarray[T,7], "columns": [...]}
  where columns = [fid, is_mean, is_std, precision, recall, density, coverage]
- Plots (PNG):
  - `fid_vs_timestep.png`
  - `is_vs_timestep.png`       (IS mean)
  - `prdc_vs_timestep.png`     (precision/recall/density/coverage)

Install requirements (pip)
--------------------------
- torch, torchvision
- torch-fidelity
- prdc
- pillow, tqdm, matplotlib, numpy

Example
-------
python3 celeba_transient_sweep.py \
  --real_dir "/data/diffusion/repos/ddim/ddim_celeba/datasets/celeba/celeba/prepocessed_imgs" \
  --gen_root "/data/diffusion/repos/ddim/transient_results_celeba_64_0.0_t0_to_t150" \
  --out_dir  "/data/diffusion/repos/ddim/transient_results_celeba_64_0.0_t0_to_t150/eval" \
  --max_images 50000 --batch_size 64 --workers 8 --nearest_k 5
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTEP_DIR_RE = re.compile(r"^t_(\d+)$")

DEFAULT_REAL_DIR = "/data/diffusion/repos/ddim/ddim_celeba/datasets/celeba/celeba/prepocessed_imgs"
DEFAULT_GEN_ROOT = "/data/diffusion/repos/ddim/transient_results_celeba_64_0.0_t0_to_t150"
DEFAULT_OUT_DIR = "/data/diffusion/repos/ddim/transient_results_celeba_64_0.0_t0_to_t150/eval"


def list_images(directory: str, exts: Iterable[str] = IMG_EXTS) -> List[str]:
    exts_l = {e.lower() for e in exts}
    if not os.path.isdir(directory):
        return []
    paths: List[str] = []
    with os.scandir(directory) as it:
        for entry in it:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower() in exts_l:
                paths.append(entry.path)
    paths.sort(key=lambda p: os.path.basename(p).lower())
    return paths


def _parse_timestep_dir(name: str) -> Optional[int]:
    m = TIMESTEP_DIR_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def find_timestep_dirs(gen_root: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(gen_root):
        return []
    out: List[Tuple[int, str]] = []
    with os.scandir(gen_root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            t = _parse_timestep_dir(entry.name)
            if t is None:
                continue
            out.append((t, entry.path))
    out.sort(key=lambda x: x[0])
    return out


def ensure_tmp64(
    files: Sequence[str],
    *,
    out_size: int,
    tmp_root: str,
    prefix: str,
) -> Tuple[str, int]:
    """
    Create a temporary directory containing exactly one 64×64 RGB image per input file.

    Strategy (robust + reasonably fast):
    - If the image is already out_size×out_size and RGB and not WEBP, copy as-is.
    - Otherwise, load with PIL, convert to RGB, resize+center-crop to out_size, save as PNG.
    """
    os.makedirs(tmp_root, exist_ok=True)
    out_dir = tempfile.mkdtemp(prefix=prefix, dir=tmp_root)

    resize_tf = transforms.Compose(
        [
            transforms.Resize(out_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(out_size),
        ]
    )

    written = 0
    for idx, src in enumerate(tqdm(files, desc=f"ensure_tmp64[{prefix}]", leave=False)):
        base = os.path.splitext(os.path.basename(src))[0]
        ext = os.path.splitext(src)[1].lower()
        dst_stem = f"{idx:06d}_{base}"

        try:
            with Image.open(src) as img:
                is_rgb = img.mode == "RGB"
                is_size = img.size == (out_size, out_size)
                if is_rgb and is_size and ext != ".webp":
                    try:
                        shutil.copy2(src, os.path.join(out_dir, f"{dst_stem}{ext}"))
                    except Exception:
                        img.convert("RGB").save(os.path.join(out_dir, f"{dst_stem}.png"))
                else:
                    img = img.convert("RGB")
                    if img.size != (out_size, out_size):
                        img = resize_tf(img)
                    img.save(os.path.join(out_dir, f"{dst_stem}.png"))
            written += 1
        except Exception as e:
            print(f"[WARN] Failed to process {src}: {e}")

    if written == 0:
        raise RuntimeError(f"No images written for prefix={prefix} (check inputs).")
    return out_dir, written


class ImageTensorDataset(Dataset):
    def __init__(self, files: Sequence[str], tfm: transforms.Compose):
        self.files = list(files)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.tfm(img)
        return x, 0


@torch.no_grad()
def extract_prdc(
    files: Sequence[str],
    *,
    batch_size: int,
    workers: int,
    desc: str,
    device: torch.device,
) -> np.ndarray:
    """
    Extract PRDC features for prdc.compute_prdc().

    This matches the repo's existing evaluation approach: use flattened 64×64 RGB tensors.
    Output shape: [N, 3*64*64] float32.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )
    ds = ImageTensorDataset(files, tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        drop_last=False,
    )

    n = len(files)
    feats = np.empty((n, 3 * 64 * 64), dtype=np.float32)
    offset = 0
    for x, _ in tqdm(dl, desc=desc, leave=False):
        x = x.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        x = x.flatten(start_dim=1)  # [B, 12288]
        b = x.shape[0]
        feats[offset : offset + b] = x.cpu().numpy()
        offset += b
    return feats


def _torch_fidelity_calculate_metrics(
    *,
    gen_dir: str,
    real_dir: str,
    cuda: bool,
    batch_size: int,
    workers: int,
) -> Dict[str, float]:
    import inspect
    import warnings

    import torch_fidelity

    warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

    base_kwargs = dict(
        input1=gen_dir,
        input2=real_dir,
        cuda=bool(cuda),
        fid=True,
        isc=True,
        verbose=False,
        samples_find_deep=False,
    )

    sig = inspect.signature(torch_fidelity.calculate_metrics)
    optional = {
        "batch_size": batch_size,
        "num_workers": workers,
        "workers": workers,
    }
    for k, v in optional.items():
        if k in sig.parameters and k not in base_kwargs:
            base_kwargs[k] = v

    metrics = torch_fidelity.calculate_metrics(**base_kwargs)
    return metrics


def _parse_fidelity_metrics(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    fid = None
    is_mean = None
    is_std = None

    for k in ("frechet_inception_distance", "fid"):
        if k in metrics:
            fid = float(metrics[k])
            break

    for k in ("inception_score_mean", "isc_mean"):
        if k in metrics:
            is_mean = float(metrics[k])
            break

    for k in ("inception_score_std", "isc_std"):
        if k in metrics:
            is_std = float(metrics[k])
            break

    if fid is None or is_mean is None or is_std is None:
        raise KeyError(
            f"Unexpected torch-fidelity keys; got: {sorted(metrics.keys())}. "
            "Expected FID + inception_score_(mean|std)."
        )
    return fid, is_mean, is_std


def plot_fid_vs_timestep(timesteps: Sequence[int], fids: Sequence[float], out_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, fids, marker="o", linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Timestep")
    plt.ylabel("FID")
    plt.title("FID vs Timestep")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_is_vs_timestep(timesteps: Sequence[int], is_mean: Sequence[float], out_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, is_mean, marker="o", linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Timestep")
    plt.ylabel("Inception Score (mean)")
    plt.title("Inception Score vs Timestep")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_prdc_vs_timestep(
    timesteps: Sequence[int],
    precision: Sequence[float],
    recall: Sequence[float],
    density: Sequence[float],
    coverage: Sequence[float],
    out_path: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, precision, marker="o", linewidth=1.5, label="precision")
    plt.plot(timesteps, recall, marker="o", linewidth=1.5, label="recall")
    plt.plot(timesteps, density, marker="o", linewidth=1.5, label="density")
    plt.plot(timesteps, coverage, marker="o", linewidth=1.5, label="coverage")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Timestep")
    plt.ylabel("PRDC")
    plt.title("PRDC vs Timestep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@dataclass(frozen=True)
class SweepConfig:
    real_dir: str
    gen_root: str
    out_dir: str
    max_images: int
    batch_size: int
    workers: int
    nearest_k: int


def _resolve_default_real_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    alt = path + "s"  # common local path: prepocessed_imgs
    if os.path.isdir(alt):
        print(f"[WARN] --real_dir not found: {path}\n       Using: {alt}")
        return alt
    return path


def _resolve_default_gen_root(path: str) -> str:
    if os.path.isdir(path):
        return path
    alt = path + "_t0_to_t150"
    if os.path.isdir(alt):
        print(f"[WARN] --gen_root not found: {path}\n       Using: {alt}")
        return alt
    return path


def main(cfg: SweepConfig) -> int:
    from prdc import compute_prdc

    real_dir = _resolve_default_real_dir(cfg.real_dir)
    gen_root = _resolve_default_gen_root(cfg.gen_root)
    out_dir = cfg.out_dir

    if not os.path.isdir(real_dir):
        raise SystemExit(f"[ERROR] real_dir does not exist: {real_dir}")
    if not os.path.isdir(gen_root):
        raise SystemExit(f"[ERROR] gen_root does not exist: {gen_root}")

    timestep_dirs = find_timestep_dirs(gen_root)
    if not timestep_dirs:
        raise SystemExit(f"[ERROR] No timestep dirs found under gen_root: {gen_root} (expected t_* subdirs)")

    real_all = list_images(real_dir)
    if not real_all:
        raise SystemExit(f"[ERROR] No images found in real_dir: {real_dir}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Determine a global deterministic subset size n across ALL timesteps.
    gen_counts: List[Tuple[int, int]] = []
    for t, d in timestep_dirs:
        gen_counts.append((t, len(list_images(d))))
    min_gen = min(c for _, c in gen_counts)
    n = min(len(real_all), min_gen, int(cfg.max_images))
    if n <= 0:
        raise SystemExit("[ERROR] Computed subset size n=0; check inputs.")
    if n < 2:
        raise SystemExit(f"[ERROR] Computed subset size n={n}; need at least 2 images per set.")

    real_files = real_all[:n]

    os.makedirs(out_dir, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="tmp_celeba_transient_sweep_", dir=out_dir)

    real_tmp64 = None
    try:
        # ---- REAL: one-time tmp64 + PRDC features ----
        real_tmp64, written_real = ensure_tmp64(
            real_files, out_size=64, tmp_root=tmp_root, prefix="real64_"
        )
        if written_real < n:
            raise SystemExit(
                f"[ERROR] Only wrote {written_real}/{n} REAL images to tmp; fix real_dir inputs."
            )

        real_tmp_files = list_images(real_tmp64)
        if len(real_tmp_files) < n:
            raise SystemExit(
                f"[ERROR] REAL tmp dir has {len(real_tmp_files)}/{n} images after processing."
            )
        real_tmp_files = real_tmp_files[:n]

        real_feats = extract_prdc(
            real_tmp_files,
            batch_size=cfg.batch_size,
            workers=cfg.workers,
            desc="PRDC real",
            device=device,
        )

        # ---- Sweep timesteps ----
        header = (
            f"{'t':>6} {'n':>7} {'FID':>10} {'IS_mean':>10} {'IS_std':>10} "
            f"{'prec':>8} {'rec':>8} {'dens':>8} {'cov':>8}"
        )
        print(header)
        print("-" * len(header))

        rows: List[Dict[str, object]] = []

        for t, d in timestep_dirs:
            gen_all = list_images(d)
            if len(gen_all) < n:
                print(f"[WARN] Skip t_{t}: only {len(gen_all)} images (< n={n})")
                continue
            gen_files = gen_all[:n]

            gen_tmp64 = None
            try:
                gen_tmp64, written_gen = ensure_tmp64(
                    gen_files, out_size=64, tmp_root=tmp_root, prefix=f"gen64_t{t}_"
                )
                if written_gen < n:
                    print(f"[WARN] Skip t_{t}: only wrote {written_gen}/{n} GEN images to tmp")
                    continue

                gen_tmp_files = list_images(gen_tmp64)
                if len(gen_tmp_files) < n:
                    print(f"[WARN] Skip t_{t}: tmp dir has {len(gen_tmp_files)}/{n} images")
                    continue
                gen_tmp_files = gen_tmp_files[:n]

                metrics = _torch_fidelity_calculate_metrics(
                    gen_dir=gen_tmp64,
                    real_dir=real_tmp64,
                    cuda=torch.cuda.is_available(),
                    batch_size=cfg.batch_size,
                    workers=cfg.workers,
                )
                fid, is_mean, is_std = _parse_fidelity_metrics(metrics)

                gen_feats = extract_prdc(
                    gen_tmp_files,
                    batch_size=cfg.batch_size,
                    workers=cfg.workers,
                    desc=f"PRDC t_{t}",
                    device=device,
                )

                k = int(cfg.nearest_k)
                k = max(1, min(k, n - 1))
                prdc = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=k)

                row = dict(
                    timestep=int(t),
                    n_images=int(n),
                    fid=float(fid),
                    is_mean=float(is_mean),
                    is_std=float(is_std),
                    precision=float(prdc["precision"]),
                    recall=float(prdc["recall"]),
                    density=float(prdc["density"]),
                    coverage=float(prdc["coverage"]),
                )
                rows.append(row)

                print(
                    f"{t:6d} {n:7d} {fid:10.4f} {is_mean:10.4f} {is_std:10.4f} "
                    f"{row['precision']:8.4f} {row['recall']:8.4f} {row['density']:8.4f} {row['coverage']:8.4f}"
                )

            finally:
                if gen_tmp64 is not None:
                    shutil.rmtree(gen_tmp64, ignore_errors=True)

        if not rows:
            raise SystemExit("[ERROR] No timestep rows computed (all skipped?).")

        rows.sort(key=lambda r: int(r["timestep"]))

        # ---- Write CSV ----
        csv_path = os.path.join(out_dir, "metrics_by_timestep.csv")
        csv_cols = [
            "timestep",
            "n_images",
            "fid",
            "is_mean",
            "is_std",
            "precision",
            "recall",
            "density",
            "coverage",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # ---- Write NPY matrix artifact ----
        timesteps = [int(r["timestep"]) for r in rows]
        columns = ["fid", "is_mean", "is_std", "precision", "recall", "density", "coverage"]
        matrix = np.stack(
            [
                np.array(
                    [
                        float(r["fid"]),
                        float(r["is_mean"]),
                        float(r["is_std"]),
                        float(r["precision"]),
                        float(r["recall"]),
                        float(r["density"]),
                        float(r["coverage"]),
                    ],
                    dtype=np.float32,
                )
                for r in rows
            ],
            axis=0,
        )
        npy_path = os.path.join(out_dir, "metrics_matrix.npy")
        np.save(npy_path, {"timesteps": timesteps, "metrics": matrix, "columns": columns}, allow_pickle=True)

        # ---- Plots ----
        plot_fid_vs_timestep(
            timesteps,
            [float(r["fid"]) for r in rows],
            os.path.join(out_dir, "fid_vs_timestep.png"),
        )
        plot_is_vs_timestep(
            timesteps,
            [float(r["is_mean"]) for r in rows],
            os.path.join(out_dir, "is_vs_timestep.png"),
        )
        plot_prdc_vs_timestep(
            timesteps,
            [float(r["precision"]) for r in rows],
            [float(r["recall"]) for r in rows],
            [float(r["density"]) for r in rows],
            [float(r["coverage"]) for r in rows],
            os.path.join(out_dir, "prdc_vs_timestep.png"),
        )

        print(f"\n[OK] Wrote: {csv_path}")
        print(f"[OK] Wrote: {npy_path}")
        print(f"[OK] Wrote plots under: {out_dir}")
        return 0

    finally:
        if real_tmp64 is not None:
            shutil.rmtree(real_tmp64, ignore_errors=True)
        shutil.rmtree(tmp_root, ignore_errors=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CelebA transient sweep: FID/IS (torch-fidelity) + PRDC (prdc) per timestep")
    p.add_argument(
        "--real_dir",
        type=str,
        default=DEFAULT_REAL_DIR,
        help="Directory containing REAL CelebA 64×64 images.",
    )
    p.add_argument(
        "--gen_root",
        type=str,
        default=DEFAULT_GEN_ROOT,
        help="Root directory containing timestep subfolders t_*.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for CSV/NPY/plots.",
    )
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nearest_k", type=int, default=5)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    resolved_gen_root = _resolve_default_gen_root(args.gen_root)
    out_dir = args.out_dir
    if out_dir == DEFAULT_OUT_DIR and args.gen_root != DEFAULT_GEN_ROOT:
        out_dir = os.path.join(resolved_gen_root, "eval")
    cfg = SweepConfig(
        real_dir=args.real_dir,
        gen_root=args.gen_root,
        out_dir=out_dir,
        max_images=args.max_images,
        batch_size=args.batch_size,
        workers=args.workers,
        nearest_k=args.nearest_k,
    )
    raise SystemExit(main(cfg))
