#!/usr/bin/env python3
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

TIMESTEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EPS = 1e-8
METHODS = ("ddpm", "iso")
METHOD_LABELS = {
    "ddpm": "DDPM (r=0.0)",
    "iso": "ISO diffusion (r=0.3)",
}


@dataclass
class WelfordState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def as_tuple(self) -> Tuple[int, float, float]:
        return self.n, self.mean, self.m2


def merge_welford(
    a: Tuple[int, float, float],
    b: Tuple[int, float, float],
) -> Tuple[int, float, float]:
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b

    if n_a == 0:
        return b
    if n_b == 0:
        return a

    delta = mean_b - mean_a
    n = n_a + n_b
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + (delta * delta) * (n_a * n_b / n)
    return n, mean, m2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated SNR comparison for transient diffusion results."
    )
    parser.add_argument("--ddpm_dir", type=str, required=True)
    parser.add_argument("--iso_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=("celeba64", "cifar10_32", "cifar100_32"),
    )
    parser.add_argument("--dataset_title", type=str, required=True)
    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Maximum number of image IDs to evaluate; -1 means all available IDs.",
    )
    parser.add_argument(
        "--write_per_image_csv",
        type=int,
        default=0,
        choices=(0, 1),
    )
    parser.add_argument(
        "--use_common_ids",
        type=int,
        default=1,
        choices=(0, 1),
        help="1 = use intersection of t_0 IDs across DDPM and ISO; 0 = union.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1",
        help="Comma-separated visible CUDA device indices (after CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--allow_cpu_fallback",
        type=int,
        default=0,
        choices=(0, 1),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plot_python",
        type=str,
        default="python3",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def filename_sort_key(filename: str) -> Tuple[int, object]:
    stem, _ = os.path.splitext(filename)
    if stem.isdigit():
        return 0, int(stem)
    return 1, stem


def list_png_filenames(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = [name for name in os.listdir(folder) if name.lower().endswith(".png")]
    files.sort(key=filename_sort_key)
    return files


def validate_timestep_folders(root: str, timesteps: List[int]) -> None:
    missing = []
    for t in timesteps:
        t_folder = os.path.join(root, f"t_{t}")
        if not os.path.isdir(t_folder):
            missing.append(t_folder)
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing timestep folders in {root}:\n{joined}")


def parse_devices(devices_arg: str) -> List[int]:
    tokens = [token.strip() for token in devices_arg.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No CUDA devices provided in --devices.")
    devices = []
    for token in tokens:
        device_index = int(token)
        if device_index < 0:
            raise ValueError(f"Invalid device index: {token}")
        devices.append(device_index)
    return devices


def split_filenames(filenames: List[str], num_workers: int) -> List[List[str]]:
    return [filenames[i::num_workers] for i in range(num_workers)]


def load_png_to_device(path: str, device: object, torch_module: object) -> object:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch_module.from_numpy(array).to(device=device, non_blocking=False)


def worker_compute(
    worker_idx: int,
    device_idx: int,
    filenames: List[str],
    ddpm_dir: str,
    iso_dir: str,
    timesteps: List[int],
    write_per_image_csv: bool,
    temp_dir: str,
    use_cuda: bool,
) -> Dict[str, object]:
    import torch

    torch.set_grad_enabled(False)

    if use_cuda:
        visible_cuda_count = torch.cuda.device_count()
        if device_idx >= visible_cuda_count:
            raise RuntimeError(
                f"Worker {worker_idx}: requested cuda:{device_idx}, but only "
                f"{visible_cuda_count} visible device(s) are available."
            )
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
        device_desc = f"cuda:{device_idx}"
    else:
        device = torch.device("cpu")
        device_desc = "cpu"

    method_roots = {
        "ddpm": ddpm_dir,
        "iso": iso_dir,
    }
    stats: Dict[str, Dict[int, WelfordState]] = {
        method: {t: WelfordState() for t in timesteps} for method in METHODS
    }

    per_image_path = None
    per_image_file = None
    per_image_writer = None
    if write_per_image_csv:
        per_image_path = os.path.join(temp_dir, f"per_image_worker_{worker_idx}.csv")
        per_image_file = open(per_image_path, "w", newline="")
        per_image_writer = csv.writer(per_image_file)
        per_image_writer.writerow(["method", "timestep", "filename", "snr_db"])

    try:
        total = len(filenames)
        for index, filename in enumerate(filenames, start=1):
            for method, root in method_roots.items():
                x0_path = os.path.join(root, "t_0", filename)
                if not os.path.isfile(x0_path):
                    continue

                x0 = load_png_to_device(x0_path, device, torch)
                signal = torch.mean(x0 * x0)

                for t in timesteps:
                    xt_path = os.path.join(root, f"t_{t}", filename)
                    if not os.path.isfile(xt_path):
                        continue

                    xt = load_png_to_device(xt_path, device, torch)
                    if xt.shape != x0.shape:
                        del xt
                        continue

                    noise = torch.mean((xt - x0) * (xt - x0))
                    snr = 10.0 * torch.log10(signal / (noise + EPS))
                    snr_value = float(snr.item())

                    if math.isfinite(snr_value):
                        stats[method][t].update(snr_value)
                        if per_image_writer is not None:
                            per_image_writer.writerow([method, t, filename, f"{snr_value:.10f}"])

                    del xt, noise, snr

                del x0, signal

            if index % 200 == 0:
                print(
                    f"[WORKER {worker_idx}] processed {index}/{total} ids on {device_desc}",
                    flush=True,
                )

        if use_cuda:
            torch.cuda.synchronize(device)
    finally:
        if per_image_file is not None:
            per_image_file.close()

    serialized = {
        method: {t: stats[method][t].as_tuple() for t in timesteps} for method in METHODS
    }
    return {
        "worker_idx": worker_idx,
        "device": device_desc,
        "stats": serialized,
        "per_image_csv": per_image_path,
    }


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["method", "timestep", "n", "snr_mean_db", "snr_std_db", "snr_ci95_db"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def combine_per_image_csvs(worker_csvs: List[str], out_csv: str) -> None:
    fieldnames = ["method", "timestep", "filename", "snr_db"]
    with open(out_csv, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(fieldnames)

        for path in worker_csvs:
            if path is None or not os.path.isfile(path):
                continue
            with open(path, "r", newline="") as in_handle:
                reader = csv.reader(in_handle)
                _ = next(reader, None)
                for row in reader:
                    writer.writerow(row)


def plot_snr_curves_matplotlib(
    method_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    xmax: int,
    out_png: str,
    out_pdf: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.2),
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "font.size": 11,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots()
    for label, (timesteps, mean_db, ci95_db) in method_data.items():
        mask = timesteps <= xmax
        x_vals = timesteps[mask]
        y_vals = mean_db[mask]
        c_vals = ci95_db[mask]

        valid = np.isfinite(y_vals) & np.isfinite(c_vals)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        c_vals = c_vals[valid]

        if x_vals.size == 0:
            continue

        ax.plot(x_vals, y_vals, marker="o", label=label)
        ax.fill_between(x_vals, y_vals - c_vals, y_vals + c_vals, alpha=0.2)

    ax.set_xlabel("Diffusion timestep (t)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title)
    ax.set_xlim(0, xmax)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_snr_curves_via_subprocess(
    plot_python: str,
    method_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    xmax: int,
    out_png: str,
    out_pdf: str,
) -> None:
    payload = {
        "title": title,
        "xmax": xmax,
        "out_png": out_png,
        "out_pdf": out_pdf,
        "method_data": {},
    }
    for label, (timesteps, mean_db, ci95_db) in method_data.items():
        payload["method_data"][label] = {
            "timesteps": [float(v) for v in timesteps.tolist()],
            "mean_db": [float(v) for v in mean_db.tolist()],
            "ci95_db": [float(v) for v in ci95_db.tolist()],
        }

    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            dir=os.path.dirname(out_png),
        ) as temp_json:
            json.dump(payload, temp_json)
            payload_path = temp_json.name

        helper_code = """
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open(sys.argv[1], 'r') as handle:
    payload = json.load(handle)

plt.rcParams.update({
    'figure.figsize': (7.0, 4.2),
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 4.5,
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots()
for label, values in payload['method_data'].items():
    timesteps = np.asarray(values['timesteps'], dtype=np.float64)
    mean_db = np.asarray(values['mean_db'], dtype=np.float64)
    ci95_db = np.asarray(values['ci95_db'], dtype=np.float64)
    mask = timesteps <= float(payload['xmax'])
    x_vals = timesteps[mask]
    y_vals = mean_db[mask]
    c_vals = ci95_db[mask]
    valid = np.isfinite(y_vals) & np.isfinite(c_vals)
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    c_vals = c_vals[valid]
    if x_vals.size == 0:
        continue
    ax.plot(x_vals, y_vals, marker='o', label=label)
    ax.fill_between(x_vals, y_vals - c_vals, y_vals + c_vals, alpha=0.2)

ax.set_xlabel('Diffusion timestep (t)')
ax.set_ylabel('SNR (dB)')
ax.set_title(payload['title'])
ax.set_xlim(0, float(payload['xmax']))
ax.grid(True, alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='best')
fig.tight_layout()
fig.savefig(payload['out_png'])
fig.savefig(payload['out_pdf'])
plt.close(fig)
"""
        subprocess.run(
            [plot_python, "-c", helper_code, payload_path],
            check=True,
        )
    finally:
        if payload_path and os.path.exists(payload_path):
            os.remove(payload_path)


def plot_snr_curves(
    plot_python: str,
    method_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    xmax: int,
    out_png: str,
    out_pdf: str,
) -> None:
    try:
        plot_snr_curves_matplotlib(method_data, title, xmax, out_png, out_pdf)
    except Exception as exc:
        print(
            "[WARN] Local matplotlib plotting failed "
            f"({type(exc).__name__}: {exc}). Falling back to subprocess: {plot_python}",
            flush=True,
        )
        plot_snr_curves_via_subprocess(
            plot_python=plot_python,
            method_data=method_data,
            title=title,
            xmax=xmax,
            out_png=out_png,
            out_pdf=out_pdf,
        )


def main() -> None:
    args = parse_args()

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be >= 1")
    if args.max_images == 0 or args.max_images < -1:
        raise ValueError("--max_images must be -1 or a positive integer")

    ddpm_dir = os.path.abspath(args.ddpm_dir)
    iso_dir = os.path.abspath(args.iso_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    validate_timestep_folders(ddpm_dir, TIMESTEPS)
    validate_timestep_folders(iso_dir, TIMESTEPS)

    ddpm_ids_t0 = set(list_png_filenames(os.path.join(ddpm_dir, "t_0")))
    iso_ids_t0 = set(list_png_filenames(os.path.join(iso_dir, "t_0")))

    if args.use_common_ids == 1:
        base_ids = sorted(ddpm_ids_t0.intersection(iso_ids_t0), key=filename_sort_key)
    else:
        base_ids = sorted(ddpm_ids_t0.union(iso_ids_t0), key=filename_sort_key)

    if not base_ids:
        raise RuntimeError("No image IDs available after applying ID selection rules.")

    requested_max = args.max_images
    if requested_max == -1:
        selected_ids = base_ids
    else:
        selected_ids = base_ids[: min(requested_max, len(base_ids))]
    max_images_effective = len(selected_ids)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for GPU SNR compute. Install torch in the active environment."
        ) from exc

    devices = parse_devices(args.devices)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        visible_cuda_count = torch.cuda.device_count()
        for device_idx in devices:
            if device_idx >= visible_cuda_count:
                raise RuntimeError(
                    f"Requested cuda:{device_idx} in --devices, but only "
                    f"{visible_cuda_count} visible device(s) are available."
                )
    elif args.allow_cpu_fallback == 1:
        print("[WARN] CUDA unavailable. Using CPU fallback for this run.", flush=True)
    else:
        raise RuntimeError(
            "CUDA is required, but torch.cuda.is_available() is False. "
            "If this is a restricted environment, rerun with --allow_cpu_fallback 1 for a non-GPU sanity run."
        )

    summary_csv_path = os.path.join(
        outdir, f"snr_summary_{args.dataset_name}_ddpm_vs_iso.csv"
    )
    per_image_csv_path = os.path.join(
        outdir, f"snr_per_image_{args.dataset_name}_ddpm_vs_iso.csv"
    )
    plot_prefix = f"snr_{args.dataset_name}_ddpm_vs_iso"
    plot_100_png = os.path.join(outdir, f"{plot_prefix}_t0_to_t100.png")
    plot_100_pdf = os.path.join(outdir, f"{plot_prefix}_t0_to_t100.pdf")
    plot_1000_png = os.path.join(outdir, f"{plot_prefix}_t0_to_t1000.png")
    plot_1000_pdf = os.path.join(outdir, f"{plot_prefix}_t0_to_t1000.pdf")

    print(f"[INFO] timesteps used: {TIMESTEPS}")
    print(f"[INFO] number of ids used: {len(selected_ids)}")
    print(
        f"[INFO] max_images effective: requested={requested_max}, effective={max_images_effective}"
    )
    print(f"[INFO] use_common_ids: {bool(args.use_common_ids)}")

    worker_splits = split_filenames(selected_ids, args.num_workers)
    worker_devices = [devices[i % len(devices)] for i in range(args.num_workers)]
    for worker_idx, (split_ids, device_idx) in enumerate(zip(worker_splits, worker_devices)):
        gpu_label = f"cuda:{device_idx}" if use_cuda else "cpu"
        print(
            f"[INFO] worker {worker_idx}: ids={len(split_ids)}, gpu={gpu_label}",
            flush=True,
        )

    print(f"[INFO] output summary csv: {summary_csv_path}")
    if args.write_per_image_csv == 1:
        print(f"[INFO] output per-image csv: {per_image_csv_path}")
    print(f"[INFO] output plot: {plot_100_png}")
    print(f"[INFO] output plot: {plot_100_pdf}")
    print(f"[INFO] output plot: {plot_1000_png}")
    print(f"[INFO] output plot: {plot_1000_pdf}")

    temp_dir = tempfile.mkdtemp(prefix="snr_tmp_", dir=outdir)
    try:
        worker_args = []
        for worker_idx, filenames in enumerate(worker_splits):
            worker_args.append(
                (
                    worker_idx,
                    worker_devices[worker_idx],
                    filenames,
                    ddpm_dir,
                    iso_dir,
                    TIMESTEPS,
                    args.write_per_image_csv == 1,
                    temp_dir,
                    use_cuda,
                )
            )

        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.num_workers) as pool:
                worker_results = pool.starmap(worker_compute, worker_args)
        except (PermissionError, OSError) as exc:
            print(
                "[WARN] Multiprocessing pool unavailable in this environment "
                f"({type(exc).__name__}: {exc}). Running worker shards sequentially.",
                flush=True,
            )
            worker_results = [worker_compute(*worker_arg) for worker_arg in worker_args]

        worker_results.sort(key=lambda x: x["worker_idx"])

        merged_stats: Dict[str, Dict[int, Tuple[int, float, float]]] = {
            method: {t: (0, 0.0, 0.0) for t in TIMESTEPS} for method in METHODS
        }
        for result in worker_results:
            result_stats = result["stats"]
            for method in METHODS:
                for t in TIMESTEPS:
                    merged_stats[method][t] = merge_welford(
                        merged_stats[method][t], tuple(result_stats[method][t])
                    )

        summary_rows: List[Dict[str, object]] = []
        method_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for method in METHODS:
            means = []
            ci95s = []
            for t in TIMESTEPS:
                n, mean, m2 = merged_stats[method][t]
                if n == 0:
                    std = float("nan")
                    ci95 = float("nan")
                    mean_out = float("nan")
                elif n == 1:
                    std = 0.0
                    ci95 = 0.0
                    mean_out = mean
                else:
                    var = m2 / (n - 1)
                    std = math.sqrt(max(var, 0.0))
                    ci95 = 1.96 * std / math.sqrt(n)
                    mean_out = mean

                summary_rows.append(
                    {
                        "method": method,
                        "timestep": t,
                        "n": n,
                        "snr_mean_db": mean_out,
                        "snr_std_db": std,
                        "snr_ci95_db": ci95,
                    }
                )
                means.append(mean_out)
                ci95s.append(ci95)

            method_arrays[METHOD_LABELS[method]] = (
                np.asarray(TIMESTEPS, dtype=np.float64),
                np.asarray(means, dtype=np.float64),
                np.asarray(ci95s, dtype=np.float64),
            )

        write_summary_csv(summary_csv_path, summary_rows)

        if args.write_per_image_csv == 1:
            combine_per_image_csvs(
                [result["per_image_csv"] for result in worker_results], per_image_csv_path
            )

        plot_title = f"{args.dataset_title} — SNR over diffusion timesteps"
        plot_snr_curves(args.plot_python, method_arrays, plot_title, 100, plot_100_png, plot_100_pdf)
        plot_snr_curves(args.plot_python, method_arrays, plot_title, 1000, plot_1000_png, plot_1000_pdf)

        print("[INFO] Completed SNR evaluation.")
        print(f"[INFO] Wrote: {summary_csv_path}")
        if args.write_per_image_csv == 1:
            print(f"[INFO] Wrote: {per_image_csv_path}")
        print(f"[INFO] Wrote: {plot_100_png}")
        print(f"[INFO] Wrote: {plot_100_pdf}")
        print(f"[INFO] Wrote: {plot_1000_png}")
        print(f"[INFO] Wrote: {plot_1000_pdf}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
