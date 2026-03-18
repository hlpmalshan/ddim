#!/usr/bin/env python3
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EPS = 1e-8
DEFAULT_TIMESTEPS = "0,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000"
METHODS = ("ddpm", "iso")
METHOD_DISPLAY = {
    "ddpm": "DDPM",
    "iso": "ISO diffusion",
}
METRICS = ("snr", "psnr", "lpips")
METRIC_LABEL = {
    "snr": "SNR",
    "psnr": "PSNR",
    "lpips": "LPIPS",
}
METRIC_YLABEL = {
    "snr": "SNR (dB)",
    "psnr": "PSNR (dB)",
    "lpips": "LPIPS (↓)",
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
        return (self.n, self.mean, self.m2)


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
    m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)
    return (n, mean, m2)


def finalize_stats(state: Tuple[int, float, float]) -> Tuple[int, float, float, float]:
    n, mean, m2 = state
    if n == 0:
        return (0, float("nan"), float("nan"), float("nan"))
    if n == 1:
        return (1, float(mean), 0.0, 0.0)

    std = math.sqrt(max(m2 / (n - 1), 0.0))
    ci95 = 1.96 * std / math.sqrt(n)
    return (n, float(mean), float(std), float(ci95))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot SNR/PSNR/LPIPS for transient diffusion results on GPU."
    )
    parser.add_argument("--ddpm_dir", type=str, required=True)
    parser.add_argument("--iso_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_title", type=str, required=True)
    parser.add_argument("--timesteps", type=str, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--max_images", type=int, default=-1, help="-1 means all")
    parser.add_argument("--use_common_ids", type=int, choices=(0, 1), default=1)
    parser.add_argument("--write_per_image_csv", type=int, choices=(0, 1), default=0)
    parser.add_argument("--lpips_net", type=str, choices=("alex", "vgg"), default="alex")
    parser.add_argument(
        "--plot_skip_t0",
        type=int,
        choices=(0, 1),
        default=1,
        help="Skip t=0 in plots only (t=0 remains in CSV).",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1",
        help="Comma-separated visible CUDA device indices after CUDA_VISIBLE_DEVICES mapping.",
    )
    parser.add_argument("--allow_cpu_fallback", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--lpips_pnet_rand", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_timesteps(timestep_str: str) -> List[int]:
    values: List[int] = []
    for chunk in timestep_str.split(","):
        token = chunk.strip()
        if not token:
            continue
        t = int(token)
        if t < 0:
            raise ValueError(f"Invalid timestep: {t}")
        values.append(t)
    if not values:
        raise ValueError("No timesteps were provided.")

    deduped: List[int] = []
    seen = set()
    for t in values:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def parse_devices(devices_arg: str) -> List[int]:
    devices: List[int] = []
    for chunk in devices_arg.split(","):
        token = chunk.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            raise ValueError(f"Invalid device index: {idx}")
        devices.append(idx)
    if not devices:
        raise ValueError("No devices provided in --devices.")
    return devices


def filename_sort_key(filename: str) -> Tuple[int, object]:
    stem, _ = os.path.splitext(filename)
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def list_png_filenames(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = [name for name in os.listdir(folder) if name.lower().endswith(".png")]
    files.sort(key=filename_sort_key)
    return files


def select_available_timesteps(ddpm_dir: str, iso_dir: str, requested: List[int]) -> List[int]:
    used: List[int] = []
    for t in requested:
        ddpm_t = os.path.join(ddpm_dir, f"t_{t}")
        iso_t = os.path.join(iso_dir, f"t_{t}")
        ddpm_ok = os.path.isdir(ddpm_t)
        iso_ok = os.path.isdir(iso_t)
        if ddpm_ok and iso_ok:
            used.append(t)
        else:
            missing = []
            if not ddpm_ok:
                missing.append(ddpm_t)
            if not iso_ok:
                missing.append(iso_t)
            print(f"[WARN] timestep t={t} skipped (missing: {', '.join(missing)})", flush=True)
    return used


def split_ids_evenly(ids: List[str], num_workers: int) -> List[List[str]]:
    return [ids[i::num_workers] for i in range(num_workers)]


def load_png_to_device(path: str, device: object, torch_module: object) -> object:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch_module.from_numpy(arr).to(device=device, non_blocking=False)


def worker_compute(
    worker_idx: int,
    device_idx: int,
    filenames: List[str],
    ddpm_dir: str,
    iso_dir: str,
    timesteps: List[int],
    write_per_image_csv: bool,
    per_image_path: str,
    lpips_net: str,
    lpips_pnet_rand: bool,
    use_cuda: bool,
) -> Dict[str, object]:
    import lpips
    import torch

    torch.set_grad_enabled(False)

    if use_cuda:
        visible_cuda = torch.cuda.device_count()
        if device_idx >= visible_cuda:
            raise RuntimeError(
                f"Worker {worker_idx}: requested cuda:{device_idx}, but only {visible_cuda} visible CUDA device(s)."
            )
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
        device_name = f"cuda:{device_idx}"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    try:
        lpips_model = lpips.LPIPS(net=lpips_net, pnet_rand=lpips_pnet_rand).to(device).eval()
    except Exception as exc:
        if lpips_pnet_rand:
            raise
        raise RuntimeError(
            "LPIPS backbone weights are unavailable offline. Cache torchvision backbone weights or "
            "rerun with --lpips_pnet_rand 1 for debug-only execution."
        ) from exc
    for param in lpips_model.parameters():
        param.requires_grad_(False)

    method_roots = {
        "ddpm": ddpm_dir,
        "iso": iso_dir,
    }

    stats: Dict[str, Dict[str, Dict[int, WelfordState]]] = {
        method: {
            metric: {t: WelfordState() for t in timesteps} for metric in METRICS
        }
        for method in METHODS
    }

    per_image_file = None
    per_image_writer = None
    if write_per_image_csv:
        per_image_file = open(per_image_path, "w", newline="")
        per_image_writer = csv.writer(per_image_file)
        per_image_writer.writerow(["method", "timestep", "filename", "snr_db", "psnr_db", "lpips"])

    with torch.no_grad():
        total_ids = len(filenames)
        for index, filename in enumerate(filenames, start=1):
            for method, root in method_roots.items():
                x0_path = os.path.join(root, "t_0", filename)
                if not os.path.isfile(x0_path):
                    continue

                x0 = load_png_to_device(x0_path, device, torch)
                signal = torch.mean(x0 * x0)
                x0_lpips = x0 * 2.0 - 1.0

                for t in timesteps:
                    xt_path = os.path.join(root, f"t_{t}", filename)
                    if not os.path.isfile(xt_path):
                        continue

                    xt = load_png_to_device(xt_path, device, torch)
                    if xt.shape != x0.shape:
                        del xt
                        continue

                    diff = xt - x0
                    mse = torch.mean(diff * diff)

                    snr_db = 10.0 * torch.log10(signal / (mse + EPS))
                    psnr_db = 10.0 * torch.log10(1.0 / (mse + EPS))
                    xt_lpips = xt * 2.0 - 1.0
                    lpips_val = lpips_model(x0_lpips.unsqueeze(0), xt_lpips.unsqueeze(0))

                    snr_value = float(snr_db.item())
                    psnr_value = float(psnr_db.item())
                    lpips_value = float(lpips_val.reshape(-1)[0].item())

                    if math.isfinite(snr_value) and math.isfinite(psnr_value) and math.isfinite(lpips_value):
                        stats[method]["snr"][t].update(snr_value)
                        stats[method]["psnr"][t].update(psnr_value)
                        stats[method]["lpips"][t].update(lpips_value)

                        if per_image_writer is not None:
                            per_image_writer.writerow(
                                [
                                    method,
                                    t,
                                    filename,
                                    f"{snr_value:.10f}",
                                    f"{psnr_value:.10f}",
                                    f"{lpips_value:.10f}",
                                ]
                            )

                    del xt, diff, mse, snr_db, psnr_db, xt_lpips, lpips_val

                del x0, signal, x0_lpips

            if index % 200 == 0:
                print(
                    f"[WORKER {worker_idx}] processed {index}/{total_ids} ids on {device_name}",
                    flush=True,
                )

    if per_image_file is not None:
        per_image_file.close()

    if use_cuda:
        torch.cuda.synchronize(device)

    serialized_stats: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method in METHODS:
        serialized_stats[method] = {}
        for metric in METRICS:
            serialized_stats[method][metric] = {}
            for t in timesteps:
                n, mean, m2 = stats[method][metric][t].as_tuple()
                serialized_stats[method][metric][str(t)] = [n, mean, m2]

    return {
        "worker_idx": worker_idx,
        "device": device_name,
        "stats": serialized_stats,
        "per_image_csv": per_image_path if write_per_image_csv else "",
    }


def worker_entry(
    worker_idx: int,
    device_idx: int,
    filenames: List[str],
    ddpm_dir: str,
    iso_dir: str,
    timesteps: List[int],
    write_per_image_csv: bool,
    per_image_path: str,
    lpips_net: str,
    lpips_pnet_rand: bool,
    use_cuda: bool,
    result_json_path: str,
) -> None:
    try:
        result = worker_compute(
            worker_idx=worker_idx,
            device_idx=device_idx,
            filenames=filenames,
            ddpm_dir=ddpm_dir,
            iso_dir=iso_dir,
            timesteps=timesteps,
            write_per_image_csv=write_per_image_csv,
            per_image_path=per_image_path,
            lpips_net=lpips_net,
            lpips_pnet_rand=lpips_pnet_rand,
            use_cuda=use_cuda,
        )
        payload = {"ok": True, "result": result}
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    with open(result_json_path, "w") as handle:
        json.dump(payload, handle)


def run_workers(
    worker_ids: List[List[str]],
    worker_devices: List[int],
    ddpm_dir: str,
    iso_dir: str,
    timesteps: List[int],
    write_per_image_csv: bool,
    lpips_net: str,
    lpips_pnet_rand: bool,
    temp_dir: str,
    use_cuda: bool,
) -> List[Dict[str, object]]:
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    result_paths: List[str] = []

    for worker_idx, ids_chunk in enumerate(worker_ids):
        result_path = os.path.join(temp_dir, f"worker_{worker_idx}_result.json")
        per_image_path = os.path.join(temp_dir, f"worker_{worker_idx}_per_image.csv")
        result_paths.append(result_path)
        process = ctx.Process(
            target=worker_entry,
            args=(
                worker_idx,
                worker_devices[worker_idx],
                ids_chunk,
                ddpm_dir,
                iso_dir,
                timesteps,
                write_per_image_csv,
                per_image_path,
                lpips_net,
                lpips_pnet_rand,
                use_cuda,
                result_path,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results: List[Dict[str, object]] = []
    for worker_idx, result_path in enumerate(result_paths):
        if not os.path.isfile(result_path):
            raise RuntimeError(f"Worker {worker_idx} did not produce a result file: {result_path}")
        with open(result_path, "r") as handle:
            payload = json.load(handle)
        if not payload.get("ok", False):
            raise RuntimeError(
                f"Worker {worker_idx} failed.\n{payload.get('error', 'Unknown error')}\n{payload.get('traceback', '')}"
            )
        results.append(payload["result"])

    results.sort(key=lambda x: int(x["worker_idx"]))
    return results


def merge_worker_stats(
    worker_results: List[Dict[str, object]],
    timesteps: List[int],
) -> Dict[str, Dict[str, Dict[int, Tuple[int, float, float]]]]:
    merged: Dict[str, Dict[str, Dict[int, Tuple[int, float, float]]]] = {
        method: {metric: {t: (0, 0.0, 0.0) for t in timesteps} for metric in METRICS}
        for method in METHODS
    }

    for result in worker_results:
        result_stats = result["stats"]
        for method in METHODS:
            for metric in METRICS:
                for t in timesteps:
                    b = tuple(result_stats[method][metric][str(t)])
                    merged[method][metric][t] = merge_welford(merged[method][metric][t], b)
    return merged


def write_summary_csv(summary_csv: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "timestep",
        "n",
        "snr_mean_db",
        "snr_std_db",
        "snr_ci95_db",
        "psnr_mean_db",
        "psnr_std_db",
        "psnr_ci95_db",
        "lpips_mean",
        "lpips_std",
        "lpips_ci95",
    ]
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def combine_per_image_csvs(per_image_csv: str, worker_results: List[Dict[str, object]]) -> None:
    with open(per_image_csv, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["method", "timestep", "filename", "snr_db", "psnr_db", "lpips"])
        for result in worker_results:
            chunk_csv = result.get("per_image_csv", "")
            if not chunk_csv or not os.path.isfile(chunk_csv):
                continue
            with open(chunk_csv, "r", newline="") as in_handle:
                reader = csv.reader(in_handle)
                _ = next(reader, None)
                for row in reader:
                    writer.writerow(row)


def plot_metric(
    metric_name: str,
    dataset_name: str,
    dataset_title: str,
    outdir: str,
    plot_skip_t0: bool,
    metric_series: Dict[str, Dict[str, np.ndarray]],
) -> List[str]:
    saved_paths: List[str] = []
    for xmax in (100, 1000):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method_label, series in metric_series.items():
            ts = series["timesteps"]
            mean = series["mean"]
            ci = series["ci95"]

            mask = ts <= xmax
            if plot_skip_t0:
                mask &= ts != 0
            ts = ts[mask]
            mean = mean[mask]
            ci = ci[mask]

            valid = np.isfinite(ts) & np.isfinite(mean) & np.isfinite(ci)
            ts = ts[valid]
            mean = mean[valid]
            ci = ci[valid]
            if ts.size == 0:
                continue

            ax.plot(ts, mean, marker="o", label=method_label)
            ax.fill_between(ts, mean - ci, mean + ci, alpha=0.2)

        ax.set_xlabel("Diffusion timestep (t)")
        ax.set_ylabel(METRIC_YLABEL[metric_name])
        ax.set_title(f"{dataset_title} — {METRIC_LABEL[metric_name]} over diffusion timesteps")
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        png_path = os.path.join(outdir, f"{metric_name}_{dataset_name}_ddpm_vs_iso_t0_to_t{xmax}.png")
        pdf_path = os.path.join(outdir, f"{metric_name}_{dataset_name}_ddpm_vs_iso_t0_to_t{xmax}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.extend([png_path, pdf_path])

    return saved_paths


def main() -> None:
    args = parse_args()

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be >= 1")
    if args.max_images < -1 or args.max_images == 0:
        raise ValueError("--max_images must be -1 or a positive integer")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required. Install torch in the selected environment.") from exc

    try:
        import lpips  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "lpips is required. Install with: /data/diffusion/envs/ddim/bin/python -m pip install lpips"
        ) from exc

    ddpm_dir = os.path.abspath(args.ddpm_dir)
    iso_dir = os.path.abspath(args.iso_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    timesteps_requested = parse_timesteps(args.timesteps)
    timesteps_used = select_available_timesteps(ddpm_dir, iso_dir, timesteps_requested)
    if not timesteps_used:
        raise RuntimeError("No valid timesteps found in both methods.")

    ddpm_ids_t0 = set(list_png_filenames(os.path.join(ddpm_dir, "t_0")))
    iso_ids_t0 = set(list_png_filenames(os.path.join(iso_dir, "t_0")))
    if args.use_common_ids == 1:
        ids = sorted(ddpm_ids_t0.intersection(iso_ids_t0), key=filename_sort_key)
    else:
        ids = sorted(ddpm_ids_t0.union(iso_ids_t0), key=filename_sort_key)

    if not ids:
        raise RuntimeError("No image IDs found after applying pairing rules.")

    requested_max = args.max_images
    if requested_max != -1:
        ids = ids[: min(requested_max, len(ids))]
    max_images_effective = len(ids)

    devices = parse_devices(args.devices)
    worker_ids = split_ids_evenly(ids, args.num_workers)
    worker_devices = [devices[i % len(devices)] for i in range(args.num_workers)]

    use_cuda = torch.cuda.is_available()
    if not use_cuda and args.allow_cpu_fallback != 1:
        raise RuntimeError(
            "CUDA is required but not available. Ensure GPU visibility and rerun with "
            "CUDA_VISIBLE_DEVICES=6,7 (or set --allow_cpu_fallback 1 for local non-GPU debug only)."
        )

    if use_cuda:
        cuda_count = torch.cuda.device_count()
        for device_idx in devices:
            if device_idx >= cuda_count:
                raise RuntimeError(
                    f"--devices includes cuda:{device_idx}, but only {cuda_count} visible CUDA device(s)."
                )
    else:
        print("[WARN] CUDA unavailable; using CPU fallback for local debug run.", flush=True)

    summary_csv = os.path.join(outdir, f"metrics_summary_{args.dataset_name}_ddpm_vs_iso.csv")
    per_image_csv = os.path.join(outdir, f"metrics_per_image_{args.dataset_name}_ddpm_vs_iso.csv")

    print(f"[INFO] timesteps used: {timesteps_used}", flush=True)
    print(f"[INFO] number of ids used: {len(ids)}", flush=True)
    print(f"[INFO] max_images effective: requested={requested_max}, effective={max_images_effective}", flush=True)
    for worker_idx in range(args.num_workers):
        device_label = f"cuda:{worker_devices[worker_idx]}" if use_cuda else "cpu"
        print(
            f"[INFO] worker {worker_idx}: ids={len(worker_ids[worker_idx])}, gpu={device_label}",
            flush=True,
        )

    print(f"[INFO] output summary csv: {summary_csv}", flush=True)
    if args.write_per_image_csv == 1:
        print(f"[INFO] output per-image csv: {per_image_csv}", flush=True)

    temp_dir = tempfile.mkdtemp(prefix="metrics_tmp_", dir=outdir)
    try:
        worker_results = run_workers(
            worker_ids=worker_ids,
            worker_devices=worker_devices,
            ddpm_dir=ddpm_dir,
            iso_dir=iso_dir,
            timesteps=timesteps_used,
            write_per_image_csv=args.write_per_image_csv == 1,
            lpips_net=args.lpips_net,
            lpips_pnet_rand=args.lpips_pnet_rand == 1,
            temp_dir=temp_dir,
            use_cuda=use_cuda,
        )

        merged = merge_worker_stats(worker_results, timesteps_used)

        summary_rows: List[Dict[str, object]] = []
        metric_plot_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
            metric: {} for metric in METRICS
        }

        for method in METHODS:
            snr_mean_vals = []
            snr_ci_vals = []
            psnr_mean_vals = []
            psnr_ci_vals = []
            lpips_mean_vals = []
            lpips_ci_vals = []

            for t in timesteps_used:
                snr_n, snr_mean, snr_std, snr_ci = finalize_stats(merged[method]["snr"][t])
                psnr_n, psnr_mean, psnr_std, psnr_ci = finalize_stats(merged[method]["psnr"][t])
                lpips_n, lpips_mean, lpips_std, lpips_ci = finalize_stats(merged[method]["lpips"][t])

                n_values = [snr_n, psnr_n, lpips_n]
                n = min(n_values)
                if max(n_values) != min(n_values):
                    print(
                        f"[WARN] count mismatch for method={method}, t={t}: "
                        f"snr_n={snr_n}, psnr_n={psnr_n}, lpips_n={lpips_n}",
                        flush=True,
                    )

                summary_rows.append(
                    {
                        "method": method,
                        "timestep": t,
                        "n": n,
                        "snr_mean_db": snr_mean,
                        "snr_std_db": snr_std,
                        "snr_ci95_db": snr_ci,
                        "psnr_mean_db": psnr_mean,
                        "psnr_std_db": psnr_std,
                        "psnr_ci95_db": psnr_ci,
                        "lpips_mean": lpips_mean,
                        "lpips_std": lpips_std,
                        "lpips_ci95": lpips_ci,
                    }
                )

                snr_mean_vals.append(snr_mean)
                snr_ci_vals.append(snr_ci)
                psnr_mean_vals.append(psnr_mean)
                psnr_ci_vals.append(psnr_ci)
                lpips_mean_vals.append(lpips_mean)
                lpips_ci_vals.append(lpips_ci)

            method_label = METHOD_DISPLAY[method]
            ts_np = np.asarray(timesteps_used, dtype=np.float64)
            metric_plot_data["snr"][method_label] = {
                "timesteps": ts_np,
                "mean": np.asarray(snr_mean_vals, dtype=np.float64),
                "ci95": np.asarray(snr_ci_vals, dtype=np.float64),
            }
            metric_plot_data["psnr"][method_label] = {
                "timesteps": ts_np,
                "mean": np.asarray(psnr_mean_vals, dtype=np.float64),
                "ci95": np.asarray(psnr_ci_vals, dtype=np.float64),
            }
            metric_plot_data["lpips"][method_label] = {
                "timesteps": ts_np,
                "mean": np.asarray(lpips_mean_vals, dtype=np.float64),
                "ci95": np.asarray(lpips_ci_vals, dtype=np.float64),
            }

        write_summary_csv(summary_csv, summary_rows)
        print(f"[INFO] saved summary csv: {summary_csv}", flush=True)

        if args.write_per_image_csv == 1:
            combine_per_image_csvs(per_image_csv, worker_results)
            print(f"[INFO] saved per-image csv: {per_image_csv}", flush=True)

        all_plot_paths: List[str] = []
        for metric_name in METRICS:
            metric_paths = plot_metric(
                metric_name=metric_name,
                dataset_name=args.dataset_name,
                dataset_title=args.dataset_title,
                outdir=outdir,
                plot_skip_t0=args.plot_skip_t0 == 1,
                metric_series=metric_plot_data[metric_name],
            )
            all_plot_paths.extend(metric_paths)

        for path in all_plot_paths:
            print(f"[INFO] saved plot: {path}", flush=True)

        print("[INFO] done.", flush=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
