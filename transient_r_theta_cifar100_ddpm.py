#!/usr/bin/env python3
"""
Compute transient r_Theta for CIFAR-100 reverse DDPM sampling.

Metric (per image, per timestep):
    r_Theta_t = ||noise_target - eps_theta(x_t, t)||_2 / d

where:
- d is the flattened sample dimension (C*H*W),
- eps_theta is the model-predicted noise in the reverse path,
- noise_target is the sampled reverse noise for t>0 and 0 for t=0.

Outputs:
- Per-image values: r_Theta_<t>_per_image.npy and .csv
- Per-timestep mean: r_Theta_<t>.txt
- Summary table: r_theta_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Allow running this script from any working directory.
_script_dir = Path(__file__).resolve().parent
REPO_ROOT = _script_dir if (_script_dir / "models").is_dir() else _script_dir.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import get_beta_schedule


def dict2namespace(d: Dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict2namespace(cfg)


def resolve_config_path(user_config: str | None, ckpt_path: str) -> str:
    if user_config:
        p = Path(user_config)
        if not p.is_file():
            raise FileNotFoundError(f"--config not found: {p}")
        return str(p)

    ckpt_dir_config = Path(ckpt_path).resolve().parent / "config.yml"
    if ckpt_dir_config.is_file():
        return str(ckpt_dir_config)

    fallback_candidates = [
        Path("configs/cifar100_transient_50k.yml"),
        Path("configs/cifar100.yml"),
    ]
    for candidate in fallback_candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve config path. Provide --config explicitly or keep a config.yml next to ckpt."
    )


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_timestamps(timestamps_arg: str, max_timestep: int) -> List[int]:
    if timestamps_arg.strip().lower() == "auto":
        timestamps = [1000] + list(range(900, 199, -100)) + list(range(150, 9, -10)) + [0]
    else:
        parts = [p.strip() for p in timestamps_arg.split(",") if p.strip()]
        if not parts:
            raise ValueError("No timestamps parsed from --timestamps.")
        timestamps = [int(p) for p in parts]

    deduped: List[int] = []
    seen = set()
    for t in timestamps:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)

    bad = [t for t in deduped if t < 0 or t > max_timestep]
    if bad:
        raise ValueError(
            f"Invalid timestamps {bad}. Allowed range is [0, {max_timestep}]."
        )
    return deduped


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device_arg}) but CUDA is unavailable.")
    return dev


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_model_from_checkpoint(
    config: argparse.Namespace, ckpt_path: str, device: torch.device, use_ema: bool = True
) -> torch.nn.Module:
    states = torch.load(ckpt_path, map_location=device)
    model = Model(config).to(device)
    model.load_state_dict(states[0], strict=True)

    if use_ema and getattr(config.model, "ema", False) and len(states) > 4:
        ema_helper = EMAHelper(mu=getattr(config.model, "ema_rate", 0.9999))
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)

    model.eval()
    return model


def build_betas(config: argparse.Namespace, device: torch.device) -> torch.Tensor:
    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    return torch.from_numpy(betas_np).float().to(device)


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def save_per_image_csv(path: str, values: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "r_theta"])
        for idx, val in enumerate(values):
            writer.writerow([idx, f"{float(val):.10f}"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transient r_Theta for CIFAR-100 reverse DDPM sampling."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_images_per_t", type=int, default=10000)
    parser.add_argument(
        "--timestamps",
        type=str,
        default="auto",
        help="Comma-separated list, or 'auto' for: 1000,900..200,150..10,0",
    )
    parser.add_argument("--use_ema", type=str2bool, default=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_images_per_t <= 0:
        raise ValueError("--num_images_per_t must be positive.")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    os.makedirs(args.outdir, exist_ok=True)

    config_path = resolve_config_path(args.config, args.ckpt)
    config = load_config(config_path)
    device = select_device(args.device)
    set_seed(args.seed, device)

    model = load_model_from_checkpoint(config, args.ckpt, device=device, use_ema=args.use_ema)
    betas = build_betas(config, device=device)

    num_timesteps = int(betas.shape[0])
    timestamps = parse_timestamps(args.timestamps, max_timestep=num_timesteps)
    target_set = set(timestamps)

    channels = int(config.data.channels)
    image_size = int(config.data.image_size)
    total = int(args.num_images_per_t)

    seq = list(range(0, num_timesteps))
    seq_next = [-1] + list(seq[:-1])
    values: Dict[int, List[float]] = {t: [] for t in timestamps}

    remaining = total
    n_rounds = math.ceil(total / args.batch_size)

    print(f"[INFO] config: {config_path}")
    print(f"[INFO] ckpt:   {args.ckpt}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] timestamps: {timestamps}")
    print(f"[INFO] images per timestep: {total}")

    with torch.no_grad():
        for _ in tqdm(range(n_rounds), desc="Reverse DDPM sampling for r_Theta"):
            n = min(args.batch_size, remaining)
            x = torch.randn(n, channels, image_size, image_size, device=device)

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                next_t = torch.full((n,), j, device=device, dtype=torch.long)

                at = compute_alpha(betas, t)
                atm1 = compute_alpha(betas, next_t)
                beta_t = 1 - at / atm1

                eps_theta = model(x, t.float())
                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1.0).sqrt() * eps_theta
                x0_from_e = torch.clamp(x0_from_e, -1.0, 1.0)

                mean_eps = (
                    (atm1.sqrt() * beta_t) * x0_from_e + ((1.0 - beta_t).sqrt() * (1.0 - atm1)) * x
                ) / (1.0 - at)

                noise = torch.randn_like(x)
                mask = (t != 0).float().view(-1, 1, 1, 1)
                noise_target = noise * mask
                sample = mean_eps + mask * torch.exp(0.5 * beta_t.log()) * noise

                timestep_label = num_timesteps if i == (num_timesteps - 1) else i
                if timestep_label in target_set:
                    e = eps_theta.reshape(n, -1)                 # (n, d)
                    per_image = (e ** 2).sum(dim=1) / float(e.shape[1])   # ||eps_theta||_2^2 / d
                    out = per_image.detach().cpu().tolist()

                    cur = values[timestep_label]
                    take = min(len(out), total - len(cur))
                    if take > 0:
                        cur.extend(out[:take])

                x = sample

            remaining -= n

    for t in timestamps:
        if len(values[t]) != total:
            raise RuntimeError(
                f"Timestep {t} has {len(values[t])} values, expected {total}. "
                "This indicates an incomplete accumulation."
            )

    summary_rows = []
    for t in timestamps:
        arr = np.asarray(values[t], dtype=np.float32)
        mean_val = float(arr.mean())

        per_image_npy = os.path.join(args.outdir, f"r_Theta_{t}_per_image.npy")
        per_image_csv = os.path.join(args.outdir, f"r_Theta_{t}_per_image.csv")
        mean_txt = os.path.join(args.outdir, f"r_Theta_{t}.txt")

        np.save(per_image_npy, arr)
        save_per_image_csv(per_image_csv, arr)
        with open(mean_txt, "w", encoding="utf-8") as f:
            f.write(f"{mean_val:.10f}\n")

        summary_rows.append(
            {
                "name": f"r_Theta_{t}",
                "timestep": int(t),
                "num_images": int(total),
                "mean": f"{mean_val:.10f}",
            }
        )

        print(f"[OK] r_Theta_{t} = {mean_val:.10f}")

    summary_csv = os.path.join(args.outdir, "r_theta_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "timestep", "num_images", "mean"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[DONE] Wrote outputs to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
