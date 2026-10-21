"""Record r_theta = ||eps_theta(x_t, t)||_2^2 / (C*H*W) on full DDPM paths.

Uses an already loaded evaluation model; performs no checkpoint loading.
Outputs and timestep labels match transient_r_theta_cifar10_ddpm.py.
"""

import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

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


def save_r_theta(model, betas, config, outdir, num_images_per_t=10000,
                 batch_size=256, timestamps_arg="auto", seed=1234):
    """Run independent DDPM trajectories and save per-image values and means.

    Preserve the caller's torch RNG state so image sampling remains unaffected.
    Call once on rank zero when sampling with multiple processes.
    """
    if num_images_per_t <= 0 or batch_size <= 0:
        raise ValueError("num_images_per_t and batch_size must be positive")
    device = betas.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            _save_r_theta(model, betas, config, outdir, num_images_per_t,
                          batch_size, timestamps_arg)
    finally:
        model.train(was_training)


def _save_r_theta(model, betas, config, outdir, num_images_per_t,
                  batch_size, timestamps_arg):
    device = betas.device
    os.makedirs(outdir, exist_ok=True)
    num_timesteps = int(betas.shape[0])
    timestamps = parse_timestamps(timestamps_arg, max_timestep=num_timesteps)
    target_set = set(timestamps)

    channels = int(config.data.channels)
    image_size = int(config.data.image_size)
    total = int(num_images_per_t)

    seq = list(range(0, num_timesteps))
    seq_next = [-1] + list(seq[:-1])
    values: Dict[int, List[float]] = {t: [] for t in timestamps}

    remaining = total
    n_rounds = math.ceil(total / batch_size)

    print(f"[INFO] device: {device}")
    print(f"[INFO] timestamps: {timestamps}")
    print(f"[INFO] images per timestep: {total}")

    with torch.no_grad():
        for _ in tqdm(range(n_rounds), desc="Reverse DDPM sampling for r_Theta"):
            n = min(batch_size, remaining)
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

        per_image_npy = os.path.join(outdir, f"r_Theta_{t}_per_image.npy")
        per_image_csv = os.path.join(outdir, f"r_Theta_{t}_per_image.csv")
        mean_txt = os.path.join(outdir, f"r_Theta_{t}.txt")

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

    summary_csv = os.path.join(outdir, "r_theta_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "timestep", "num_images", "mean"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[DONE] Wrote outputs to: {outdir}")
