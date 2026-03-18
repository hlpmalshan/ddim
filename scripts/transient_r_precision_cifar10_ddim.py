#!/usr/bin/env python3
"""
Compute transient R-Precision for a CIFAR-10 DDIM checkpoint without saving images.

Repo note:
- No existing R-Precision / CLIP retrieval implementation was found in this repository.
- This script therefore implements a documented proxy that is consistent with repo deps:
  CLIP prompt-retrieval over 10 CIFAR-10 class prompts.

R-Precision proxy used here:
- Prompts: "a photo of a {class}" for the 10 CIFAR-10 classes.
- For each generated sample x_t, compute CLIP similarities to the 10 prompts.
- Convert similarities to probabilities with softmax.
- Per-sample R-Precision proxy = top-1 prompt probability.
- Report mean across all samples for each timestamp.

This is a streaming computation:
- Samples are generated batch-by-batch from DDIM reverse diffusion.
- Only running sums/counts are kept in memory.
- No generated images are written to disk.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import inverse_data_transform
from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import get_beta_schedule


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CSV_FIELDS = [
    "timestamp",
    "num_images",
    "r_precision_mean",
    "ckpt_path",
    "ddim_iso",
    "seed",
    "batch_size",
]


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
        REPO_ROOT / "configs" / "cifar10_transient_50k.yml",
        REPO_ROOT / "configs" / "cifar10.yml",
    ]
    for candidate in fallback_candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve config path. Provide --config explicitly or ensure "
        "a config.yml exists next to the checkpoint."
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
        timestamps = list(range(10, 151, 10)) + list(range(200, 1001, 100))
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

    min_allowed = 0
    max_allowed = max_timestep
    bad = [t for t in deduped if t < min_allowed or t > max_allowed]
    if bad:
        raise ValueError(
            f"Invalid timestamps {bad}. Allowed range is [{min_allowed}, {max_allowed}]."
        )
    return deduped


def infer_ddim_iso(ckpt_path: str) -> str:
    m = re.search(r"ddim_iso_([0-9]+(?:\.[0-9]+)?)", ckpt_path)
    return m.group(1) if m else "unknown"


def load_completed_timestamps(csv_path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "timestamp" not in (reader.fieldnames or []):
            return done
        for row in reader:
            try:
                done.add(int(row["timestamp"]))
            except Exception:
                continue
    return done


def ensure_csv_with_header(csv_path: str) -> None:
    if os.path.isfile(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


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


def build_sqrt_alpha_tables(config: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    betas_np = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas_np).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alpha = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod)

    sqrt_alpha_next = torch.empty_like(sqrt_alpha)
    sqrt_one_minus_alpha_next = torch.empty_like(sqrt_one_minus_alpha)
    sqrt_alpha_next[0] = 1.0
    sqrt_one_minus_alpha_next[0] = 0.0
    sqrt_alpha_next[1:] = sqrt_alpha[:-1]
    sqrt_one_minus_alpha_next[1:] = sqrt_one_minus_alpha[:-1]

    return {
        "sqrt_alpha": sqrt_alpha,
        "sqrt_one_minus_alpha": sqrt_one_minus_alpha,
        "sqrt_alpha_next": sqrt_alpha_next,
        "sqrt_one_minus_alpha_next": sqrt_one_minus_alpha_next,
    }


class CLIPRPrecisionScorer:
    def __init__(self, device: torch.device, model_id: str = "openai/clip-vit-base-patch32") -> None:
        try:
            from transformers import AutoTokenizer, CLIPModel
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for CLIP-based R-Precision. "
                "Install it in your environment."
            ) from exc

        self.device = device
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()

        prompts = [f"a photo of a {cls_name}" for cls_name in CIFAR10_CLASS_NAMES]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**tokenized)
            self.text_features = F.normalize(text_features, dim=-1)

        self.clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)

    @torch.no_grad()
    def score_batch(self, images_01: torch.Tensor) -> torch.Tensor:
        if images_01.device != self.device:
            images_01 = images_01.to(self.device, non_blocking=True)

        images_01 = images_01.clamp(0.0, 1.0)
        clip_pixels = F.interpolate(
            images_01, size=(224, 224), mode="bicubic", align_corners=False
        )
        clip_pixels = (clip_pixels - self.clip_mean) / self.clip_std

        image_features = self.model.get_image_features(pixel_values=clip_pixels)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.model.logit_scale.exp() * image_features @ self.text_features.t()
        probs = logits.softmax(dim=-1)
        return probs.max(dim=-1).values


def update_r_precision_stats(
    *,
    timestamp: int,
    x: torch.Tensor,
    config: argparse.Namespace,
    scorer: CLIPRPrecisionScorer,
    stats: Dict[int, Dict[str, float]],
) -> None:
    images = inverse_data_transform(config, x)
    scores = scorer.score_batch(images)
    stats[timestamp]["sum"] += float(scores.sum().item())
    stats[timestamp]["count"] += int(scores.numel())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transient DDIM CIFAR-10 R-Precision (CLIP prompt-retrieval proxy)."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--timestamps", type=str, default="auto")
    parser.add_argument("--num_images_per_t", type=int, default=50000)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--ddim_iso",
        type=str,
        default=None,
        help="Optional value written to CSV. If unset, inferred from ckpt path.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_images_per_t <= 0:
        raise ValueError("--num_images_per_t must be positive.")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "transient_r_precision.csv")
    ensure_csv_with_header(csv_path)

    config_path = resolve_config_path(args.config, args.ckpt)
    config = load_config(config_path)

    device = select_device(args.device)
    set_seed(args.seed, device)

    num_timesteps = int(config.diffusion.num_diffusion_timesteps)
    timestamps_all = parse_timestamps(args.timestamps, max_timestep=num_timesteps)

    completed = load_completed_timestamps(csv_path) if args.resume else set()
    pending_timestamps = [t for t in timestamps_all if t not in completed]
    if not pending_timestamps:
        print(f"[INFO] Nothing to do. All timestamps already present in {csv_path}.")
        return 0

    print(f"[INFO] config: {config_path}")
    print(f"[INFO] ckpt:   {args.ckpt}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] timestamps pending ({len(pending_timestamps)}): {pending_timestamps}")

    model = load_model_from_checkpoint(config, args.ckpt, device=device, use_ema=True)
    tables = build_sqrt_alpha_tables(config, device=device)
    scorer = CLIPRPrecisionScorer(device=device, model_id=args.clip_model)

    stats: Dict[int, Dict[str, float]] = {
        t: {"sum": 0.0, "count": 0.0} for t in pending_timestamps
    }
    target_set = set(pending_timestamps)

    channels = int(config.data.channels)
    image_size = int(config.data.image_size)
    total = int(args.num_images_per_t)
    remaining = total
    n_rounds = math.ceil(total / args.batch_size)

    sqrt_alpha = tables["sqrt_alpha"]
    sqrt_one_minus_alpha = tables["sqrt_one_minus_alpha"]
    sqrt_alpha_next = tables["sqrt_alpha_next"]
    sqrt_one_minus_alpha_next = tables["sqrt_one_minus_alpha_next"]

    ddim_iso_val = args.ddim_iso if args.ddim_iso is not None else infer_ddim_iso(args.ckpt)
    ckpt_abs = str(Path(args.ckpt).resolve())

    written = set()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        def maybe_write_row(timestamp: int) -> None:
            if timestamp in written:
                return
            count = int(stats[timestamp]["count"])
            if count < total:
                return
            if count != total:
                raise RuntimeError(
                    f"Timestamp {timestamp} has count={count}, expected {total}. "
                    "This indicates an incomplete accumulation."
                )
            mean_val = float(stats[timestamp]["sum"]) / float(stats[timestamp]["count"])
            row = {
                "timestamp": int(timestamp),
                "num_images": int(total),
                "r_precision_mean": f"{mean_val:.8f}",
                "ckpt_path": ckpt_abs,
                "ddim_iso": ddim_iso_val,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
            }
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            written.add(timestamp)
            print(f"[OK] timestamp={timestamp} r_precision_mean={mean_val:.8f}")

        with torch.no_grad():
            for _ in tqdm(range(n_rounds), desc="DDIM transient generation"):
                n = min(args.batch_size, remaining)
                x = torch.randn(n, channels, image_size, image_size, device=device)

                # x at conceptual timestamp T (e.g., 1000 for CIFAR config).
                if num_timesteps in target_set:
                    update_r_precision_stats(
                        timestamp=num_timesteps,
                        x=x,
                        config=config,
                        scorer=scorer,
                        stats=stats,
                    )
                    maybe_write_row(num_timesteps)

                # Reverse DDIM trajectory from t=T-1 down to t=0.
                for i in range(num_timesteps - 1, -1, -1):
                    t_batch = torch.full((n,), i, device=device, dtype=torch.float32)
                    eps = model(x, t_batch)

                    sa_t = sqrt_alpha[i].view(1, 1, 1, 1)
                    so_t = sqrt_one_minus_alpha[i].view(1, 1, 1, 1)
                    sa_next = sqrt_alpha_next[i].view(1, 1, 1, 1)
                    so_next = sqrt_one_minus_alpha_next[i].view(1, 1, 1, 1)

                    x0_t = (x - so_t * eps) / sa_t
                    x = sa_next * x0_t + so_next * eps  # DDIM eta=0

                    if i in target_set:
                        update_r_precision_stats(
                            timestamp=i,
                            x=x,
                            config=config,
                            scorer=scorer,
                            stats=stats,
                        )
                        maybe_write_row(i)

                remaining -= n

        for t in pending_timestamps:
            maybe_write_row(t)

    missing_rows = [t for t in pending_timestamps if t not in written]
    if missing_rows:
        raise RuntimeError(
            f"Missing CSV rows for timestamps: {missing_rows}. "
            "Check accumulation/count logic."
        )

    print(f"[DONE] Wrote CSV rows to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
