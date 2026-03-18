"""
Distributed bpd (bits/dim) evaluation for your diffusion model on an image folder.

- DDP (torchrun)
- Input: a single image folder path (images can be directly inside or in subfolders)
- Preprocessing: uses config.data.image_size, config.data.channels + your data_transform(config, x)
- "Standard" bpd loop: vb terms (t=0 decoder NLL + KL for t>0) + prior KL
- Saves: bpd_mean.txt and {vb,mse,xstart_mse}_terms.npz on rank 0
- Progress: tqdm shown on rank 0
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import yaml
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import get_beta_schedule
from datasets import data_transform
from datetime import timedelta


# -------------------------
# Config helpers
# -------------------------
def dict2namespace(config):
    ns = type("Namespace", (), {})()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str):
    cfg = yaml.safe_load(Path(path).read_text())
    return dict2namespace(cfg)


def build_betas(config, device):
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    return torch.from_numpy(betas).float().to(device)


def load_model(ckpt_path: str, config, device, use_ema: bool = True):
    states = torch.load(ckpt_path, map_location=device)
    model = Model(config).to(device)
    model.load_state_dict(states[0], strict=True)

    if use_ema and getattr(config.model, "ema", False) and len(states) > 4:
        ema_helper = EMAHelper(mu=getattr(config.model, "ema_rate", 0.999))
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)

    model.eval()
    return model


# -------------------------
# DDP setup
# -------------------------
def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Returns (rank, world_size, local_rank, device).
    Works with torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=24))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_allreduce_(x: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# -------------------------
# Image folder dataset (no class subfolders required)
# -------------------------
def list_images_recursive(root: str) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No images found under: {root}")
    return files


class SimpleImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int, channels: int):
        self.paths = list_images_recursive(root)
        self.channels = int(channels)

        # Deterministic preprocessing controlled by config.data.image_size / channels.
        # Resize(int) in torchvision resizes shorter side to int; CenterCrop makes it square.
        self.tfm = transforms.Compose(
            [
                transforms.Resize(int(image_size)),
                transforms.CenterCrop(int(image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p)
        if self.channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        x = self.tfm(img)  # [0,1], shape CxHxW
        return x


# -------------------------
# Diffusion math (standard bpd loop style)
# -------------------------
def _extract(a_1d: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a_1d.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def normal_kl(mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        -1.0
        + (logvar2 - logvar1)
        + torch.exp(logvar1 - logvar2)
        + (mu1 - mu2).pow(2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def discretized_gaussian_log_likelihood(x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """
    x, means: in [-1, 1]
    log_scales: log std
    returns per-pixel log prob (same shape as x)
    """
    centered_x = x - means
    inv_std = torch.exp(-log_scales)
    plus_in = inv_std * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_std * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    mid_log = torch.log(cdf_delta.clamp(min=1e-12))

    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, mid_log),
    )
    return log_probs


@torch.no_grad()
def calc_bpd_loop_eps_model(
    model,
    x0: torch.Tensor,
    betas: torch.Tensor,
    clip_denoised: bool = True,
) -> dict:
    """
    Returns (per-image):
      - total_bpd: (B,)
      - prior_bpd: (B,)
      - vb: (B, T)  where vb[:,0]=decoder NLL bpd, vb[:,t>0]=KL bpd
      - mse: (B, T)
      - xstart_mse: (B, T)

    Assumes model predicts epsilon: eps_theta = model(x_t, t_float)
    Uses fixed variance based on posterior variance (clipped).
    """
    device = x0.device
    B = x0.shape[0]
    T = betas.shape[0]

    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]], dim=0)

    posterior_var = betas * (1.0 - abar_prev) / (1.0 - abar)
    if T > 1:
        posterior_logvar_clipped = torch.log(
            torch.cat([posterior_var[1:2], posterior_var[1:]], dim=0).clamp(min=1e-20)
        )
    else:
        posterior_logvar_clipped = torch.log(betas.clamp(min=1e-20))

    coef1 = betas * torch.sqrt(abar_prev) / (1.0 - abar)
    coef2 = torch.sqrt(alphas) * (1.0 - abar_prev) / (1.0 - abar)

    vb = torch.zeros((B, T), device=device)
    mse = torch.zeros((B, T), device=device)
    xstart_mse = torch.zeros((B, T), device=device)

    num_dims = x0[0].numel()
    log2 = np.log(2.0)

    for t_idx in range(T):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)

        abar_t = _extract(abar, t, x0.shape)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * noise

        eps_theta = model(x_t, t.float())
        x0_pred = (x_t - torch.sqrt(1.0 - abar_t) * eps_theta) / torch.sqrt(abar_t.clamp(min=1e-20))
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        c1 = _extract(coef1, t, x0.shape)
        c2 = _extract(coef2, t, x0.shape)
        model_mean = c1 * x0_pred + c2 * x_t

        model_logvar = _extract(posterior_logvar_clipped, t, x0.shape)

        true_mean = c1 * x0 + c2 * x_t
        true_logvar = model_logvar

        if t_idx == 0:
            dec_ll = discretized_gaussian_log_likelihood(
                x=x0,
                means=model_mean,
                log_scales=0.5 * model_logvar,
            )
            term_nats = (-dec_ll).sum(dim=[1, 2, 3])
        else:
            kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar).sum(dim=[1, 2, 3])
            term_nats = kl

        vb[:, t_idx] = term_nats / (log2 * num_dims)
        mse[:, t_idx] = (noise - eps_theta).pow(2).mean(dim=[1, 2, 3])
        xstart_mse[:, t_idx] = (x0 - x0_pred).pow(2).mean(dim=[1, 2, 3])

    abar_T = abar[-1]
    mu_qT = torch.sqrt(abar_T) * x0
    var_qT = (1.0 - abar_T)
    prior_kl = 0.5 * (mu_qT.pow(2) + var_qT - 1.0 - torch.log(var_qT.clamp(min=1e-20))).sum(dim=[1, 2, 3])
    prior_bpd = prior_kl / (log2 * num_dims)

    total_bpd = vb.sum(dim=1) + prior_bpd

    return {
        "total_bpd": total_bpd,
        "prior_bpd": prior_bpd,
        "vb": vb,
        "mse": mse,
        "xstart_mse": xstart_mse,
    }


# -------------------------
# Evaluation driver
# -------------------------
@torch.no_grad()
def evaluate_ddp(
    model,
    betas: torch.Tensor,
    config,
    image_dir: str,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    num_samples: Optional[int],
    clip_denoised: bool,
    rank: int,
):
    dataset = SimpleImageFolderDataset(
        root=image_dir,
        image_size=int(config.data.image_size),
        channels=int(config.data.channels),
    )

    if num_samples is not None:
        num_samples = min(int(num_samples), len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if (dist.is_available() and dist.is_initialized()) else None
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = next(model.parameters()).device
    T = betas.shape[0]

    count = torch.zeros((), device=device, dtype=torch.long)
    total_sum = torch.zeros((), device=device, dtype=torch.float64)

    vb_sum = torch.zeros((T,), device=device, dtype=torch.float64)
    mse_sum = torch.zeros((T,), device=device, dtype=torch.float64)
    xstart_mse_sum = torch.zeros((T,), device=device, dtype=torch.float64)

    pbar = dl
    if rank == 0:
        pbar = tqdm(dl, total=len(dl), ncols=100, desc="bpd_eval", dynamic_ncols=True)

    for x0 in pbar:
        x0 = x0.to(device, non_blocking=True)
        x0 = data_transform(config, x0)  # normalization / preprocessing from your codebase

        metrics = calc_bpd_loop_eps_model(model, x0, betas, clip_denoised=clip_denoised)

        B = x0.shape[0]
        count += B
        total_sum += metrics["total_bpd"].double().sum()

        vb_sum += metrics["vb"].double().sum(dim=0)
        mse_sum += metrics["mse"].double().sum(dim=0)
        xstart_mse_sum += metrics["xstart_mse"].double().sum(dim=0)

        if rank == 0:
            running_bpd = (total_sum / count.clamp(min=1)).item()
            pbar.set_postfix({"seen": int(count.item()), "bpd": f"{running_bpd:.4f}"})

    ddp_allreduce_(count)
    ddp_allreduce_(total_sum)
    ddp_allreduce_(vb_sum)
    ddp_allreduce_(mse_sum)
    ddp_allreduce_(xstart_mse_sum)

    n = max(int(count.item()), 1)
    bpd_mean = (total_sum / n).item()
    vb_mean = (vb_sum / n).detach().cpu().numpy()
    mse_mean = (mse_sum / n).detach().cpu().numpy()
    xstart_mse_mean = (xstart_mse_sum / n).detach().cpu().numpy()
    
    print(f"bpd_mean : , {bpd_mean}, count : {n}" )
    # print(f"vb_mean : ", vb_mean)
    # print(f"mse_mean : ", mse_mean)
    # print(f"xstart_mean : ", xstart_mse_mean)

    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_dir, "bpd_mean.txt"), "w", encoding="utf-8") as f:
            f.write(f"num_samples={n}\n")
            f.write(f"bpd_mean={bpd_mean:.6f}\n")

        np.savez(os.path.join(out_dir, "vb_terms.npz"), vb=vb_mean)
        np.savez(os.path.join(out_dir, "mse_terms.npz"), mse=mse_mean)
        np.savez(os.path.join(out_dir, "xstart_mse_terms.npz"), xstart_mse=xstart_mse_mean)

    ddp_barrier()
    return bpd_mean


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    parser.add_argument("--out_dir", type=str, default="./bpd_eval_out")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_ddp()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    config = load_config(args.config)
    betas = build_betas(config, device)

    model = load_model(args.ckpt, config, device, use_ema=True)

    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    evaluate_ddp(
        model=model,
        betas=betas,
        config=config,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        clip_denoised=args.clip_denoised,
        rank=rank,
    )


if __name__ == "__main__":
    main()
