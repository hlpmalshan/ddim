import os
import logging
import time
import glob
import math

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
'''
def update_csv(csv_path, mean_array, std_array, filled_len, stride=100):

    steps = [1] + [k * stride for k in range(1, filled_len)]
    df = pd.DataFrame({
        "step": steps[:filled_len],
        "mean": mean_array[:filled_len],
        "standard_deviation": std_array[:filled_len],
    })
    if os.path.exists(csv_path):
        last_step = pd.read_csv(csv_path, usecols=["step"])["step"].max()
    else:
        last_step = -1
    new = df[df["step"] > last_step]
    if not new.empty:
        mode = "a" if os.path.exists(csv_path) else "w"
        new.to_csv(csv_path, mode=mode, header=not os.path.exists(csv_path), index=False)
    return len(new)
    '''

def update_csv(csv_path, step, mean_val, std_val):
    """Append a single row (step, mean, std) to the CSV, safely avoiding duplicates."""
    if os.path.exists(csv_path):
        last_step = pd.read_csv(csv_path, usecols=["step"])["step"].max()
    else:
        last_step = -1

    if step <= last_step:  # nothing new to add
        return 0

    df = pd.DataFrame([{
        "step": step,
        "mean": mean_val,
        "standard_deviation": std_val
    }])

    mode = "a" if os.path.exists(csv_path) else "w"
    df.to_csv(csv_path, mode=mode, header=not os.path.exists(csv_path), index=False)
    return 1

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device(f"cuda:{self.config.device}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = getattr(self.config, "tb_logger", None)
        is_distributed = bool(getattr(self.config, "distributed", False))
        rank = getattr(self.config, "rank", 0)
        local_rank = getattr(self.config, "local_rank", 0)
        world_size = getattr(self.config, "world_size", 1)
        is_main_process = (not is_distributed) or (rank == 0)

        dataset, test_dataset = get_dataset(args, config)
        train_sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if is_distributed
            else None
        )
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(not is_distributed),
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        model = Model(config)

        model = model.to(self.device)
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        batch_norm_mean_list = np.zeros(config.training.n_iters, dtype=float)
        batch_norm_standard_deviation_list = np.zeros(config.training.n_iters, dtype=float)
        if self.args.resume_training:
            states = torch.load(
                os.path.join(self.args.log_path, "ckpt.pth"),
                map_location=self.device,
            )
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, batch_norm_mean, batch_norm_standard_deviation = loss_registry[config.model.type](model, x, t, e, b, 
                                                                                                        reg=args.reg, 
                                                                                                        weighting_type=args.weighting_type, 
                                                                                                        snr_gamma=config.snr_gamma)

                reduced_loss = loss.detach()
                if is_distributed:
                    reduced_loss = reduced_loss.clone()
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                    reduced_loss /= world_size

                if tb_logger is not None and is_main_process:
                    tb_logger.add_scalar("loss", reduced_loss.item(), global_step=step)

                if is_main_process:
                    logging.info(
                        f"step: {step}, epoch : {epoch},  loss: {reduced_loss.item()}, data time: {data_time / (i+1)}"
                    )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if (step % self.config.training.snapshot_freq == 0 or step == 1) and is_main_process:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    states = [
                        model_to_save.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                
                if is_main_process and config.model.save_statistics and (step == 1 or step % 100 == 0):
                        '''
                        if step == 1:
                            indedx = 0
                        else:
                            index = step // 100
                        
                        batch_norm_mean_list[index] = batch_norm_mean
                        batch_norm_standard_deviation_list[index] = batch_norm_standard_deviation
                        
                        filled_len = index + 1
                        appended = update_csv(os.path.join(self.args.log_path, "statistics.csv"), 
                                              batch_norm_mean_list, 
                                              batch_norm_standard_deviation_list,
                                              filled_len, 
                                              stride=100)
                        print(f"Appended {appended} new rows (up to step {step})")
                        '''
                        appended = update_csv(
                            os.path.join(self.args.log_path, "statistics.csv"),
                            step,
                            batch_norm_mean.item(),
                            batch_norm_standard_deviation.item()
                        )
                        if appended > 0:
                            print(f"Saved statistics at step {step}")
                      
                data_start = time.time()

    def sample(self):
        is_distributed = bool(getattr(self.config, "distributed", False))
        model = Model(self.config).to(self.device)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def sample_fid(self, model):
        config = self.config
        rank = getattr(config, "rank", 0)
        world_size = getattr(config, "world_size", 1)
        base_folder = self.args.image_folder

        # Ensure the output directory exists before other ranks proceed.
        if rank == 0:
            os.makedirs(base_folder, exist_ok=True)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

        total_n_samples = config.sampling.n_samples
        batch_size = config.sampling.batch_size

        if world_size > 1 and dist.is_available() and dist.is_initialized():
            base_existing = len(glob.glob(os.path.join(base_folder, "*.png")))
            existing_tensor = torch.tensor([base_existing], device=self.device, dtype=torch.long)
            dist.broadcast(existing_tensor, src=0)
            global_existing = int(existing_tensor.item())

            total_remaining = max(total_n_samples - global_existing, 0)
            if total_remaining == 0:
                if rank == 0:
                    logging.info(
                        "Requested %d samples already present in %s; skipping generation.",
                        total_n_samples,
                        base_folder,
                    )
                return

            per_rank = total_remaining // world_size
            remainder = total_remaining % world_size
            my_target = per_rank + (1 if rank < remainder else 0)
            start_offset = per_rank * rank + min(rank, remainder)
            img_id = global_existing + start_offset
        else:
            global_existing = len(glob.glob(os.path.join(base_folder, "*.png")))
            my_target = max(total_n_samples - global_existing, 0)
            img_id = global_existing

        if my_target == 0:
            logging.info("No samples left to generate for rank %d", rank)
            return

        print(f"starting from image {img_id} (rank {rank})")

        remaining = my_target
        n_rounds = math.ceil(remaining / batch_size)

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = min(batch_size, remaining)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1
                remaining -= n

        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
