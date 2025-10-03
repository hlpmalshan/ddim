import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributed as dist

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--reg", type=float, required=True, help="Iso Reg", default=0.0
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--gpu", type=int, default=1, help="id of the gpu"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    # Multi-GPU / DDP controls
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DistributedDataParallel multi-GPU mode (launch with torchrun)",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="Distributed backend to use (nccl/gloo)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (set by torchrun)",
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # Distributed setup and device selection (before any filesystem ops)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    using_env_dist = env_world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ
    # Decide whether to run in distributed mode (torchrun sets env vars)
    args.distributed = bool(getattr(args, "distributed", False) or using_env_dist)

    # Read ranks
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))) if args.distributed else 0
    world_size = env_world_size if args.distributed else 1
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))) if args.distributed else 0

    # Initialize process group if needed
    if args.distributed:
        if not torch.cuda.is_available():
            logging.warning("Distributed requested but CUDA not available; falling back to single-process.")
            args.distributed = False
        else:
            torch.cuda.set_device(local_rank)
            backend = getattr(args, "dist_backend", os.environ.get("DIST_BACKEND", "nccl"))
            dist.init_process_group(backend=backend, init_method="env://")

    is_main = (not args.distributed) or (dist.get_rank() == 0)
    device = torch.device(f"cuda:{local_rank}") if (torch.cuda.is_available() and args.distributed) else (
        torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    )
    if is_main:
        logging.info("Using device: {}".format(device))
        if args.distributed:
            logging.info(f"Distributed initialized: world_size={dist.get_world_size()}, rank={dist.get_rank()}, local_rank={local_rank}")
    new_config.device = device
    new_config.distributed = args.distributed
    new_config.rank = (dist.get_rank() if args.distributed else 0)
    new_config.world_size = (dist.get_world_size() if args.distributed else 1)
    new_config.local_rank = local_rank

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path) and is_main:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            elif is_main:
                os.makedirs(args.log_path)

            if is_main:
                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path) if is_main else None
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt")) if is_main else None
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        if handler2 is not None:
            handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        if handler2 is not None:
            logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample and is_main:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # set random seed (offset by rank for DDP)
    seed = args.seed + (new_config.rank if args.distributed else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config, device=config.device)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())
    finally:
        if getattr(config, "distributed", False) and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
