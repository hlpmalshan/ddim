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
from datetime import timedelta

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
        default="ddpm_noisy",
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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DistributedDataParallel multi-GPU training (launch with torchrun)",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="Distributed backend to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank provided by torchrun",
    )
    parser.add_argument("--sequence", action="store_true")

    parser.add_argument(
        "--weighting_type",
        type=str,
        default='constant',
        help="Loss weighting type ('constant' or 'snr')"
    )
    
    parser.add_argument(
        "--intermediate_timesteps",
        type=str,
        default="",
        help="Comma-separated list of timesteps to save intermediates at (e.g., '900,800,700,600,500,400,300,200,100,50,0')"
    )
    
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # Parse intermediate_timesteps into a sorted list of integers
    if args.intermediate_timesteps:
        args.intermediate_timesteps = [int(s.strip()) for s in args.intermediate_timesteps.split(',')]
        args.intermediate_timesteps = sorted(set(args.intermediate_timesteps), reverse=True)  # Sort descending
    else:
        args.intermediate_timesteps = []
    
    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    os.makedirs(tb_path, exist_ok=True)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    using_env_dist = env_world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ
    args.distributed = bool(args.distributed or using_env_dist)

    if args.distributed and not using_env_dist and args.local_rank < 0:
        logging.warning(
            "--distributed specified but torchrun environment not detected; running in single-process mode."
        )
        args.distributed = False

    if args.distributed and not torch.cuda.is_available():
        logging.warning("Distributed requested but CUDA not available; falling back to single-process mode.")
        args.distributed = False

    if args.distributed:
        if args.local_rank >= 0:
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, timeout=timedelta(minutes=240),init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_main_process = (rank == 0)
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path) and is_main_process:
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
                    os.makedirs(tb_path, exist_ok=True)
                    # if os.path.exists(tb_path):
                    #     shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            elif is_main_process:
                os.makedirs(args.log_path)
                os.makedirs(tb_path, exist_ok=True)

            if is_main_process:
                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path) if is_main_process else None
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if is_main_process:
            file_handler = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(level)

        if args.sample:
            image_samples_root = os.path.join(args.exp, "image_samples")
            requested_folder = os.path.join(image_samples_root, args.image_folder)

            if is_main_process:
                os.makedirs(image_samples_root, exist_ok=True)
                if not os.path.exists(requested_folder):
                    os.makedirs(requested_folder)
                else:
                    if not (args.fid or args.interpolation):
                        overwrite = False
                        if args.ni:
                            overwrite = True
                        else:
                            response = input(
                                f"Image folder {requested_folder} already exists. Overwrite? (Y/N)"
                            )
                            if response.upper() == "Y":
                                overwrite = True

                        if overwrite:
                            shutil.rmtree(requested_folder)
                            os.makedirs(requested_folder)
                        else:
                            print("Output image folder exists. Program halted.")
                            sys.exit(0)

            args.image_folder = requested_folder

    if args.distributed and dist.is_initialized():
        dist.barrier()

    # add device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if args.distributed else f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    new_config.device = device
    new_config.distributed = args.distributed
    new_config.rank = rank
    new_config.world_size = world_size
    new_config.local_rank = local_rank

    if not args.distributed or is_main_process:
        logging.info("Using device: {}".format(device))
        if args.distributed:
            logging.info(
                "Distributed initialized: world_size=%d, rank=%d, local_rank=%d",
                world_size,
                rank,
                local_rank,
            )

    # set random seed
    seed = args.seed + (rank if args.distributed else 0)
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
        if getattr(config, "distributed", False) and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
