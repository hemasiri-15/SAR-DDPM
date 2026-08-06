"""
Train SAR-DDPM model.
"""

import argparse
import os
import datetime
from torch.utils.data import DataLoader
import blobfile as bf
import torch

# ==========================================================
# NVIDIA Tensor Core Optimization
# ==========================================================
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    set_seed,
)
from guided_diffusion.train_util import TrainLoop
from structdiff.data.wavelet_dataset import WaveletDataset
from parameters import default_args

def main():
    args = create_argparser().parse_args()
    print("Args: " + str(args) + "\n")

    if (args.seed is not None):
        set_seed(args.seed)

    # cuDNN's autotuner benchmarks convolution algorithms on first use and
    # picks the fastest for the given (fixed) input size. Now driven by
    # --benchmark instead of being hardcoded at import time (previously the
    # flag was defined but had no effect at all). Default is True, matching
    # this repo's prior always-on behavior.
    torch.backends.cudnn.benchmark = args.benchmark

    dist_util.setup_dist()

    # Derive a human-readable dataset name for the log-folder timestamp
    # prefix from args.train_dir, robust to a leading "./", a trailing
    # slash, or no separators at all. os.path.normpath collapses all of
    # these to a single canonical form before taking the final path
    # component, so e.g. "./Training_Data", "Training_Data", and
    # "/data/SEN12/train/" all resolve sensibly (to "Training_Data",
    # "Training_Data", and "train" respectively). Falls back to a fixed
    # literal only for a fully degenerate path (e.g. ".").
    dataset_name = os.path.basename(os.path.normpath(args.train_dir)) or "dataset"

    log_folder = bf.join(
        args.log_path,
        datetime.datetime.now().strftime(
            f"{dataset_name}_%Y-%m-%d-%H-%M-%S"
        ),
    )
    logger.configure(dir=log_folder, log_suffix="_train", format_strs=["log", "csv"])

    logger.log("Training dataset: " + args.train_dir)
    logger.log("Validation dataset: " + args.val_dir)
    if (args.resume_checkpoint):
        logger.log("Pretrained checkpoint: " + args.resume_checkpoint)
    logger.log("Args: " + str(args) + "\n")

    logger.log("Creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if args.compile_model:
        # torch.compile is not yet wired in here: it hasn't been verified
        # against this repo's custom autograd hook (EpsInterceptHook), the
        # structure/edge/wavelet/SSIM auxiliary losses, and DDP together.
        # Left explicit rather than silently compiling (or silently doing
        # nothing behind a misleading log message) until that's verified.
        logger.log("compile_model=True requested, but torch.compile is not yet enabled for this model; running eagerly.")

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loaders...")

    train_dataset = WaveletDataset(args.train_dir, train=True, num_channels=args.in_channels, crop_size=(args.large_size, args.large_size), seed=args.seed)

    loader_kwargs = {}

    if args.num_workers > 0:
        loader_kwargs.update(
            dict(
                persistent_workers=True,
                prefetch_factor=2,
            )
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, **loader_kwargs)
    val_dataset = WaveletDataset(args.val_dir, train=False, num_channels=args.in_channels, crop_size=(args.large_size, args.large_size), length=((args.val_samples//args.batch_size)*args.batch_size), seed=args.seed)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, **loader_kwargs)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        in_channels=args.in_channels,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_ddim=args.use_ddim,
        learn_sigma=args.learn_sigma,
        profile=args.profile,
    ).run_loop()


def create_argparser():
    custom_defaults = dict(
        # train and eval
        # NOTE: log_interval, save_interval, batch_size, compile_model, and
        # timestep_respacing are intentionally NOT set here even though this
        # is a natural place to look for them: parameters.py::default_args()
        # defines the authoritative value for each of these (applied after
        # this dict via defaults.update(default_args()) below), so a value
        # set here would be silently overridden. Keeping both previously
        # caused custom_defaults to display values that were never actually
        # in effect (e.g. batch_size=2 here, but 8 at runtime) -- see audit.
        val_samples = 20,
        use_ddim = False,
        microbatch = 1,
        lr_anneal_steps = 0,
        weight_decay = 0.0,
        seed = None,  # authoritative value also comes from default_args(); kept
                      # here only as the argparse fallback if that key is absent.
        num_workers=4,

        # model
        large_size = 256,
        learn_sigma = True,
        in_channels = 3,
        ema_rate = "",
        lr = 1e-4,
        use_fp16 = False,
        fp16_scale_growth = 1e-3,
        num_channels = 192,
        num_heads = 4,
        num_res_blocks = 2,
        resblock_updown = True,
        use_scale_shift_norm = True,
        attention_resolutions = "32,16,8",
        class_cond = False,
        compile_model=True,
        profile=False,
        # Default True to preserve this repo's prior behavior, where cuDNN
        # benchmarking was unconditionally enabled regardless of this flag's
        # value. The flag is now actually wired to torch.backends.cudnn.benchmark
        # in main(), so it can be explicitly disabled if needed.
        benchmark=True,

        # diffusion
        diffusion_steps = 1000,
        schedule_sampler = "uniform",
        noise_schedule = "linear",
    )
    defaults = sr_model_and_diffusion_defaults()
    defaults.update(custom_defaults)
    defaults.update(default_args())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    import traceback
    import sys

    try:
        args = create_argparser().parse_args()

        main()

    except Exception:
        print("\n========== PYTHON EXCEPTION ==========", flush=True)
        traceback.print_exc()
        sys.exit(1)
