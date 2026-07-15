import copy
import functools
import time
from tqdm import tqdm
import blobfile as bf

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW, Adam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .test_util import evaluate
from structdiff.losses.structure_consistency_loss import (
    MultiScaleStructureConsistencyLoss,
)

from structdiff.losses.edge_aware_loss import (
    EdgeAwareLoss,
    DEFAULT_LAMBDA_EDGE,
)

from structdiff.losses.wavelet_consistency_loss import (
    WaveletConsistencyLoss,
    DEFAULT_LAMBDA_WAVELET,
)

from structdiff.losses.ssim_loss import (
    SSIMLoss,
    DEFAULT_LAMBDA_SSIM,
)

from structdiff.losses.eps_intercept_hook import (
    EpsInterceptHook,
    reconstruct_x0,
)

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_loader,
        val_loader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        in_channels,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_ddim=False,
        learn_sigma=True,
    ): #__init__
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",") if x]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = find_resume_checkpoint() or resume_checkpoint
        self.in_channels = in_channels
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_ddim = use_ddim
        self.learn_sigma = learn_sigma

        self.step = 0

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_batch = self.batch_size * world_size

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if torch.cuda.is_available():
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Make a separate copy of the master parameters for each EMA rate.
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

       # Use DDP only if distributed training has actually been initialized.
        if (
            torch.cuda.is_available()
            and dist.is_available()
            and dist.is_initialized()
        ):
            print("Using DistributedDataParallel")
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[0],
                output_device=0,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            print("Using single-GPU training")
            self.use_ddp = False
            self.ddp_model = self.model 

        self.lambda_struct = 0.1

        self.struct_loss_fn = MultiScaleStructureConsistencyLoss(
            kernels=(3, 5, 9)
        )

        # A5 — Edge-Aware Loss
        # lambda_edge is set below lambda_struct=0.1 because Sobel gradient
        # magnitudes are naturally larger than structure tensor components,
        # so a smaller weight prevents edge loss from dominating.
        self.lambda_edge = DEFAULT_LAMBDA_EDGE   # 0.05; set 0.0 to disable
        self.edge_loss_fn = EdgeAwareLoss(
            alpha=0.6,   # weight on L1 gradient-magnitude term
            beta=0.4,    # weight on directional (Gx / Gy) L1 term
        ).to(dist_util.dev())

        # A34 — Wavelet Consistency Loss
        self.lambda_wavelet = DEFAULT_LAMBDA_WAVELET

        self.wavelet_loss_fn = WaveletConsistencyLoss().to(
            dist_util.dev()
        )

        self.eps_hook = EpsInterceptHook(
            self.ddp_model,
            learn_sigma=self.learn_sigma,
        )

        # A36 — SSIM Loss
        self.lambda_ssim = DEFAULT_LAMBDA_SSIM

        self.ssim_loss_fn = SSIMLoss().to(
            dist_util.dev()
        )

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.log(f"Loading model from checkpoint: {self.resume_checkpoint}...")
                state_dict = dist_util.load_state_dict(self.resume_checkpoint, map_location=dist_util.dev())
                state_dict = self.modify_state_dict(state_dict) # modify size of state dict if necessary
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

                print("\n================ CHECKPOINT REPORT ================")
                print("Missing keys:", len(missing))
                for k in missing:
                    print("  MISSING:", k)

                print("Unexpected keys:", len(unexpected))
                for k in unexpected:
                    print("  UNEXPECTED:", k)
                print("===================================================\n")


        dist_util.sync_params(self.model.parameters())


    # Load parameters for a specific EMA rate from a checkpoint at the resume step
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        ema_checkpoint = find_ema_checkpoint(self.resume_checkpoint, rate)
        if ema_checkpoint:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.log(f"Loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                state_dict = self.modify_state_dict(state_dict) # modify size of state dict if necessary
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params


    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), "opt_latest.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    # Define a function to modify the state dictionary
    def modify_state_dict(self, state_dict, new_c_start=2):
        new_state_dict = {}
        # cond_channels = total_channels - gaussian_channels
        # gaussian_channel = new_c_start
        # cond_channel = new_c_start + 3
        # sigma_channel = new_c_start + 

        for key, value in state_dict.items():
            if ('input_blocks.0.0.weight' == key) and (value.shape != self.model.input_blocks[0][0].weight.shape):
            # if ('input_blocks.0.0.weight' == key):
                # Extract the weights for the specified channels
                gaussian_weights = value[:, :3, :, :]
                cond_weights = value[:, 3:6, :, :]
                if (self.in_channels == 6):
                    # new_weights = torch.cat((gaussian_weights/100.0, new_weights), dim=1)
                    gaussian_weights = torch.cat((gaussian_weights/2.0, gaussian_weights/2.0), dim=1)
                    cond_weights = torch.cat((cond_weights/2.0, cond_weights/2.0), dim=1)
                elif (self.in_channels == 1):
                    gaussian_weights = gaussian_weights.mean(dim=1, keepdim=True)
                    cond_weights = cond_weights.mean(dim=1, keepdim=True)
                new_weights = torch.cat((gaussian_weights, cond_weights), dim=1)

                # Stack the duplicated Gaussian weights with the noisy weights along the second axis
                new_state_dict[key] = new_weights
            elif ('out.2.weight' == key) and (value.shape != self.model.out[2].weight.shape):
            # elif ('out.2.weight' == key):
                new_weights = value[:3, :, :, :]
                if (self.in_channels == 1):
                    new_weights = new_weights.mean(dim=0, keepdim=True)
                elif (self.in_channels == 6):
                    new_weights = torch.cat((new_weights, new_weights), dim=0)
                if self.learn_sigma:
                    sigma_weights = value[3:, :, :, :]
                    if (self.in_channels == 1):
                        sigma_weights = sigma_weights.mean(dim=0, keepdim=True)
                    elif (self.in_channels == 6):
                        sigma_weights = torch.cat((sigma_weights, sigma_weights), dim=0)
                    new_weights = torch.cat((new_weights, sigma_weights), dim=0)
                new_state_dict[key] = new_weights
            elif ('out.2.bias' == key) and (value.shape != self.model.out[2].bias.shape):
            # elif ('out.2.bias' == key):
                new_weights = value[:3]
                if (self.in_channels == 1):
                    new_weights = new_weights.mean(dim=0, keepdim=True)
                elif (self.in_channels == 6):
                    new_weights = torch.cat((new_weights, new_weights), dim=0)
                if self.learn_sigma:
                    sigma_weights = value[3:]
                    if (self.in_channels == 1):
                        sigma_weights = sigma_weights.mean(dim=0, keepdim=True)
                    elif (self.in_channels == 6):
                        sigma_weights = torch.cat((sigma_weights, sigma_weights), dim=0)
                    new_weights = torch.cat((new_weights, sigma_weights), dim=0)
                new_state_dict[key] = new_weights
            else:
                new_state_dict[key] = value
        return new_state_dict


    def run_loop(self):
        best_psnr = 0.0
        best_max_psnr = 0.0

        images_folder = bf.join(logger.get_dir(), "val_images")
        if not bf.exists(images_folder):
            bf.makedirs(images_folder)

        def load_data(loader):
            while True:
                yield from loader
        train_generator = load_data(self.train_loader)

        net_val_time = 0.0
        start_time = time.perf_counter()
        net_loss = 0.0

        # Get performance before training
        avg_psnr, avg_ssim, _, mse, max_psnr = evaluate(
            self.val_loader,
            self.diffusion,
            self.ddp_model,
            dist_util.dev(),
            images_folder,
            cycle_spinning=True,
            cycle_width=128,
            use_ddim=self.use_ddim,
        )
        logger.log(f"\tStep = {self.step:>5},  PSNR: {avg_psnr:5.2f},  SSIM: {avg_ssim:5.3f},  MSE: {mse:2.1e},  Loss: 0.00e+00,  Net training time: {(time.perf_counter() - start_time - net_val_time):.1f}s,  Net validation time: {net_val_time:.1f}s")
        progress_bar = tqdm(total=self.log_interval, desc="[Training] Step:     0, Loss: 0.00e+00", unit="step")

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            clean_tensor, noisy_tensor, image_filename, look_num, \
            struct_tensor_s1, struct_tensor_s2, struct_tensor_s3, \
            spectral_tensor, wavelet_tensor = next(train_generator)

            noisy_tensor = noisy_tensor.to(dist_util.dev())
            clean_tensor = clean_tensor.to(dist_util.dev())
            look_num = look_num.to(dist_util.dev())

            struct_tensor_s1 = struct_tensor_s1.to(dist_util.dev())
            struct_tensor_s2 = struct_tensor_s2.to(dist_util.dev())
            struct_tensor_s3 = struct_tensor_s3.to(dist_util.dev())
            spectral_tensor = spectral_tensor.to(dist_util.dev())
            wavelet_tensor = wavelet_tensor.to(dist_util.dev())

            model_kwargs = {
                "noisy": noisy_tensor,
                "look_num": look_num,
                "struct_tensors": (
                    struct_tensor_s1,
                    struct_tensor_s2,
                    struct_tensor_s3,
                ),
                "spectral_tensor": spectral_tensor,
                "wavelet_tensor": wavelet_tensor,
            }
            net_loss += self.run_step(clean_tensor, model_kwargs)

            self.step += 1

            progress_bar.set_description(desc=f"[Training] Step: {self.step:5d}, Loss: {(net_loss/(((self.step-1) % self.log_interval)+1)):2.2e}")
            progress_bar.update()

            if ((self.step % self.log_interval) == 0):
                progress_bar.close()

                val_time = time.perf_counter()
                avg_psnr, avg_ssim, _, mse, max_psnr = evaluate(
                    self.val_loader,
                    self.diffusion,
                    self.ddp_model,
                    dist_util.dev(),
                    images_folder,
                    cycle_spinning=True,
                    cycle_width=128,
                )
                net_val_time += time.perf_counter() - val_time

                logger.log(f"\tStep = {self.step:>5},  PSNR: {avg_psnr:5.2f},  SSIM: {avg_ssim:5.3f},  MSE: {mse:2.1e},  Loss: {(net_loss/(((self.step-1) % self.log_interval)+1)):2.2e},  Net training time: {(time.perf_counter() - start_time - net_val_time):.1f}s,  Net validation time: {net_val_time:.1f}s")

                if best_psnr < avg_psnr:
                    best_psnr = avg_psnr
                    logger.log(f"New best PSNR: {avg_psnr:5.2f}, SSIM: {avg_ssim:5.3f}")
                    print(f"New best PSNR: {avg_psnr:5.2f}, SSIM: {avg_ssim:5.3f}. Saving... ", end="", flush=True)
                    self.save()
                    print("Done")
                if best_max_psnr < max_psnr:
                    best_max_psnr = max_psnr
                    logger.log(f"New best maximum PSNR: {max_psnr:5.2f}")
                    print(f"New best maximum PSNR: {max_psnr:5.2f}. Saving... ", end="", flush=True)
                    self.save(max=True)
                    print("Done")
                if ((self.step % self.save_interval) == 0):
                    logger.log(f"Saving latest model")
                    self.save(latest=True)

                net_loss = 0.0
                progress_bar = tqdm(total=self.log_interval, desc=f"[Training] Step: {self.step:5d}, Loss: 0.00e+00", unit="step")
                
        # Save the last checkpoint if it wasn't already saved.
        if ((self.step % self.save_interval) != 0):
            self.save(latest=True)


    def run_step(self, batch, cond):
        loss = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return loss


    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        net_loss = 0.0

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k : v[i : i + self.microbatch].to(dist_util.dev()) if isinstance(v, torch.Tensor) else v
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.eps_hook,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            x0_hat = reconstruct_x0(
                self.eps_hook.last_x_t,
                self.eps_hook.last_t,
                self.eps_hook.last_eps_hat.float(),
                self.diffusion.original_alphas_cumprod,
            )

            struct_loss = self.struct_loss_fn(
                x_hat=x0_hat,
                x_clean=micro.float().detach(),
            )

            loss = loss + self.lambda_struct * struct_loss

            logger.logkv_mean(
                "struct_loss",
                struct_loss.item(),
            )

            # ── A5: Edge-Aware Loss ────────────────────────────────────
            # Reuses x0_hat already computed for A33 — zero extra UNet
            # forward passes.  micro.float().detach() prevents the loss
            # from computing gradients through the ground-truth path.
            if self.lambda_edge > 0.0:
                edge_loss = self.edge_loss_fn(
                    x_pred=x0_hat,               # in autograd graph: x0_hat→eps_hat→UNet
                    x_gt=micro.float().detach(), # GT detached: only x0_hat gets grads
                )
                loss = loss + self.lambda_edge * edge_loss
                logger.logkv_mean("edge_loss", edge_loss.item())
            # ── end A5 ────────────────────────────────────────────────

            # ── A34: Wavelet Consistency Loss ─────────────────────────
            if self.lambda_wavelet > 0.0:
                wavelet_loss = self.wavelet_loss_fn(
                    x_pred=x0_hat,
                    x_gt=micro.float().detach(),
            )

            loss = loss + self.lambda_wavelet * wavelet_loss

            logger.logkv_mean(
                "wavelet_loss",
                wavelet_loss.item(),
            )
            # ── A36: SSIM Loss ────────────────────────────────────────
            if self.lambda_ssim > 0.0:
                ssim_loss = self.ssim_loss_fn(
                    x_pred=x0_hat,
                    x_gt=micro.float().detach(),
                )

                loss = loss + self.lambda_ssim * ssim_loss

                logger.logkv_mean(
                    "ssim_loss",
                    ssim_loss.item(),
                )
            # ──────────────────────────────────────────────────────────

            net_loss += loss

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

        return net_loss / self.batch_size


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr


    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def save(self, latest=False, max=False):
        logger_dir = logger.get_dir()
        savetype = "latest" if latest else ("max" if max else "best")

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                if not rate:
                    filename = f"model_{savetype}.pt"
                else:
                    filename = f"ema_{savetype}_{rate}.pt"

                with bf.BlobFile(bf.join(logger_dir, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        #     opt_filename = f"opt_{savetype}.pt"
        #     with bf.BlobFile(bf.join(logger_dir, opt_filename), "wb") as f:
        #         torch.save(self.opt.state_dict(), f)

        if dist.is_initialized():
            dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_latest_{rate}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    logger.dumpkvs()
