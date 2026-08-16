import os
import sys
import csv
import time
import math
import itertools
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import DataLoader

import lpips
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

from . import logger

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the other directory
other_dir = os.path.join(current_dir, "..", "scripts")
# Add the other directory to sys.path
sys.path.append(other_dir)
from datasets import SARDataset

# TODO: change these for your own data paths
sen12_sar_list_path = "../sen12/sar/sar_test_samples.txt"
hrsid_sar_list_path = "../HRSID_png/inshore_images/sar_test_samples.txt"

_VALID_SAMPLE_MODES = ("LAST", "MAX", "SWEEP")

# ----------------------------------------------------------------------
# LPIPS is constructed lazily, on the device of the tensors actually being
# compared (one instance cached per device), rather than at module-import
# time on a single hardcoded device. This avoids paying model-load cost on
# unrelated imports of this module and avoids a cross-device error under
# multi-GPU DDP, where each rank's tensors live on that rank's own device.
# ----------------------------------------------------------------------
_lpips_models = {}


def _get_lpips_model(device):
    key = str(device)
    if key not in _lpips_models:
        _lpips_models[key] = lpips.LPIPS(net='alex').to(device)
    return _lpips_models[key]


def _prepare_conditions(conditions, device):
    """
    Move every conditioning tensor produced by the dataset to `device`.

    `conditions` is the dict yielded by the dataset alongside
    (clean_tensor, noisy_tensor, image_filename); it must contain:
        "look_num"        : tensor, [B] or [B, 1]              (non-spatial)
        "struct_tensors"  : 3-tuple of tensors, one per structure-tensor scale,
                             each [B, C_s, H, W]
        "spectral_tensor" : tensor, [B, C_f, H, W]
        "wavelet_tensor"  : tensor, [B, C_w, H, W]
    """
    look_num = conditions["look_num"].to(device)
    struct_tensors = tuple(t.to(device) for t in conditions["struct_tensors"])
    spectral_tensor = conditions["spectral_tensor"].to(device)
    wavelet_tensor = conditions["wavelet_tensor"].to(device)
    return look_num, struct_tensors, spectral_tensor, wavelet_tensor


def _build_model_kwargs(noisy, look_num, struct_tensors, spectral_tensor, wavelet_tensor):
    """Assemble the model_kwargs dict shared by every sampling call."""
    struct_s1, struct_s2, struct_s3 = struct_tensors
    return {
        "noisy": noisy,
        "look_num": look_num,
        "struct_tensor": struct_s1,
        "struct_tensors": struct_tensors,
        "spectral_tensor": spectral_tensor,
        "wavelet_tensor": wavelet_tensor,
    }


def _sample_canonical(sample_fn, model, shape, model_kwargs, use_ddim, progress=False):
    """
    Run the diffusion sampler and normalize its output to a single canonical
    layout: [T, B, C, H, W], values in [-1, 1].

    T == 1            for DDIM (ddim_sample_loop returns only the final sample)
    T == num_timesteps for DDPM ancestral sampling (p_sample_loop returns the
                        full trajectory)

    All downstream evaluation code (metrics, cycle spinning, sample
    selection) operates on this one layout, so no DDPM/DDIM branching is
    needed anywhere else in this file.
    """
    sample = sample_fn(model, shape, clip_denoised=True, model_kwargs=model_kwargs, progress=progress)
    if use_ddim:
        sample = sample.unsqueeze(0)  # [B, C, H, W] -> [1, B, C, H, W]
    return sample


def _select_prediction(pred_image, pred_tensor, sample_to_use, best_idx_per_sample):
    """
    Reduce the canonical [T, B, H, W] / [T, B, C, H, W] trajectories to a
    single prediction per sample, according to `sample_to_use`.

    Returns (pred_image[B, H, W], pred_tensor[B, C, H, W]) — always the SAME
    underlying prediction for both, so every downstream metric (MSE, PSNR,
    SSIM, LPIPS) describes one consistent image.
    """
    batch_size = pred_image.shape[1]

    if sample_to_use == "LAST":
        return pred_image[-1], pred_tensor[-1]

    if sample_to_use == "MAX":
        return (
            pred_image[best_idx_per_sample, range(batch_size)],
            pred_tensor[best_idx_per_sample, range(batch_size)],
        )

    # sample_to_use == "SWEEP" (validated at evaluate() entry)
    n = min(8, pred_image.shape[0])
    return pred_image[-n:].mean(dim=0), pred_tensor[-n:].mean(dim=0)


def evaluate(loader, diffusion, model, device, images_dir, cycle_spinning=False, cycle_width=0,
             log=False, test=False, use_ddim=False, sample_to_use="LAST", save_images=True):
    """
    Run full validation/test evaluation over `loader`.

    Tensor shape conventions used throughout this function:
        clean_tensor, noisy_tensor              : [B, C, H, W], values in [-1, 1]
        struct_tensor_s{1,2,3}                   : [B, C_s, H, W]  (per-scale structure maps)
        spectral_tensor                          : [B, C_f, H, W]
        wavelet_tensor                           : [B, C_w, H, W]
        look_num                                 : [B] or [B, 1]  (not spatial; never shifted
                                                    during cycle spinning)
        pred_tensor (canonical, see _sample_canonical) : [T, B, C, H, W], values in [-1, 1]
        pred_image / clean_image / noisy_image   : channel-averaged grayscale;
                                                    [T, B, H, W] before selection,
                                                    [B, H, W] after sample_to_use selection

    Returns: (net_psnr, net_ssim, net_time, net_mse, max_psnr)
    """
    if sample_to_use not in _VALID_SAMPLE_MODES:
        raise ValueError(f"Unknown sample_to_use={sample_to_use!r}. Expected one of {_VALID_SAMPLE_MODES}.")

    # Always logged (not gated by `log`), since silently mismatched
    # use_ddim/cycle_spinning between calls is exactly the kind of bug
    # that's easy to reintroduce at a call site and hard to notice without
    # this -- see run_loop()'s baseline vs periodic validation calls.
    logger.log(
        f"[evaluate] use_ddim={use_ddim}, cycle_spinning={cycle_spinning}, "
        f"cycle_width={cycle_width}, sample_to_use={sample_to_use}\n"
    )

    sample_fn = diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop

    # Trajectory SSIM (per-iteration, used only for MAX-selection and the
    # diagnostic plot) can use a faster GPU-based SSIM during training-time
    # validation. The FINAL reported SSIM (img_ssim / net_ssim / CSV) always
    # stays on skimage, unconditionally -- see the per-sample loop below --
    # so nothing that gets logged, plotted-as-a-headline-number, or reported
    # in a paper ever depends on this switch.
    use_fast_ssim = not test
    ssim_torch = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device) if use_fast_ssim else None

    if log:
        sample_mode_messages = {
            "LAST": "!! Used the last sample !!",
            "MAX": "!! Used the max sample !!",
            "SWEEP": "!! Averaged the last 8 samples !!",
        }
        logger.log(sample_mode_messages[sample_to_use] + "\n")
        if use_ddim and sample_to_use != "LAST":
            # DDIM only ever produces a single sample (T == 1), so "MAX"/"SWEEP"
            # have no distinct effect from "LAST" — surfaced explicitly rather
            # than left as a silent no-op.
            logger.log(
                f"NOTE: sample_to_use={sample_to_use!r} has no effect under DDIM "
                f"(a single sample is produced); behavior is identical to 'LAST'.\n"
            )

    model.eval()

    net_psnr = 0.0  # sum PSNR metrics
    net_ssim = 0.0  # sum SSIM metrics
    net_mse = 0.0
    net_lpips = 0.0

    net_time = 0.0  # sum evaluation times

    csv_file = None
    csv_writer = None

    if images_dir is not None:
        csv_path = os.path.join(images_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image", "PSNR", "SSIM", "LPIPS", "Runtime_sec"])

    with torch.no_grad():
        data_iter = iter(loader)

        if test:
            # Warm up the model on the GPU using a REAL batch (with the full,
            # required conditioning kwargs — the model requires look_num /
            # struct_tensor(s) / spectral_tensor / wavelet_tensor on every
            # call, so a partial kwargs dict here would raise). The batch is
            # not discarded afterward: it's fed back as the first iteration
            # of the main loop below, so no extra dataset pass is wasted.
            try:
                warm_batch = next(data_iter)
            except StopIteration:
                warm_batch = None

            if warm_batch is not None:
                _, warm_noisy, _, warm_conditions = warm_batch
                warm_noisy = warm_noisy.to(device)
                warm_look_num, warm_struct, warm_spectral, warm_wavelet = _prepare_conditions(warm_conditions, device)
                warm_kwargs = _build_model_kwargs(warm_noisy, warm_look_num, warm_struct, warm_spectral, warm_wavelet)
                _ = _sample_canonical(sample_fn, model, warm_noisy.shape, warm_kwargs, use_ddim, progress=True)
                data_iter = itertools.chain([warm_batch], data_iter)

        # Metric buffers, allocated lazily once the true dataset length and
        # the actual number of diffusion iterations returned are known.
        total_samples = len(loader.dataset)
        all_tensor_psnr = None
        all_tensor_ssim = None
        all_tensor_lpips = None

        progress_bar = tqdm(
            data_iter, total=len(loader),
            desc=f"[{'Test' if test else 'Validation'}] PSNR: 00.00/00.00, SSIM: 0.000/0.000",
            unit='batch',
        )

        for batch_idx, data_tuple in enumerate(progress_bar):
            clean_tensor, noisy_tensor, image_filename, conditions = data_tuple

            clean_tensor = clean_tensor.to(device)
            noisy_tensor = noisy_tensor.to(device)
            look_num, struct_tensors, spectral_tensor, wavelet_tensor = _prepare_conditions(conditions, device)
            struct_tensor_s1, struct_tensor_s2, struct_tensor_s3 = struct_tensors

            model_kwargs = _build_model_kwargs(noisy_tensor, look_num, struct_tensors, spectral_tensor, wavelet_tensor)

            # Reformat the clean image for metrics: [-1,1] -> channel-averaged [0,1]
            clean_image = ((clean_tensor + 1.0) * 127.5).clamp(0, 255.0)
            clean_image = torch.round(torch.mean(clean_image, dim=1)) / 255.0
            clean_image = clean_image.contiguous()

            batch_size = clean_tensor.shape[0]

            batch_start = time.perf_counter()

            if cycle_spinning:
                [_, _, num_rows, num_cols] = noisy_tensor.size()
                N = int(np.ceil(num_rows / cycle_width) * np.ceil(num_cols / cycle_width))

                pred_tensor = None
                for row in range(0, num_rows, cycle_width):
                    for col in range(0, num_cols, cycle_width):
                        # All spatial conditioning tensors MUST receive the same
                        # shift as the noisy input to preserve spatial alignment.
                        # look_num is non-spatial and is intentionally left unshifted.
                        val_inputv = torch.roll(noisy_tensor, shifts=(row, col), dims=(-2, -1))
                        shifted_struct = tuple(
                            torch.roll(t, shifts=(row, col), dims=(-2, -1)) for t in struct_tensors
                        )
                        shifted_spectral = torch.roll(spectral_tensor, shifts=(row, col), dims=(-2, -1))
                        shifted_wavelet = torch.roll(wavelet_tensor, shifts=(row, col), dims=(-2, -1))

                        spin_kwargs = _build_model_kwargs(
                            val_inputv, look_num, shifted_struct, shifted_spectral, shifted_wavelet
                        )

                        sample = _sample_canonical(sample_fn, model, val_inputv.shape, spin_kwargs, use_ddim)

                        # Unspin (shift back) and accumulate the average.
                        # dims=(-2, -1) always targets the spatial axes
                        # regardless of whether `sample` is [B,C,H,W] (DDIM)
                        # or [T,B,C,H,W] (DDPM) — a zero shift is the identity
                        # roll, so no special-case handling is needed for the
                        # first iteration.
                        unshifted = torch.roll(sample, shifts=(-row, -col), dims=(-2, -1))
                        pred_tensor = (unshifted / N) if pred_tensor is None else pred_tensor + (unshifted / N)
            else:
                pred_tensor = _sample_canonical(sample_fn, model, noisy_tensor.shape, model_kwargs, use_ddim)

            elapsed_time = time.perf_counter() - batch_start
            net_time += elapsed_time

            iterations = pred_tensor.shape[0]

            if all_tensor_psnr is None:
                metric_shape = (total_samples, iterations)
                all_tensor_psnr = np.zeros(metric_shape, dtype=np.float32)
                all_tensor_ssim = np.zeros(metric_shape, dtype=np.float32)
                all_tensor_lpips = np.zeros(metric_shape, dtype=np.float32)

            pred_image = ((pred_tensor + 1.0) * 127.5).clamp(0, 255.0)
            pred_image = torch.round(torch.mean(pred_image, dim=2)) / 255.0
            pred_image = pred_image.contiguous()

            pred_image_np = pred_image.cpu().numpy()
            clean_image_np = clean_image.cpu().numpy()

            # Per-iteration PSNR/SSIM (used to locate the best iteration for
            # "MAX" selection, and for the trajectory plot below).
            best_psnr_idx_per_sample = [0] * batch_size
            best_psnr_per_sample = [0.0] * batch_size
            for b in range(batch_size):
                idx = batch_idx * loader.batch_size + b

                if use_fast_ssim:
                    # One GPU call covering all T iterations for this sample,
                    # instead of T separate calls each forcing a sync via
                    # .item() -- that per-call pattern is what actually makes
                    # a "fast" GPU metric end up slower than the CPU version
                    # for small per-call payloads.
                    pred_seq = torch.from_numpy(pred_image_np[:, b]).unsqueeze(1).to(device)  # [T, 1, H, W]
                    clean_rep = (
                        torch.from_numpy(clean_image_np[b])
                        .unsqueeze(0).unsqueeze(0).to(device)
                        .expand(iterations, 1, -1, -1)  # [T, 1, H, W], view-only, no copy
                    )
                    all_tensor_ssim[idx, :] = ssim_torch(pred_seq, clean_rep).detach().cpu().numpy()

                for t in range(iterations):
                    all_tensor_psnr[idx, t] = psnr(clean_image_np[b], pred_image_np[t, b], data_range=1)
                    if not use_fast_ssim:
                        all_tensor_ssim[idx, t] = ssim(clean_image_np[b], pred_image_np[t, b], data_range=1)

                best_psnr_idx_per_sample[b] = int(np.argmax(all_tensor_psnr[idx]))
                best_psnr_per_sample[b] = all_tensor_psnr[idx, best_psnr_idx_per_sample[b]]

            pred_image, pred_tensor = _select_prediction(pred_image, pred_tensor, sample_to_use, best_psnr_idx_per_sample)

            # MSE, PSNR, SSIM, LPIPS below all read from this SAME selected
            # (pred_image, pred_tensor) pair, so they always describe one
            # consistent prediction regardless of sample_to_use.
            batch_mse = F.mse_loss(clean_image, pred_image, reduction="mean").item()

            pred_image_np = pred_image.cpu().numpy()
            clean_image_np = clean_image.cpu().numpy()

            img_lpips = compute_lpips_batch(clean_tensor, pred_tensor)

            for b in range(batch_size):
                idx = batch_idx * loader.batch_size + b
                all_tensor_lpips[idx, :] = img_lpips[b]

            img_psnr = [0.0] * batch_size
            img_ssim = [0.0] * batch_size
            for b in range(batch_size):
                img_psnr[b] = psnr(clean_image_np[b], pred_image_np[b], data_range=1)
                img_ssim[b] = ssim(clean_image_np[b], pred_image_np[b], data_range=1)

            noisy_image = ((noisy_tensor + 1.0) * 127.5).clamp(0, 255.0)
            noisy_image = torch.round(torch.mean(noisy_image, dim=1)) / 255.0
            noisy_image = noisy_image.contiguous()
            noisy_image_np = noisy_image.cpu().numpy()

            clean_image_np *= 255.0
            noisy_image_np *= 255.0
            pred_image_np *= 255.0

            for i in range(batch_size):
                if images_dir is not None:
                    save_filename = os.path.basename(image_filename[i])

                    if save_images:
                        save_test_images(
                            os.path.join(images_dir, save_filename),
                            noisy_image_np[i], pred_image_np[i], clean_image_np[i],
                        )
                        save_paper_images(
                            images_dir, save_filename,
                            noisy_image_np[i], pred_image_np[i], clean_image_np[i],
                        )
                    csv_writer.writerow([
                        save_filename,
                        f"{img_psnr[i]:.4f}",
                        f"{img_ssim[i]:.6f}",
                        f"{img_lpips[i]:.6f}",
                        f"{elapsed_time / batch_size:.4f}",
                    ])

                if log:
                    num_digits = int(math.log10(len(loader))) + 1 if len(loader) != 0 else 1
                    status = f"[{(batch_idx+1):>{num_digits}d}/{len(loader)}]  "
                    status += f"PSNR: {img_psnr[i]:5.2f} dB,  "
                    status += f"SSIM: {img_ssim[i]:5.3f},  "
                    status += f"Max PSNR: {best_psnr_per_sample[i]:5.2f} at {best_psnr_idx_per_sample[i]:2d}  |  "
                    status += f"[{elapsed_time/batch_size:3.1f}s]  {os.path.basename(image_filename[i])}"
                    logger.log(status)

                    logger.logkv('Time', elapsed_time / batch_size)
                    logger.logkv('PSNR', img_psnr[i])
                    logger.logkv('SSIM', img_ssim[i])
                    logger.logkv('LPIPS', img_lpips[i])
                    logger.dumpkvs()

            batch_psnr = sum(img_psnr) / batch_size
            batch_ssim = sum(img_ssim) / batch_size
            batch_lpips = sum(img_lpips) / batch_size
            net_psnr += batch_psnr
            net_ssim += batch_ssim
            net_lpips += batch_lpips
            net_mse += batch_mse

            progress_bar.set_description(
                desc=f"[{'Test' if test else 'Validation'}] "
                     f"PSNR: {batch_psnr:5.2f}/{(net_psnr/(batch_idx+1)):5.2f}, "
                     f"SSIM: {batch_ssim:5.3f}/{(net_ssim/(batch_idx+1)):5.3f}"
            )

            del pred_tensor, pred_image, clean_image, noisy_image

        progress_bar.close()

    if not test:
        model.train()

    net_time /= len(loader)
    net_psnr /= len(loader)
    net_ssim /= len(loader)
    net_lpips /= len(loader)
    net_mse /= len(loader)

    if log:
        logger.log(f"\nAverage elapsed time: {net_time:.3f} s")
        logger.log(f"Average PSNR: {net_psnr:.3f} dB")
        logger.log(f"Average SSIM: {net_ssim:.3f}")
        logger.log(f"Average MSE: {net_mse:2.2e}")
        logger.log(f"Average LPIPS: {net_lpips:.4f}")

    # Plot the metrics over all diffusion iterations (T == 1 under DDIM).
    itr_indexes = list(range(all_tensor_psnr.shape[1]))

    psnr_means = np.mean(all_tensor_psnr, axis=0)
    max_psnr_index = np.argmax(psnr_means)
    max_psnr = psnr_means[max_psnr_index]
    if test:
        logger.log(f"Average best PSNR: {max_psnr:5.2f} dB with average index of {max_psnr_index}.")

    ssim_means = np.mean(all_tensor_ssim, axis=0)
    max_ssim_index = np.argmax(ssim_means)
    max_ssim = ssim_means[max_ssim_index]

    lpips_means = np.mean(all_tensor_lpips, axis=0)
    min_lpips_index = np.argmin(lpips_means)
    min_lpips = lpips_means[min_lpips_index]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(itr_indexes, psnr_means, label='PSNR', marker='o', color='blue', markersize=4)
    ax1.scatter(itr_indexes[max_psnr_index], max_psnr, color='blue', s=50, zorder=5, label='Max PSNR')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('PSNR (dB)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(itr_indexes, ssim_means, label='SSIM', marker='o', color='orange', markersize=4)
    ax2.scatter(itr_indexes[max_ssim_index], max_ssim, color='orange', s=50, zorder=5, label='Max SSIM')
    ax2.plot(itr_indexes, lpips_means, label='LPIPS', marker='o', color='red', markersize=4)
    ax2.scatter(itr_indexes[min_lpips_index], min_lpips, color='red', s=50, zorder=5, label='Min LPIPS')
    ax2.set_ylabel('SSIM / LPIPS', color='black')
    ax2.set_ylim(0, 0.9)
    ax2.tick_params(axis='y', labelcolor='black')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.1, 0.7), title="Legend")

    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.title('Change in PSNR and SSIM per diffusion iteration.')
    plt.savefig(os.path.join(logger.get_dir(), 'PSNR_plot.png'))
    plt.close(fig)

    if images_dir is not None:
        summary_path = os.path.join(images_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("===== Evaluation Summary =====\n\n")
            f.write(f"Average PSNR   : {net_psnr:.4f}\n")
            f.write(f"Average SSIM   : {net_ssim:.6f}\n")
            f.write(f"Average Runtime: {net_time:.4f} sec\n")
            f.write(f"Average MSE    : {net_mse:.6f}\n")
            f.write(f"Best Avg PSNR  : {max_psnr:.4f}\n")

    if csv_file is not None:
        csv_file.close()

    return net_psnr, net_ssim, net_time, net_mse, max_psnr


def compute_lpips_batch(sr_tensors, gt_tensors):
    """
    Compute per-sample LPIPS for a batch of tensors already in [-1, 1].

    Accepts [N, H, W] or [N, C, H, W] for any C (C == 1, C == 3, and
    C == 6 dual-pol SAR are all handled below; other channel counts are
    reduced to grayscale then broadcast to RGB, since LPIPS's backbone is
    only defined for 3-channel input).
    """

    def _to_rgb(x):
        if x.dim() == 3:  # [N, H, W] -> [N, 1, H, W]
            x = x.unsqueeze(1)
        c = x.shape[1]
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

    sr_tensors = _to_rgb(sr_tensors)
    gt_tensors = _to_rgb(gt_tensors)

    assert sr_tensors.dim() == 4, sr_tensors.shape
    assert gt_tensors.dim() == 4, gt_tensors.shape
    assert sr_tensors.shape[1] == 3, sr_tensors.shape
    assert gt_tensors.shape[1] == 3, gt_tensors.shape

    lpips_model = _get_lpips_model(sr_tensors.device)
    lpips_values = lpips_model(sr_tensors, gt_tensors)
    lpips_values = lpips_values.squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu().tolist()

    if not isinstance(lpips_values, list):
        lpips_values = [lpips_values]

    return lpips_values


def _make_strip(images, border=2, mode="L"):
    """
    Horizontally concatenate a list of same-size PIL Images with a thin
    border between them. Shared by save_test_images (legacy comparison
    strip) and save_paper_images (publication panel).
    """
    heights = {img.size[1] for img in images}
    if len(heights) != 1:
        raise ValueError("All images in a strip must share the same height.")

    width, height = images[0].size
    total_width = width * len(images) + border * (len(images) - 1)

    strip = Image.new(mode, (total_width, height))
    for i, img in enumerate(images):
        strip.paste(img, (i * (width + border), 0))
    return strip


def save_test_images(img_name, *arrays):
    """Save a simple side-by-side [noisy | prediction | clean] comparison strip."""
    images = [Image.fromarray(arr) for arr in arrays]
    _make_strip(images, border=2, mode="L").save(img_name)


def save_paper_images(output_dir, image_name, noisy_img, pred_img, clean_img):
    """
    Save publication-quality per-image outputs.

    Directory layout:
        output_dir/noisy/<image_name>
        output_dir/prediction/<image_name>
        output_dir/clean/<image_name>
        output_dir/difference/<image_name>
        output_dir/panel/<image_name>   (noisy | prediction | clean | |pred-clean|, side by side)
    """
    folders = {
        name: os.path.join(output_dir, name)
        for name in ("noisy", "prediction", "clean", "difference", "panel")
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    noisy_img = noisy_img.astype(np.uint8)
    pred_img = pred_img.astype(np.uint8)
    clean_img = clean_img.astype(np.uint8)
    diff_img = np.abs(pred_img.astype(np.int16) - clean_img.astype(np.int16)).astype(np.uint8)

    Image.fromarray(noisy_img).save(os.path.join(folders["noisy"], image_name))
    Image.fromarray(pred_img).save(os.path.join(folders["prediction"], image_name))
    Image.fromarray(clean_img).save(os.path.join(folders["clean"], image_name))
    Image.fromarray(diff_img).save(os.path.join(folders["difference"], image_name))

    panel_images = [Image.fromarray(a) for a in (noisy_img, pred_img, clean_img, diff_img)]
    _make_strip(panel_images, border=3, mode="L").save(os.path.join(folders["panel"], image_name))


def evaluate_sar(model, device, num_channels, image_size):
    logger.log("Creating data loader...")

    sample_size = 3

    for sar_list_path, dataset_name in zip([sen12_sar_list_path, hrsid_sar_list_path], ["sen12", "hrsid"]):
        logger.log("Creating " + dataset_name + " data loader...")
        test_dataset = SARDataset(sar_list_path, num_channels=num_channels, crop_size=(image_size, image_size))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

        images_folder = os.path.join(logger.get_dir(), dataset_name + "_clean_samples")
        os.makedirs(images_folder, exist_ok=True)
        combined_images_folder = os.path.join(logger.get_dir(), dataset_name + "_clean_samples", "combined_samples")
        os.makedirs(combined_images_folder, exist_ok=True)

        logger.log("Beginning " + dataset_name + " testing...")
        running_enl = 0.0
        running_epd_h = 0.0
        running_epd_v = 0.0
        running_time = 0.0
        enl_cv = 0.0
        epd_cv = 0.0
        progress_bar = tqdm(
            test_loader,
            desc=f"[Test - {dataset_name}] ENL: 0000.00/0000.00, EPD-ROA_H: 0.000/0.000, EPD-ROA_V: 0.000/0.000",
            unit='image',
        )

        with torch.no_grad():
            for batch_idx, (noisy_tensor, hm_coords, ht_coords, fnames) in enumerate(progress_bar):
                sample_enl = [0.0] * sample_size
                sample_epd_h = [0.0] * sample_size
                sample_epd_v = [0.0] * sample_size
                sample_time = [0.0] * sample_size
                for i in range(sample_size):
                    noisy_image = ((noisy_tensor + 1.0) * 127.5).clamp(0, 255.0)
                    noisy_image = torch.round(torch.mean(noisy_image, dim=1)) / 255.0
                    noisy_image = noisy_image.contiguous()

                    start_time = time.time()
                    pred_tensor = model(noisy_tensor.to(device))
                    sample_time[i] = (time.time() - start_time)

                    pred_image = ((pred_tensor + 1.0) * 127.5).clamp(0, 255.0)
                    pred_image = torch.round(torch.mean(pred_image, dim=1)) / 255.0
                    pred_image = pred_image.contiguous().cpu()

                    pil_image = transforms.ToPILImage()(pred_image)
                    image_path = os.path.join(images_folder, f'{fnames[0]}_{i}.png')
                    pil_image.save(image_path)

                    concatenated_image = torch.cat((noisy_image, pred_image), dim=2)
                    pil_image = transforms.ToPILImage()(concatenated_image)
                    image_path = os.path.join(combined_images_folder, f'{fnames[0]}_{i}.png')
                    pil_image.save(image_path)

                    noisy_image, pred_image = noisy_image.numpy(), pred_image.numpy()

                    img_enl = enl(pred_image, hm_coords)
                    img_epd_h, img_epd_v = epd_roa(noisy_image, pred_image, ht_coords)

                    sample_enl[i] = img_enl
                    sample_epd_h[i] = img_epd_h
                    sample_epd_v[i] = img_epd_v

                mean_enl = np.mean(sample_enl)
                mean_epd_h = np.mean(sample_epd_h)
                mean_epd_v = np.mean(sample_epd_v)

                running_enl += mean_enl
                running_epd_h += mean_epd_h
                running_epd_v += mean_epd_v
                running_time += np.mean(sample_time)

                enl_cv += np.std(sample_enl) / mean_enl
                epd_cv += (np.std(sample_epd_h) / mean_epd_h + np.std(sample_epd_v) / mean_epd_v) / 2

                enl_sample_string = f"{mean_enl:7.1f}["
                epd_h_sample_string = f"{mean_epd_h:5.3f}["
                epd_v_sample_string = f"{mean_epd_v:5.3f}["
                for i in range(sample_size):
                    enl_sample_string += f"{sample_enl[i]:7.1f},"
                    epd_h_sample_string += f"{sample_epd_h[i]:5.3f},"
                    epd_v_sample_string += f"{sample_epd_v[i]:5.3f},"
                enl_sample_string = enl_sample_string[:-1] + "]"
                epd_h_sample_string = epd_h_sample_string[:-1] + "]"
                epd_v_sample_string = epd_v_sample_string[:-1] + "]"

                logger.log(
                    f"[{batch_idx+1:2d}/{len(test_loader):2d}] ENL: {enl_sample_string}, "
                    f"EPD-ROA_H: {epd_h_sample_string}, EPD-ROA_V: {epd_v_sample_string}, | {fnames[0]}"
                )

                progress_bar.set_description(
                    desc=f"[Test - {dataset_name}] ENL: {mean_enl:7.2f}/{(running_enl/(batch_idx+1)):7.2f}, "
                         f"EPD-ROA_H: {mean_epd_h:5.3f}/{(running_epd_h/(batch_idx+1)):5.3f}, "
                         f"EPD-ROA_V: {mean_epd_v:5.3f}/{(running_epd_v/(batch_idx+1)):5.3f}"
                )

        progress_bar.close()

        logger.log(f"Testing for {dataset_name} complete!")
        logger.log(f"Average ENL: {(running_enl/(len(test_loader))):5.2f}")
        logger.log(f"Average ENL CV: {(enl_cv/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD (H): {(running_epd_h/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD (V): {(running_epd_v/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD: {((running_epd_h+running_epd_v)/(2*len(test_loader))):5.3f}")
        logger.log(f"Average EPD CV: {(epd_cv/(len(test_loader))):5.3f}")
        logger.log(f"Average time (secs): {(running_time/(len(test_loader))):5.3f}\n")


def enl(pred_image, hm_coords):
    """
    Calculate the Equivalent Number of Looks (ENL) for a specified rectangle in the image.

    Parameters:
        pred_image (numpy.ndarray): A 2D numpy array with values between 0.0 and 255.0.
        hm_coords (tuple): A tuple (x1, y1, x2, y2) defining the rectangle in the image.
                           (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.

    Returns:
        float: The ENL value for the specified rectangle. A perfectly flat
        (zero-variance) region is a plausible outcome of a strong despeckler
        and is reported as +inf rather than raising.
    """
    x1, y1, x2, y2 = hm_coords

    roi = pred_image[:, y1:y2 + 1, x1:x2 + 1] / 255.0

    mean_value = np.mean(roi)
    std_dev = np.std(roi)

    if std_dev == 0:
        return float("inf")

    return (mean_value / std_dev) ** 2


def epd_roa(noisy_image, pred_image, ht_coords):
    """
    Calculate the Edge Preservation Degree using the Ratio of Averages (EPD-ROA)
    for a specific region of interest (ROI) defined by ht_coords, for images in the
    format [1, height, width].

    Parameters:
        noisy_image (np.ndarray): Noisy image with shape [1, height, width], values between 0.0 and 1.0.
        pred_image (np.ndarray): Processed image with shape [1, height, width], values between 0.0 and 1.0.
        ht_coords (tuple): Tuple (x1, y1, x2, y2) defining the region of interest (ROI).

    Returns:
        (float, float): Edge Preservation Degree (EPD-ROA) (horizontal, vertical).
    """
    x1, y1, x2, y2 = ht_coords

    noisy_roi = noisy_image[0, y1:y2 + 1, x1:x2 + 1]
    pred_roi = pred_image[0, y1:y2 + 1, x1:x2 + 1]

    def _quotient_sum(roi, axis):
        if axis == "h":
            numer, denom = roi[:, :-1], roi[:, 1:]
        else:
            numer, denom = roi[:-1, :], roi[1:, :]
        quotient = np.divide(
            numer, denom,
            out=np.zeros_like(numer, dtype=np.float64),
            where=denom != 0,
        )
        return quotient.sum()

    def _directional_epd(axis):
        sum_noisy = _quotient_sum(noisy_roi, axis)
        sum_pred = _quotient_sum(pred_roi, axis)
        return float(sum_pred / sum_noisy) if sum_noisy != 0 else 0.0

    return _directional_epd("h"), _directional_epd("v")
