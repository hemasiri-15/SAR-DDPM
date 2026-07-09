"""
structdiff/losses/wavelet_consistency_loss.py
===============================================
A34: Wavelet Subband Consistency Loss — frequency-domain speckle-aware
regulariser for SAR despeckling diffusion models.

Literature gap addressed
-------------------------
Frequency-domain SAR despeckling work (e.g. wavelet/frequency-adaptive
branches such as SAR-FDD and SAR-FAH, and wavelet-conditioned diffusion
encoders such as ECDM) consistently reports that speckle energy is
concentrated in the high-frequency detail subbands of a wavelet
decomposition, while the low-frequency approximation subband is
comparatively speckle-suppressed. Every method that exploits this fact
does so architecturally: a dedicated frequency-domain branch, a
wavelet-domain encoder, or a modified downsampling path. None of them
express the same physical prior as a pure *training-time loss* that can
be dropped onto an already-fixed, otherwise unmodified diffusion U-Net.

This module fills that gap: it is a differentiable, GPU-batched,
single-level Haar DWT applied identically to x_hat (the reconstructed
x0 estimate) and x_gt (the clean reference), with subband-specific
weighting reflecting the different physical roles of each band. No
U-Net input/output channel, block, or forward signature is touched —
the loss operates purely on the two RGB/greyscale tensors already
available in the training loop (mirrors the A5 / A33 integration
pattern in train_util.py exactly).

Mathematical foundation
------------------------
Single-level 2-D Haar DWT, computed as a fixed (non-trainable) depthwise
convolution with stride 2:

    LL = (low_row  ⊛ low_col ) ↓2      — approximation (speckle-suppressed)
    LH = (low_row  ⊛ high_col) ↓2      — horizontal detail
    HL = (high_row ⊛ low_col ) ↓2      — vertical detail
    HH = (high_row ⊛ high_col) ↓2      — diagonal detail (speckle-richest)

    low  = [1, 1] / sqrt(2)      (Haar scaling filter)
    high = [1, -1] / sqrt(2)     (Haar wavelet filter)

Loss terms (all L1, spatially averaged):

    L_LL     = E[ |LL(x_hat) − LL(x_gt)| ]
    L_detail = E[ |LH(x_hat) − LH(x_gt)| ]
             + E[ |HL(x_hat) − HL(x_gt)| ]
             + E[ |HH(x_hat) − HH(x_gt)| ]                (averaged over 3)

    L_wavelet = w_LL · L_LL + w_detail · L_detail

Default w_LL = 0.3, w_detail = 0.7: the approximation band is already
substantially constrained by the pixel-space diffusion loss and A33's
structure-tensor term, so it receives a smaller share of this term's
budget; the detail bands — where genuine texture and speckle compete —
receive the larger share, since that is precisely where existing
per-pixel losses under-constrain the model (a network can match L_LL
almost for free while still blurring or over-sharpening LH/HL/HH).

Why Haar (not db2, as used by the A12 conditioning-side wavelet
features)?
------------------------------------------------------------------
A12's ``wavelet_features.py`` deliberately chose db2 for dataset-time
*feature extraction*, where boundary artefacts feed into a learned
encoder and can be amortised over training. Here the transform sits
directly in a loss computed every step on both a leaf-like GT tensor
and an autograd-tracked prediction tensor; Haar's 2-tap filters give
an exact, alias-free, trivially differentiable stride-2 conv with no
periodization bookkeeping, at the cost of slightly coarser frequency
localisation — an acceptable trade for a loss term (as opposed to a
conditioning feature) where the model has three other structural
losses (A5, A33) already supplying fine-grained gradient information.

Integration point in train_util.py
------------------------------------
    # AFTER A5 edge-loss block, BEFORE mp_trainer.backward(loss):

    if self.lambda_wavelet > 0.0:
        wavelet_loss = self.wavelet_loss_fn(
            x_pred=x0_hat,
            x_gt=micro.float().detach(),
        )
        loss = loss + self.lambda_wavelet * wavelet_loss
        logger.logkv_mean("wavelet_loss", wavelet_loss.item())

Design contracts (mirrors A5 / A33)
-------------------------------------
- gaussian_diffusion.py and unet.py are NOT modified.
- Reuses x0_hat already computed for A33 — zero extra UNet forward passes.
- Fixed (non-trainable) filter bank, registered as buffers.
- Channel-agnostic: depthwise conv over all input channels.
- AMP-safe: internal computation in float32.
- No in-place ops (autograd-safe).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public constant — imported by train_util.py
# ---------------------------------------------------------------------------

#: Default lambda_wavelet scaling factor for the combined loss.
#: Set to 0.0 in train_util.py to disable A34 entirely.
#: Chosen below lambda_struct=0.1 and lambda_edge=0.05: the wavelet term
#: overlaps partially with A5/A33 (both already penalise high-frequency
#: mismatch), so a conservative weight avoids double-counting gradients.
DEFAULT_LAMBDA_WAVELET: float = 0.04


def _make_haar_filters() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the 1-D Haar low-pass and high-pass filters.

    Returns
    -------
    low, high : torch.Tensor
        Float32 tensors, shape [2] each: [1/sqrt(2), 1/sqrt(2)] and
        [1/sqrt(2), -1/sqrt(2)].
    """
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    low = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float32)
    high = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=torch.float32)
    return low, high


def _make_haar_kernels() -> torch.Tensor:
    """Build the four 2×2 separable Haar kernels (LL, LH, HL, HH).

    Returns
    -------
    torch.Tensor
        Shape [4, 1, 2, 2], stacked in order (LL, LH, HL, HH).
    """
    low, high = _make_haar_filters()

    # Outer products: rows × cols. Row filter first, column filter second.
    ll = torch.outer(low, low)    # low_row  ⊛ low_col
    lh = torch.outer(low, high)   # low_row  ⊛ high_col  (horizontal detail)
    hl = torch.outer(high, low)   # high_row ⊛ low_col   (vertical detail)
    hh = torch.outer(high, high)  # high_row ⊛ high_col  (diagonal detail)

    kernels = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
    return kernels.unsqueeze(1)  # [4, 1, 2, 2]


def _haar_dwt2_multichannel(
    image: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    """Compute a single-level 2-D Haar DWT for every input channel.

    Parameters
    ----------
    image : torch.Tensor
        [B, C, H, W] float32 tensor. H and W should be even; odd sizes
        are handled by truncating the last row/column (documented,
        deterministic, matches PyTorch's default stride-2 conv floor
        behaviour).
    kernels : torch.Tensor
        [4, 1, 2, 2] Haar kernel stack (LL, LH, HL, HH), on the same
        device/dtype as image.

    Returns
    -------
    torch.Tensor
        [B, 4*C, H/2, W/2] tensor. The 4 subbands for each input
        channel are laid out contiguously: channel c occupies output
        indices [4c : 4c+4] in order (LL, LH, HL, HH).
    """
    B, C, H, W = image.shape

    # Depthwise conv producing 4 subbands per input channel.
    # weight shape for groups=C: [C*4, 1, 2, 2] — repeat the 4 kernels
    # once per input channel.
    weight = kernels.repeat(C, 1, 1, 1)  # [4C, 1, 2, 2]
    out = F.conv2d(image, weight, stride=2, groups=C)  # [B, 4C, H/2, W/2]
    return out


class WaveletConsistencyLoss(nn.Module):
    """A34: Wavelet Subband Consistency Loss for SAR image despeckling.

    Penalises mismatch between the single-level Haar DWT subbands of
    the predicted x_0 and the ground-truth clean image, with separate
    weights for the approximation (LL) and detail (LH/HL/HH) bands.

    Parameters
    ----------
    w_ll : float
        Weight on the low-frequency approximation-band L1 term.
    w_detail : float
        Weight on the combined high-frequency detail-band L1 term
        (LH + HL + HH, averaged over the three bands).
    """

    def __init__(self, w_ll: float = 0.3, w_detail: float = 0.7) -> None:
        super().__init__()
        self.w_ll = w_ll
        self.w_detail = w_detail

        kernels = _make_haar_kernels()
        self.register_buffer("haar_kernels", kernels, persistent=False)

    def forward(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        """Compute the wavelet subband consistency loss.

        Parameters
        ----------
        x_pred : torch.Tensor
            [B, C, H, W] predicted x_0 (e.g. x0_hat from EpsInterceptHook
            + reconstruct_x0). Must remain in the autograd graph.
        x_gt : torch.Tensor
            [B, C, H, W] ground-truth clean image. Should be detached
            by the caller (mirrors A5 / A33 convention).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        assert x_pred.shape == x_gt.shape, (
            f"x_pred shape {tuple(x_pred.shape)} != x_gt shape "
            f"{tuple(x_gt.shape)}"
        )

        # Upcast to float32 for AMP safety (mirrors A5's convention).
        x_pred_f = x_pred.float()
        x_gt_f = x_gt.float()

        kernels = self.haar_kernels.to(device=x_pred_f.device, dtype=torch.float32)

        C = x_pred_f.shape[1]
        subbands_pred = _haar_dwt2_multichannel(x_pred_f, kernels)  # [B,4C,H/2,W/2]
        subbands_gt = _haar_dwt2_multichannel(x_gt_f, kernels)

        # Reshape to [B, C, 4, H/2, W/2] to index subbands cleanly.
        B, _, Hh, Ww = subbands_pred.shape
        subbands_pred = subbands_pred.view(B, C, 4, Hh, Ww)
        subbands_gt = subbands_gt.view(B, C, 4, Hh, Ww)

        ll_pred, lh_pred, hl_pred, hh_pred = (
            subbands_pred[:, :, 0],
            subbands_pred[:, :, 1],
            subbands_pred[:, :, 2],
            subbands_pred[:, :, 3],
        )
        ll_gt, lh_gt, hl_gt, hh_gt = (
            subbands_gt[:, :, 0],
            subbands_gt[:, :, 1],
            subbands_gt[:, :, 2],
            subbands_gt[:, :, 3],
        )

        l_ll = F.l1_loss(ll_pred, ll_gt)
        l_detail = (
            F.l1_loss(lh_pred, lh_gt)
            + F.l1_loss(hl_pred, hl_gt)
            + F.l1_loss(hh_pred, hh_gt)
        ) / 3.0

        loss = self.w_ll * l_ll + self.w_detail * l_detail
        return loss.to(x_pred.dtype)
