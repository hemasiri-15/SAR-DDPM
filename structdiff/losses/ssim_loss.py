"""
structdiff/losses/ssim_loss.py
================================
A36: Structural Similarity (SSIM) Loss for SAR image despeckling.

Rationale
---------
A33 (structure tensor) and A5 (Sobel edges) constrain local gradient
and orientation consistency; neither constrains *luminance/contrast*
consistency over a local window the way SSIM does. SSIM captures a
different, complementary perceptual axis — the literature's
recurring complaint about pixel-wise (L1/L2) losses is that they
correlate poorly with perceived structural fidelity even when
gradient-based terms are already present. This adds that missing
axis as a pure loss term; no network change.

Formulation
-----------
Standard windowed SSIM (Wang et al. 2004), computed with a Gaussian
window via depthwise convolution:

    SSIM(x,y) = [(2*mu_x*mu_y + C1)(2*sigma_xy + C2)] /
                [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]

    L_ssim = 1 - mean(SSIM(x_pred, x_gt))

Integration point in train_util.py (same pattern as A5/A33/A34):

    ssim_loss = self.ssim_loss_fn(x_pred=x0_hat, x_gt=x_gt)
    # then pass into KendallUncertaintyWeighting alongside struct/edge/wavelet,
    # or add lambda_ssim * ssim_loss directly if not using A35.

Design contracts
-----------------
- gaussian_diffusion.py and unet.py are NOT modified.
- Fixed (non-trainable) Gaussian window, registered as a buffer.
- Channel-agnostic (depthwise conv).
- Reuses x0_hat — zero extra UNet forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LAMBDA_SSIM: float = 0.08


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d / window_2d.sum()


class SSIMLoss(nn.Module):
    """A36: windowed SSIM loss, 1 - mean(SSIM(x_pred, x_gt))."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,   # images assumed in [-1, 1] (diffusion convention)
        c1_scale: float = 0.01,
        c2_scale: float = 0.03,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.C1 = (c1_scale * data_range) ** 2
        self.C2 = (c2_scale * data_range) ** 2

        window = _gaussian_window(window_size, sigma)  # [win, win]
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0), persistent=False)

    def _ssim_map(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        C = x_pred.shape[1]
        window = self.window.to(device=x_pred.device, dtype=torch.float32).repeat(C, 1, 1, 1)
        pad = self.window_size // 2

        mu_x = F.conv2d(x_pred, window, padding=pad, groups=C)
        mu_y = F.conv2d(x_gt, window, padding=pad, groups=C)

        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = F.conv2d(x_pred * x_pred, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(x_gt * x_gt, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x_pred * x_gt, window, padding=pad, groups=C) - mu_xy

        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return numerator / denominator.clamp_min(1e-8)

    def forward(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        assert x_pred.shape == x_gt.shape
        x_pred_f, x_gt_f = x_pred.float(), x_gt.float()
        ssim_map = self._ssim_map(x_pred_f, x_gt_f)
        loss = 1.0 - ssim_map.mean()
        return loss.to(x_pred.dtype)
