"""
Unified evaluation metrics for SAR-DDPM.
"""

from __future__ import annotations

import torch

from structdiff.metrics.psnr import PSNR
from structdiff.metrics.ssim import SSIM
from structdiff.metrics.lpips_metric import LPIPSMetric


class Evaluator:

    def __init__(self):

        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPSMetric()

    def reset(self):

        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):

        self.psnr.update(pred, target)
        self.ssim.update(pred, target)

        # LPIPS expects RGB images
        if pred.shape[1] == 1:
            pred_lpips = pred.repeat(1, 3, 1, 1)
            target_lpips = target.repeat(1, 3, 1, 1)
        else:
            pred_lpips = pred
            target_lpips = target

        self.lpips.update(pred_lpips, target_lpips)

    def compute(self):

        return {
            "PSNR": self.psnr.compute(),
            "SSIM": self.ssim.compute(),
            "LPIPS": self.lpips.compute(),
        }
