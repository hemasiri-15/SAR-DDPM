"""
PSNR metric for SAR-DDPM evaluation.

Higher is better.
"""

from __future__ import annotations

import math
import torch


class PSNR:
    """
    Computes average PSNR over multiple batches.
    """

    def __init__(self, data_range: float = 2.0):
        self.data_range = data_range
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):

        pred = pred.float()
        target = target.float()

        mse = torch.mean((pred - target) ** 2)

        if mse.item() == 0:
            psnr = float("inf")
        else:
            psnr = 20.0 * math.log10(self.data_range) - 10.0 * math.log10(mse.item())

        self.total += psnr
        self.count += 1

    def compute(self):

        if self.count == 0:
            return 0.0

        return self.total / self.count
