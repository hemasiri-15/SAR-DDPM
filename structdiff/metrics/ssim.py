"""
SSIM metric for SAR-DDPM evaluation.

Higher is better.
"""

from __future__ import annotations

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIM:
    """
    Computes average SSIM over multiple batches.
    """

    def __init__(self, data_range: float = 2.0):

        self.metric = StructuralSimilarityIndexMeasure(
            data_range=data_range
        )

        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):

        pred = pred.float()
        target = target.float()

        score = self.metric(pred, target).item()

        self.total += score
        self.count += 1

    def compute(self):

        if self.count == 0:
            return 0.0

        return self.total / self.count
