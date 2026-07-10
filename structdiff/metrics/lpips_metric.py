"""
LPIPS metric for SAR-DDPM evaluation.

Lower is better.
"""

from __future__ import annotations

import torch
import lpips


class LPIPSMetric:
    """
    Computes average LPIPS.
    """

    def __init__(self):

        self.metric = lpips.LPIPS(
            net="alex"
        )

        self.metric.eval()

        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):

        pred = pred.float()
        target = target.float()

        score = self.metric(
            pred,
            target,
        ).mean()

        self.total += score.item()
        self.count += 1

    def compute(self):

        if self.count == 0:
            return 0.0

        return self.total / self.count
