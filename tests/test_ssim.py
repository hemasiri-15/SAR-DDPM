import torch
from structdiff.metrics.ssim import SSIM

metric = SSIM()

pred = torch.randn(2,1,64,64)
gt = pred.clone()

metric.update(pred, gt)

print(metric.compute())
