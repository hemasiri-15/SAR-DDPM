import torch
from structdiff.metrics.psnr import PSNR

metric = PSNR()

pred = torch.randn(2,1,64,64)
gt = pred.clone()

metric.update(pred, gt)

print(metric.compute())
