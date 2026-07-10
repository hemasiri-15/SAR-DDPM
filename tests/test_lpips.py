import torch

from structdiff.metrics.lpips_metric import LPIPSMetric

metric = LPIPSMetric()

pred = torch.randn(2,3,64,64)
gt = pred.clone()

metric.update(pred, gt)

print(metric.compute())
