import torch

from structdiff.metrics.evaluator import Evaluator

metric = Evaluator()

pred = torch.randn(2,1,64,64)
gt = pred.clone()

metric.update(pred, gt)

print(metric.compute())
