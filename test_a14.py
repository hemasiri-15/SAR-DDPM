import torch

from structdiff.sampling.config import SamplingConfig
from structdiff.sampling.confidence_guidance import ConfidenceGuidance

cfg = SamplingConfig()
cg = ConfidenceGuidance(cfg)

for i in range(5):
    conf = torch.rand(2,1,256,256)
    unc = torch.rand(2,1,256,256)

    gmap, state = cg.compute_guidance_map(
        conf,
        unc,
        device=torch.device("cpu")
    )

print("entropy =", cg.confidence_entropy())
print("variance =", cg.confidence_variance())
print("temporal variance =", cg.temporal_variance())
