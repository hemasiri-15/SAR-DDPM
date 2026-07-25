import torch
from structdiff.sampling.cycle_spinning.engine import (
    CycleSpinningEngine,
    EngineConfig,
)

B, C, H, W = 2, 1, 32, 32
N = 9

engine = CycleSpinningEngine(
    EngineConfig(
        method="ultimate",
        device=torch.device("cpu"),
    )
)

outputs = [
    torch.randn(B, C, H, W, requires_grad=True)
    for _ in range(N)
]

confidence = [
    torch.rand(B, 1, H, W)
    for _ in range(N)
]

wavelets = [
    torch.randn(B, 4, H // 2, W // 2)
    for _ in range(N)
]

structure = [
    torch.randn(B, 13, H, W)
    for _ in range(N)
]

shifts = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

timestep = torch.randint(0, 1000, (B,))

result = engine.fuse(
    outputs,
    shifts,
    confidence_maps=confidence,
    wavelet_features=wavelets,
    structure_features=structure,
    timestep=timestep,
)

print("result.fused.requires_grad =", result.fused.requires_grad)
print("result.fused.grad_fn =", result.fused.grad_fn)

loss = result.fused.mean()

loss.backward()

print("Backward succeeded.")

print(outputs[0].grad.abs().mean())

print(outputs[0].requires_grad)

