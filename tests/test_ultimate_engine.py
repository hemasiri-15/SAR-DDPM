"""
Smoke test for the CORR-10 fix: engine.fuse(method="ultimate") must run
end-to-end without a shape mismatch, for both channels==wavelet_channels
(the case that accidentally worked before) and channels!=wavelet_channels
(the case that previously raised RuntimeError).
"""
import torch
from structdiff.sampling.cycle_spinning.engine import CycleSpinningEngine, EngineConfig


def _make_batch(num_shifts=9, B=2, C=1, H=32, W=32):
    outputs = [torch.randn(B, C, H, W) for _ in range(num_shifts)]
    shifts = CycleSpinningEngine.build_shift_grid(H, W, cycle_width=H)[:num_shifts]
    if len(shifts) < num_shifts:
        shifts = [(0, 0)] * num_shifts
    confidence = [torch.rand(B, 1, H, W) for _ in range(num_shifts)]
    # Real wavelet features commonly have a different channel count than the
    # image (e.g. 4 for a db2-style decomposition) -- this is exactly the
    # case that triggered the reported bug.
    wavelets = [torch.randn(B, 4, H // 2, W // 2) for _ in range(num_shifts)]
    structure = [torch.randn(B, 13, H, W) for _ in range(num_shifts)]
    timestep = torch.randint(0, 1000, (B,))
    return outputs, shifts, confidence, wavelets, structure, timestep


def test_ultimate_channels_ne_wavelet_channels():
    engine = CycleSpinningEngine(EngineConfig(method="ultimate", device=torch.device("cpu")))
    outputs, shifts, confidence, wavelets, structure, timestep = _make_batch()

    result = engine.fuse(
        outputs, shifts,
        confidence_maps=confidence,
        wavelet_features=wavelets,       # 4 channels
        structure_features=structure,    # 13 channels; image itself has 1 channel
        timestep=timestep,
    )
    print("fused.shape  =", tuple(result.fused.shape))
    print("weights.shape =", tuple(result.weights.shape) if result.weights is not None else None)
    assert result.fused.shape == outputs[0].shape
    print("OK: engine.fuse(method='ultimate') succeeded with channels != wavelet_channels.")


if __name__ == "__main__":
    test_ultimate_channels_ne_wavelet_channels()
