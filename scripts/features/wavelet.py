"""
WaveletCondition (A34-adjacent — wavelet conditioning, paired with
WaveletConsistencyLoss already used in train_util.py).

ASSUMPTION FLAGGED FOR REVIEW:
Uses a single-level 2D discrete wavelet transform (default: Haar / 'db1')
via PyWavelets, stacking the four subbands (LL, LH, HL, HH) into a
4-channel tensor. If your existing wavelet module uses a different
wavelet family, decomposition level, or only keeps high-frequency
subbands, replace `compute` below — this is the integration point.

Requires: pip install PyWavelets
"""

import numpy as np
import torch

try:
    import pywt
except ImportError as e:
    raise ImportError(
        "WaveletCondition requires PyWavelets. Install with: "
        "pip install PyWavelets --break-system-packages"
    ) from e


class WaveletCondition:
    def __init__(self, wavelet: str = "haar"):
        self.wavelet = wavelet

    def compute(self, clean_array: np.ndarray) -> dict:
        """
        clean_array: normalized single-channel array, shape (1, H, W).

        Returns:
            {"wavelet_tensor": Tensor (4, H/2, W/2)} — stacked
            [LL, LH, HL, HH] subbands from a single-level 2D DWT.
        """
        img = clean_array[0]  # (H, W)
        ll, (lh, hl, hh) = pywt.dwt2(img, self.wavelet)

        stacked = np.stack([ll, lh, hl, hh], axis=0).astype(np.float32)
        wavelet_tensor = torch.from_numpy(stacked)
        return {"wavelet_tensor": wavelet_tensor}
