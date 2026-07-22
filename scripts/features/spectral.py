"""
SpectralCondition (A12 — Frequency-aware conditioning).

ASSUMPTION FLAGGED FOR REVIEW:
Your memory notes reference A12 as "Frequency-Aware Conditioning with
wavelet encoder" — the wavelet half is split out into features/wavelet.py.
This module covers a Fourier-domain feature: log-magnitude spectrum,
which is a common speckle-relevant frequency feature since multiplicative
speckle noise has a distinct signature in log-power space. If your actual
A12 spectral feature is something else (e.g. radially-averaged power
spectral density, or a learned spectral encoder), replace `compute` below
with the exact logic — this is the single integration point.
"""

import numpy as np
import torch


class SpectralCondition:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute(self, clean_array: np.ndarray) -> dict:
        """
        clean_array: normalized single-channel array, shape (1, H, W).

        Returns:
            {"spectral_tensor": Tensor (1, H, W)} — log-magnitude of the
            2D FFT, shifted to center the DC component, min-max normalized
            to roughly [-1, 1] to match image tensor scale.
        """
        img = clean_array[0]  # (H, W)
        fft = np.fft.fftshift(np.fft.fft2(img))
        magnitude = np.abs(fft)
        log_mag = np.log(magnitude + self.eps)

        lo, hi = log_mag.min(), log_mag.max()
        if hi - lo > self.eps:
            norm = 2.0 * (log_mag - lo) / (hi - lo) - 1.0
        else:
            norm = np.zeros_like(log_mag)

        spectral_tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)
        return {"spectral_tensor": spectral_tensor}
