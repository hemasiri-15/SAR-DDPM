"""
StructureCondition — multi-scale structure tensor conditioning.

ASSUMPTION FLAGGED FOR REVIEW:
train_util.py already instantiates
    MultiScaleStructureConsistencyLoss(kernels=(3, 5, 9))
so this implementation computes the structure tensor at Gaussian smoothing
scales (3, 5, 9) to stay consistent with that loss. If your existing
structure_tensor.py uses different kernel sizes, a different smoothing
kernel (box vs Gaussian), or eigenvalue/orientation-based features instead
of raw Jxx/Jxy/Jyy, replace `_structure_tensor_at_scale` below with the
exact function from that file — this is the module I'd wire it into.

Structure tensor definition used here (standard formulation):
    Ix, Iy = image gradients (Sobel)
    Jxx = G_sigma * (Ix * Ix)
    Jxy = G_sigma * (Ix * Iy)
    Jyy = G_sigma * (Iy * Iy)
Each scale's tensor is returned as a 3-channel map [Jxx, Jxy, Jyy].
"""

import numpy as np
import torch
import torch.nn.functional as F


def _sobel_kernels():
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T.copy()
    return kx, ky


def _gaussian_kernel(size: int, sigma: float = None) -> np.ndarray:
    if sigma is None:
        sigma = size / 3.0
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


class StructureCondition:
    def __init__(self, kernels=(3, 5, 9)):
        self.kernels = kernels
        kx, ky = _sobel_kernels()
        self._sobel_x = torch.from_numpy(kx).view(1, 1, 3, 3)
        self._sobel_y = torch.from_numpy(ky).view(1, 1, 3, 3)
        self._gauss = {
            k: torch.from_numpy(_gaussian_kernel(k)).view(1, 1, k, k)
            for k in kernels
        }

    def _structure_tensor_at_scale(self, img: torch.Tensor, k: int) -> torch.Tensor:
        # img: (1, 1, H, W), single channel, float32
        pad = 1
        ix = F.conv2d(img, self._sobel_x, padding=pad)
        iy = F.conv2d(img, self._sobel_y, padding=pad)

        jxx = ix * ix
        jxy = ix * iy
        jyy = iy * iy

        gpad = k // 2
        gk = self._gauss[k]
        jxx = F.conv2d(jxx, gk, padding=gpad)
        jxy = F.conv2d(jxy, gk, padding=gpad)
        jyy = F.conv2d(jyy, gk, padding=gpad)

        return torch.cat([jxx, jxy, jyy], dim=1).squeeze(0)  # (3, H, W)

    def compute(self, clean_array: np.ndarray) -> dict:
        """
        clean_array: normalized single-channel array, shape (1, H, W),
            same array used to build clean_tensor in the dataset.

        Returns:
            {
              "struct_tensor": Tensor (3, H, W)   -- scale kernels[0], i.e. S1
              "struct_tensors": (S1, S2, S3)       -- one per kernel size
            }
        """
        img = torch.from_numpy(clean_array).float().unsqueeze(0)  # (1, 1, H, W)
        scales = [self._structure_tensor_at_scale(img, k) for k in self.kernels]
        return {
            "struct_tensor": scales[0],
            "struct_tensors": tuple(scales),
        }
