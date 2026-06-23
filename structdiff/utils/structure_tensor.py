"""
structdiff/utils/structure_tensor.py
=====================================
A3: Structure Tensor — pure-NumPy computation utility.

Computes the 2×2 structure tensor J = Kσ * (∇I ⊗ ∇I) for a 2-D image,
returning the three unique components (J11, J12, J22) stacked as a
float32 array of shape [3, H, W].

This module has no PyTorch dependency and is designed to run in
DataLoader worker processes (CPU) at dataset time.  It is fully
unit-testable in isolation.

Pipeline
--------
1. Pre-smooth I with Gaussian(ρ)   — suppresses speckle before differentiation
2. Compute Scharr gradients Ix, Iy  — numerically accurate finite differences
3. Form outer products P11, P12, P22
4. Integrate with Gaussian(σ)       — neighbourhood averaging
5. Stack and per-channel normalise  → [3, H, W] float32 in [-1, 1]

Default scales:  ρ = 1.0 pixel,  σ = 5.0 pixels.
These are ablation parameters; pass custom values via ``compute_structure_tensor``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Scharr kernel (more isotropic than Sobel for SAR)
# Applied as separable 1-D convolutions via scipy.
# ---------------------------------------------------------------------------

_SCHARR_X: np.ndarray = np.array(
    [[-3, 0, 3],
     [-10, 0, 10],
     [-3, 0, 3]], dtype=np.float32
) / 32.0

_SCHARR_Y: np.ndarray = _SCHARR_X.T.copy()


def _scharr_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Ix, Iy) via Scharr finite-difference filters.

    Parameters
    ----------
    image:
        2-D float32 array, shape [H, W].

    Returns
    -------
    Ix, Iy:
        Gradient arrays, each shape [H, W], float32.
    """
    from scipy.ndimage import convolve
    Ix = convolve(image, _SCHARR_X, mode="reflect").astype(np.float32)
    Iy = convolve(image, _SCHARR_Y, mode="reflect").astype(np.float32)
    return Ix, Iy


def _normalise_channel(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a 2-D array to [-1, 1] by its absolute maximum.

    If the array is all-zero (flat region), returns zeros.

    Parameters
    ----------
    arr:
        2-D float32 array.
    eps:
        Minimum denominator to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalised array in [-1, 1], same shape, float32.
    """
    abs_max = np.abs(arr).max()
    if abs_max < eps:
        return np.zeros_like(arr)
    return (arr / abs_max).astype(np.float32)


def compute_structure_tensor(
    image: np.ndarray,
    rho: float = 1.0,
    sigma: float = 5.0,
    normalise: bool = True,
) -> np.ndarray:
    """Compute the structure tensor of a 2-D image.

    Parameters
    ----------
    image:
        2-D float32 array of shape [H, W].  Expected range [0, 1] (amplitude
        image before [-1,1] normalisation).  Single-channel; do NOT pass
        [C, H, W] — squeeze first.
    rho:
        Pre-smoothing Gaussian std (pixels) applied before differentiation.
        Suppresses speckle.  Default 1.0.
    sigma:
        Integration Gaussian std (pixels) applied to outer-product components.
        Controls the neighbourhood size.  Default 5.0.
    normalise:
        If True (default), normalise each of J11, J12, J22 independently
        to [-1, 1] by its absolute maximum.  Keeps values in a consistent
        range regardless of image contrast.

    Returns
    -------
    np.ndarray
        Shape [3, H, W], dtype float32.
        Channel 0: J11  (horizontal gradient energy)
        Channel 1: J12  (cross-term / orientation coupling)
        Channel 2: J22  (vertical gradient energy)

    Raises
    ------
    ValueError
        If ``image`` is not 2-D or not float32-compatible.

    Examples
    --------
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> J = compute_structure_tensor(img)
    >>> J.shape
    (3, 256, 256)
    >>> J.dtype
    dtype('float32')
    >>> np.all(J >= -1.0) and np.all(J <= 1.0)
    True
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D [H, W], got shape {image.shape}. "
            "Squeeze the channel dimension before calling."
        )

    # ------------------------------------------------------------------
    # Step 1: Pre-smoothing (speckle suppression before differentiation)
    # ------------------------------------------------------------------
    if rho > 0.0:
        smoothed = gaussian_filter(image, sigma=rho, mode="reflect").astype(np.float32)
    else:
        smoothed = image

    # ------------------------------------------------------------------
    # Step 2: Scharr gradients
    # ------------------------------------------------------------------
    Ix, Iy = _scharr_gradients(smoothed)

    # ------------------------------------------------------------------
    # Step 3: Outer-product components
    # ------------------------------------------------------------------
    P11: np.ndarray = Ix * Ix   # [H, W]
    P12: np.ndarray = Ix * Iy   # [H, W]
    P22: np.ndarray = Iy * Iy   # [H, W]

    # ------------------------------------------------------------------
    # Step 4: Integration smoothing (neighbourhood averaging)
    # ------------------------------------------------------------------
    J11 = gaussian_filter(P11, sigma=sigma, mode="reflect").astype(np.float32)
    J12 = gaussian_filter(P12, sigma=sigma, mode="reflect").astype(np.float32)
    J22 = gaussian_filter(P22, sigma=sigma, mode="reflect").astype(np.float32)

    # ------------------------------------------------------------------
    # Step 5: Normalise and stack → [3, H, W]
    # ------------------------------------------------------------------
    if normalise:
        J11 = _normalise_channel(J11)
        J12 = _normalise_channel(J12)
        J22 = _normalise_channel(J22)

    return np.stack([J11, J12, J22], axis=0)  # [3, H, W]


# ---------------------------------------------------------------------------
# Derived features (optional, for analysis / ablation)
# ---------------------------------------------------------------------------

def structure_tensor_features(
    J: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute derived scalar features from the structure tensor components.

    Parameters
    ----------
    J:
        Output of ``compute_structure_tensor`` — shape [3, H, W], un-normalised
        or normalised.  Components must be in the order (J11, J12, J22).
    eps:
        Regularisation for division-by-zero in coherence.

    Returns
    -------
    dict with keys:
        ``coherence``   — C ∈ [0, 1], 0=isotropic, 1=perfect edge
        ``orientation`` — θ ∈ [-π/2, π/2] in radians, dominant edge direction
        ``energy``      — S = J11 + J22, total gradient energy
        ``cornerness``  — E = sqrt(λ1 · λ2), junction strength
    """
    J11, J12, J22 = J[0], J[1], J[2]

    half_trace = (J11 + J22) / 2.0
    half_diff  = (J11 - J22) / 2.0
    discriminant = np.sqrt(np.maximum(half_diff ** 2 + J12 ** 2, 0.0))

    lambda1 = half_trace + discriminant  # ≥ lambda2
    lambda2 = half_trace - discriminant

    # Coherence: (λ1 - λ2)² / (λ1 + λ2)²
    denom = (lambda1 + lambda2) ** 2
    coherence = np.where(denom > eps, (lambda1 - lambda2) ** 2 / denom, 0.0)
    coherence = coherence.astype(np.float32)

    # Dominant orientation
    orientation = (0.5 * np.arctan2(2.0 * J12, J11 - J22)).astype(np.float32)

    # Energy (trace)
    energy = (J11 + J22).astype(np.float32)

    # Cornerness (geometric mean of eigenvalues)
    cornerness = np.sqrt(np.maximum(lambda1 * lambda2, 0.0)).astype(np.float32)

    return {
        "coherence":   coherence,
        "orientation": orientation,
        "energy":      energy,
        "cornerness":  cornerness,
    }
