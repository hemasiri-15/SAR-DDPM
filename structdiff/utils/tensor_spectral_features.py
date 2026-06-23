"""
structdiff/utils/tensor_spectral_features.py
==============================================
A11: Tensor Eigenvalue Conditioning — pure-NumPy feature computation.

Computes per-pixel eigenvalue-derived features (lambda1, lambda2,
anisotropy, coherence) from an A10 structure tensor [3, H, W]
(channels J11, J12, J22), and concatenates these features across A10's
three scales into a single [12, H, W] array.

This module performs NO gradient computation and does NOT touch image
data at all -- it operates purely on the structure tensors (s1, s2, s3)
already produced by ``compute_structure_tensor_multiscale`` (A10,
structure_tensor_multiscale.py). A10's gradient computation is reused
unchanged; this module is a pure post-processing step on its outputs.

For a symmetric 2x2 matrix
    J = [[J11, J12],
         [J12, J22]]
the eigenvalues are
    lambda1 = half_trace + discriminant   (lambda1 >= lambda2)
    lambda2 = half_trace - discriminant
where
    half_trace   = (J11 + J22) / 2
    half_diff    = (J11 - J22) / 2
    discriminant = sqrt(half_diff^2 + J12^2)   (>= 0)

Derived features (per A11 spec):
    anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    coherence  = (lambda1 - lambda2)^2 / ((lambda1 + lambda2)^2 + eps)

Numerical note
--------------
A10's structure tensor channels are normalised INDEPENDENTLY per
channel to a bounded range (J11, J22 in [0, 1]; J12 in [-1, 1]; see
structure_tensor.py's _normalise_channel). This means
``lambda1 + lambda2 = J11 + J22`` can be very small in regions where
the average gradient energy is low but J12 (and hence the
discriminant) is not -- e.g. a faint, highly anisotropic edge. In such
regions, ``anisotropy`` and ``coherence`` as defined above can take
very large values even with ``eps=1e-8``.

To guard against this without changing the specified formulas, an
optional ``clip_range`` is applied to ``anisotropy`` and ``coherence``
AFTER computing them (default ``(-10.0, 10.0)``). Set
``clip_range=None`` to disable and inspect raw values during ablation.
``lambda1``/``lambda2`` are NOT clipped.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Default epsilon, as specified for A11 (anisotropy / coherence denominators).
_EPS: float = 1e-8

#: Default clip range applied to anisotropy/coherence after computation.
#: Flagged as a safety addition beyond the literal A11 spec -- see module
#: docstring "Numerical note". Set to None to disable.
_DEFAULT_CLIP_RANGE: Tuple[float, float] = (-10.0, 10.0)


def compute_spectral_features(
    J: np.ndarray,
    eps: float = _EPS,
    clip_range: Optional[Tuple[float, float]] = _DEFAULT_CLIP_RANGE,
) -> np.ndarray:
    """Compute (lambda1, lambda2, anisotropy, coherence) from one A10 structure tensor.

    Parameters
    ----------
    J:
        Structure tensor, shape [3, H, W], float32, channel order
        (J11, J12, J22) -- one of A10's s1/s2/s3 (already normalised
        to a bounded range by compute_structure_tensor_multiscale).
    eps:
        Denominator regulariser for anisotropy/coherence, as specified
        for A11. Default 1e-8.
    clip_range:
        If not None, ``(lo, hi)`` clip applied to anisotropy and
        coherence after computation (NOT to lambda1/lambda2). Default
        (-10.0, 10.0). See module docstring "Numerical note".

    Returns
    -------
    np.ndarray
        Shape [4, H, W], float32, channel order:
            0: lambda1   (lambda1 >= lambda2)
            1: lambda2
            2: anisotropy = (lambda1-lambda2) / (lambda1+lambda2+eps)
            3: coherence  = (lambda1-lambda2)^2 / ((lambda1+lambda2)^2+eps)

    Raises
    ------
    ValueError
        If ``J`` is not shape [3, H, W].
    """
    J = np.asarray(J, dtype=np.float32)
    if J.ndim != 3 or J.shape[0] != 3:
        raise ValueError(
            f"J must have shape [3, H, W] (J11, J12, J22), got shape {J.shape}."
        )

    J11, J12, J22 = J[0], J[1], J[2]

    half_trace = (J11 + J22) / 2.0
    half_diff = (J11 - J22) / 2.0
    discriminant = np.sqrt(np.maximum(half_diff ** 2 + J12 ** 2, 0.0))

    lambda1 = (half_trace + discriminant).astype(np.float32)  # >= lambda2
    lambda2 = (half_trace - discriminant).astype(np.float32)

    denom = lambda1 + lambda2  # == J11 + J22

    anisotropy = ((lambda1 - lambda2) / (denom + eps)).astype(np.float32)
    coherence = (((lambda1 - lambda2) ** 2) / (denom ** 2 + eps)).astype(np.float32)

    if clip_range is not None:
        lo, hi = clip_range
        anisotropy = np.clip(anisotropy, lo, hi).astype(np.float32)
        coherence = np.clip(coherence, lo, hi).astype(np.float32)

    return np.stack([lambda1, lambda2, anisotropy, coherence], axis=0)  # [4, H, W]


def compute_spectral_features_multiscale(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    eps: float = _EPS,
    clip_range: Optional[Tuple[float, float]] = _DEFAULT_CLIP_RANGE,
) -> np.ndarray:
    """Compute and concatenate spectral features across A10's three scales.

    Parameters
    ----------
    s1, s2, s3:
        A10 structure tensors (fine, medium, coarse scales), each
        shape [3, H, W], float32 -- the outputs of
        ``compute_structure_tensor_multiscale``.
    eps, clip_range:
        See ``compute_spectral_features``.

    Returns
    -------
    np.ndarray
        Shape [12, H, W], float32. Channel layout (4 channels per
        scale, concatenated fine -> medium -> coarse):
            [0:4]  = (lambda1, lambda2, anisotropy, coherence) for s1
            [4:8]  = (lambda1, lambda2, anisotropy, coherence) for s2
            [8:12] = (lambda1, lambda2, anisotropy, coherence) for s3
    """
    f1 = compute_spectral_features(s1, eps=eps, clip_range=clip_range)  # [4,H,W]
    f2 = compute_spectral_features(s2, eps=eps, clip_range=clip_range)  # [4,H,W]
    f3 = compute_spectral_features(s3, eps=eps, clip_range=clip_range)  # [4,H,W]
    return np.concatenate([f1, f2, f3], axis=0)  # [12, H, W]
