"""
structdiff/utils/wavelet_features.py
======================================
A12: Wavelet Features — pure-NumPy computation utility.

Computes a single-level 2-D discrete wavelet transform (DWT) of a SAR
amplitude image, returning the four subbands (LL, LH, HL, HH) stacked
as a float32 array of shape [4, H/2, W/2].

This module has no PyTorch dependency and is designed to run in
DataLoader worker processes (CPU) at dataset time, exactly like A3's
``structure_tensor.py`` and A10's ``structure_tensor_multiscale.py``.
It is fully unit-testable in isolation.

Mathematical background
------------------------
The single-level 2-D DWT decomposes an image I into four subbands by
applying a 1-D low-pass filter (scaling function) and a 1-D high-pass
filter (wavelet function) separably along rows and columns, then
downsampling by 2 along each axis:

    LL = (lowpass_rows  * lowpass_cols ) downsampled    — approximation
    LH = (lowpass_rows  * highpass_cols) downsampled    — horizontal detail
    HL = (highpass_rows * lowpass_cols ) downsampled    — vertical detail
    HH = (highpass_rows * highpass_cols) downsampled    — diagonal detail

LL is a coarse, low-frequency approximation of I (half resolution).
LH, HL, HH are high-frequency detail coefficients capturing edges and
texture in the horizontal, vertical, and diagonal directions
respectively. Speckle noise in SAR imagery is predominantly a
high-frequency, low-coherence phenomenon, so LH/HL/HH carry strong
speckle content alongside genuine edge information; LL is comparatively
speckle-suppressed by construction (it is a smoothed, downsampled copy
of the input).

Wavelet basis and boundary handling
-------------------------------------
Default wavelet: ``db2`` (Daubechies-2, 4-tap filters). Chosen for its
balance of compact support (short filter, low computational cost) and
smoothness (one vanishing moment more than Haar), which better
separates true edge energy from speckle-induced high-frequency noise
than the discontinuous Haar basis.

Boundary mode: ``periodization``. Standard ``symmetric``/``reflect``
boundary extension in PyWavelets produces a coefficient array of
length ``floor((H + filter_len - 1) / 2)``, which is NOT exactly
``H / 2`` for any filter with ``filter_len > 2`` (i.e. anything except
Haar). ``periodization`` mode is specifically designed to guarantee
exact output length ``H / 2`` for even ``H`` regardless of filter
length, which is required here because every downstream consumer
(``WaveletEncoder``, the A12 dataset collation) expects exactly
``[4, H/2, W/2]`` with no cropping or padding logic of its own.

This module has been verified empirically: for ``db2`` and input sizes
64, 128, 256, 512 (all even, matching ``StructTensorDataset``'s
``crop_size`` convention), ``periodization`` mode yields exactly
``(H/2, W/2)`` coefficient arrays.

Pipeline
--------
1. Validate input is 2-D with even H and W.
2. Single-level ``pywt.dwt2`` with the chosen wavelet and
   ``mode="periodization"``.
3. Stack subbands in channel order (LL, LH, HL, HH).
4. Per-channel independent normalisation to [-1, 1] (mirrors A3's
   ``_normalise_channel`` convention in ``structure_tensor.py``).
"""

from __future__ import annotations

from typing import Final, Tuple

import numpy as np
import pywt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default wavelet basis. Daubechies-2: 4-tap filters, one vanishing moment
#: beyond Haar, compact support. See module docstring for justification.
DEFAULT_WAVELET: Final[str] = "db2"

#: Boundary extension mode. REQUIRED to be "periodization" — this is the
#: only PyWavelets mode that guarantees exact [H/2, W/2] output for even
#: H, W with filter lengths > 2 (i.e. for any wavelet other than Haar).
#: See module docstring "Boundary mode" section.
DWT_MODE: Final[str] = "periodization"

#: Number of output subbands (LL, LH, HL, HH).
NUM_SUBBANDS: Final[int] = 4

#: Channel index labels, for documentation and external reference.
SUBBAND_NAMES: Final[Tuple[str, str, str, str]] = ("LL", "LH", "HL", "HH")


# ---------------------------------------------------------------------------
# Helper: per-channel normalisation
# ---------------------------------------------------------------------------


def _normalise_subband(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a 2-D subband array to [-1, 1] by its absolute maximum.

    Mirrors ``structure_tensor.py``'s ``_normalise_channel`` convention
    (A3) so that all conditioning tensors in the StructDiff-SAR pipeline
    share the same per-channel scaling rule.

    If the array is all-zero (or below ``eps`` in absolute value
    everywhere — e.g. a perfectly flat LL band on a constant-intensity
    crop), returns zeros rather than dividing by a near-zero value.

    Parameters
    ----------
    arr:
        2-D float32 array, one wavelet subband.
    eps:
        Minimum absolute-maximum threshold below which the array is
        treated as flat and zeros are returned instead of normalising.

    Returns
    -------
    np.ndarray
        Normalised array in [-1, 1], same shape, dtype float32.

    Examples
    --------
    >>> arr = np.array([[2.0, -4.0], [1.0, 0.0]], dtype=np.float32)
    >>> out = _normalise_subband(arr)
    >>> bool(out.max() <= 1.0 and out.min() >= -1.0)
    True
    >>> _normalise_subband(np.zeros((2, 2), dtype=np.float32))
    array([[0., 0.],
           [0., 0.]], dtype=float32)
    """
    abs_max = np.abs(arr).max()
    if abs_max < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / abs_max).astype(np.float32)


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate and coerce the input image to a 2-D float32 array.

    Parameters
    ----------
    image:
        Candidate input array.

    Returns
    -------
    np.ndarray
        ``image`` coerced to dtype float32, guaranteed 2-D with even
        height and width.

    Raises
    ------
    ValueError
        If ``image`` is not 2-D, or if either spatial dimension is odd
        (required for exact ``[H/2, W/2]`` output under
        ``mode="periodization"``).
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D [H, W], got shape {image.shape}. "
            "Squeeze the channel dimension before calling."
        )
    H, W = image.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(
            f"image height and width must both be even for exact "
            f"[H/2, W/2] wavelet output, got shape ({H}, {W})."
        )
    return image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_wavelet_features(
    image: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    normalise: bool = True,
) -> np.ndarray:
    """Compute a single-level 2-D DWT of an image, returning four subbands.

    Parameters
    ----------
    image:
        2-D float32 array of shape ``[H, W]``, with both ``H`` and ``W``
        even.  Expected range [0, 1] (amplitude image before [-1, 1]
        normalisation), matching the convention used by A3's
        ``compute_structure_tensor`` and A10's
        ``compute_structure_tensor_multiscale``.  Single-channel; do
        NOT pass ``[C, H, W]`` — squeeze first.
    wavelet:
        PyWavelets wavelet name.  Default ``"db2"`` (Daubechies-2).
        Any orthogonal/biorthogonal wavelet supported by
        ``pywt.dwt2`` may be passed for ablation studies (e.g.
        ``"haar"``, ``"db4"``, ``"sym2"``).
    normalise:
        If True (default), each of the four subbands is independently
        normalised to [-1, 1] by its absolute maximum, via
        ``_normalise_subband``.  Keeps values in a consistent range
        regardless of image contrast or wavelet energy scaling.

    Returns
    -------
    np.ndarray
        Shape ``[4, H/2, W/2]``, dtype float32.
        Channel 0: LL  (approximation — low-pass / low-pass)
        Channel 1: LH  (horizontal detail — low-pass rows / high-pass cols)
        Channel 2: HL  (vertical detail — high-pass rows / low-pass cols)
        Channel 3: HH  (diagonal detail — high-pass / high-pass)

    Raises
    ------
    ValueError
        If ``image`` is not 2-D, if either dimension is odd, or if the
        resulting subbands do not have shape exactly ``[H/2, W/2]``
        (defensive check; should not trigger under ``mode="periodization"``
        for any wavelet, but guards against future PyWavelets behaviour
        changes or unsupported wavelet/mode combinations).

    Examples
    --------
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> W = compute_wavelet_features(img)
    >>> W.shape
    (4, 128, 128)
    >>> W.dtype
    dtype('float32')
    >>> bool(np.all(W >= -1.0) and np.all(W <= 1.0))
    True

    >>> # Ablation: compare wavelet bases on the same image
    >>> W_db2 = compute_wavelet_features(img, wavelet="db2")
    >>> W_haar = compute_wavelet_features(img, wavelet="haar")
    >>> W_db2.shape == W_haar.shape
    True
    """
    image = _validate_image(image)
    H, W = image.shape

    # ------------------------------------------------------------------
    # Single-level 2-D DWT.
    # mode="periodization" guarantees exact (H/2, W/2) subband shapes
    # for even H, W, regardless of filter length — see module docstring.
    # ------------------------------------------------------------------
    LL, (LH, HL, HH) = pywt.dwt2(image, wavelet, mode=DWT_MODE)

    expected_shape = (H // 2, W // 2)
    for name, band in zip(SUBBAND_NAMES, (LL, LH, HL, HH)):
        if band.shape != expected_shape:
            raise ValueError(
                f"Subband {name} has shape {band.shape}, expected "
                f"{expected_shape}. This indicates an unsupported "
                f"wavelet/mode combination that does not preserve exact "
                f"H/2, W/2 output; verify 'wavelet' and DWT_MODE."
            )

    LL = LL.astype(np.float32)
    LH = LH.astype(np.float32)
    HL = HL.astype(np.float32)
    HH = HH.astype(np.float32)

    if normalise:
        LL = _normalise_subband(LL)
        LH = _normalise_subband(LH)
        HL = _normalise_subband(HL)
        HH = _normalise_subband(HH)

    # Stack in fixed channel order (LL, LH, HL, HH) → [4, H/2, W/2]
    return np.stack([LL, LH, HL, HH], axis=0)
