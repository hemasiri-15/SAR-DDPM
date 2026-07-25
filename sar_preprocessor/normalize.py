"""Image normalization, applied after reading and before patchifying.

Readers deliberately know nothing about normalization; this module is
where raw reader output (arbitrary-range float32, possibly complex for
UAVSAR) gets converted into a normalized float32 array in [0, 1] ready
for patchifying and PNG export.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

NormalizeMethod = Literal["percentile", "minmax", "zscore"]


def _to_real_magnitude(image: np.ndarray) -> np.ndarray:
    """Convert a possibly-complex array to a real-valued magnitude array.

    UAVSAR reads can be complex64; every other reader already produces
    real-valued float32. This is a no-op for real input.

    Args:
        image: Input array, real or complex.

    Returns:
        Real-valued float32 array (magnitude, if input was complex).
    """
    if np.iscomplexobj(image):
        return np.abs(image).astype(np.float32)
    return image.astype(np.float32)


def _normalize_2d(
    band: np.ndarray,
    method: NormalizeMethod,
    low: float,
    high: float,
) -> np.ndarray:
    """Normalize a single 2D band to [0, 1] using the given method.

    Args:
        band: (H, W) real-valued array.
        method: One of "percentile", "minmax", "zscore".
        low: Lower percentile (only used for method="percentile").
        high: Upper percentile (only used for method="percentile").

    Returns:
        (H, W) float32 array with values clipped to [0, 1].
    """
    band = band.astype(np.float32)
    finite_mask = np.isfinite(band)
    if not finite_mask.any():
        logger.warning("Band is entirely non-finite; returning zeros.")
        return np.zeros_like(band, dtype=np.float32)

    finite_values = band[finite_mask]

    if method == "percentile":
        lo_val = np.percentile(finite_values, low)
        hi_val = np.percentile(finite_values, high)
    elif method == "minmax":
        lo_val = finite_values.min()
        hi_val = finite_values.max()
    elif method == "zscore":
        mean = finite_values.mean()
        std = finite_values.std()
        std = std if std > 1e-12 else 1.0
        z = (band - mean) / std
        # Map z-scores in [-3, 3] to [0, 1]; clip outliers beyond that.
        normalized = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
        normalized[~finite_mask] = 0.0
        return normalized.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown normalize method {method!r}; expected one of "
            "'percentile', 'minmax', 'zscore'."
        )

    if hi_val <= lo_val:
        logger.warning(
            "Normalization range is degenerate (low=%s, high=%s); "
            "returning zeros.",
            lo_val,
            hi_val,
        )
        return np.zeros_like(band, dtype=np.float32)

    normalized = (band - lo_val) / (hi_val - lo_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32)


def normalize(
    image: np.ndarray,
    method: NormalizeMethod = "percentile",
    low: float = 1.0,
    high: float = 99.0,
    per_channel: bool = True,
) -> np.ndarray:
    """Normalize a SARScene image to float32 in [0, 1].

    Args:
        image: (H, W) or (C, H, W) array, real or complex. Complex input
            (e.g. from the UAVSAR reader) is converted to magnitude
            first.
        method: Normalization strategy:
            - "percentile": clip/scale between the `low`/`high`
              percentiles (robust to speckle outliers; default).
            - "minmax": scale between the true min and max.
            - "zscore": standardize then map [-3, 3] std devs to
              [0, 1].
        low: Lower percentile for method="percentile". Ignored
            otherwise.
        high: Upper percentile for method="percentile". Ignored
            otherwise.
        per_channel: If True and image is (C, H, W), normalize each
            channel independently. If False, a single global
            statistic is computed across all channels together.

    Returns:
        float32 array of the same shape as `image`, with values in
        [0, 1]. Non-finite input values are mapped to 0.0.

    Raises:
        ValueError: If `method` is not recognized, or `image` doesn't
            have 2 or 3 dimensions.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"normalize() expects a 2D or 3D array, got shape {image.shape!r}"
        )

    real_image = _to_real_magnitude(image)

    if real_image.ndim == 2:
        return _normalize_2d(real_image, method, low, high)

    if per_channel:
        return np.stack(
            [_normalize_2d(band, method, low, high) for band in real_image],
            axis=0,
        )

    # Global statistics across all channels, applied per-channel so the
    # relative brightness between channels is preserved.
    flat = real_image.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        logger.warning("Image is entirely non-finite; returning zeros.")
        return np.zeros_like(real_image, dtype=np.float32)

    if method == "percentile":
        lo_val = np.percentile(finite, low)
        hi_val = np.percentile(finite, high)
    elif method == "minmax":
        lo_val = finite.min()
        hi_val = finite.max()
    elif method == "zscore":
        mean = finite.mean()
        std = finite.std()
        std = std if std > 1e-12 else 1.0
        z = (real_image - mean) / std
        normalized = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
        normalized[~np.isfinite(real_image)] = 0.0
        return normalized.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown normalize method {method!r}; expected one of "
            "'percentile', 'minmax', 'zscore'."
        )

    if hi_val <= lo_val:
        logger.warning(
            "Global normalization range is degenerate (low=%s, high=%s); "
            "returning zeros.",
            lo_val,
            hi_val,
        )
        return np.zeros_like(real_image, dtype=np.float32)

    normalized = np.clip((real_image - lo_val) / (hi_val - lo_val), 0.0, 1.0)
    normalized[~np.isfinite(real_image)] = 0.0
    return normalized.astype(np.float32)
