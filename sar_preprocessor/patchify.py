"""Split a normalized image into fixed-size square patches.

Runs after normalize() and before export(). Patchify has no knowledge
of dataset type, file formats, or export -- it only knows how to slice
arrays.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

#: A patch paired with its (row, col) top-left offset in the source image.
Patch = Tuple[np.ndarray, Tuple[int, int]]


def _pad_to_multiple(image: np.ndarray, patch_size: int, pad_mode: str) -> np.ndarray:
    """Pad the trailing (H, W) dims of `image` up to a multiple of patch_size.

    Args:
        image: (H, W) or (C, H, W) array.
        patch_size: Target patch size; H and W are padded up to the next
            multiple of this.
        pad_mode: Mode passed to `numpy.pad` (e.g. "reflect", "constant").

    Returns:
        Padded array with the same number of dimensions as `image`.
    """
    *leading, height, width = image.shape
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if pad_h == 0 and pad_w == 0:
        return image

    pad_width = [(0, 0)] * len(leading) + [(0, pad_h), (0, pad_w)]
    return np.pad(image, pad_width, mode=pad_mode)


def patchify(
    image: np.ndarray,
    patch_size: int,
    stride: int | None = None,
    pad: bool = False,
    pad_mode: str = "reflect",
) -> List[Patch]:
    """Split `image` into `patch_size` x `patch_size` patches.

    Args:
        image: (H, W) or (C, H, W) float array (typically the output of
            `normalize.normalize`).
        patch_size: Side length of each square patch, in pixels.
        stride: Step between patch top-left corners. Defaults to
            `patch_size` (non-overlapping patches). A stride smaller
            than `patch_size` produces overlapping patches.
        pad: If True, pad H and W up to the next multiple of
            `patch_size` (using `pad_mode`) so no pixels are dropped.
            If False (default), any remainder smaller than a full patch
            is simply not covered by a patch.
        pad_mode: Padding mode forwarded to `numpy.pad` when `pad=True`.

    Returns:
        List of (patch, (row, col)) tuples, where `patch` has shape
        (patch_size, patch_size) or (C, patch_size, patch_size) matching
        the input's dimensionality, and (row, col) is the top-left pixel
        offset of that patch in the (possibly padded) input image.

    Raises:
        ValueError: If `patch_size` is not positive, `stride` is not
            positive, or `image` doesn't have 2 or 3 dimensions.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"patchify() expects a 2D or 3D array, got shape {image.shape!r}"
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride is None:
        stride = patch_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    working = _pad_to_multiple(image, patch_size, pad_mode) if pad else image
    height, width = working.shape[-2], working.shape[-1]

    if height < patch_size or width < patch_size:
        logger.warning(
            "Image (%dx%d) is smaller than patch_size (%d); no patches produced.",
            height,
            width,
            patch_size,
        )
        return []

    patches: List[Patch] = []
    row = 0
    while row + patch_size <= height:
        col = 0
        while col + patch_size <= width:
            if working.ndim == 2:
                patch = working[row : row + patch_size, col : col + patch_size]
            else:
                patch = working[:, row : row + patch_size, col : col + patch_size]
            patches.append((patch.copy(), (row, col)))
            col += stride
        row += stride

    logger.info(
        "Patchified %dx%d image into %d patches (patch_size=%d, stride=%d, pad=%s)",
        height,
        width,
        len(patches),
        patch_size,
        stride,
        pad,
    )
    return patches
