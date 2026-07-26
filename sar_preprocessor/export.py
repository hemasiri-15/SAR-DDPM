"""Export normalized, patchified patches to PNG files on disk.

Per project requirement: this pipeline writes PNGs only. Scene metadata
rides along through the pipeline (for logging / future use) but nothing
beyond patches touches disk here -- no sidecar metadata/JSON files are
written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .patchify import Patch
from .utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _patch_to_uint8_images(patch: np.ndarray) -> List[np.ndarray]:
    """Convert a normalized ([0, 1]) patch into one or more uint8 images.

    Args:
        patch: (patch_size, patch_size) or (C, patch_size, patch_size)
            float array with values in [0, 1].

    Returns:
        A list of (patch_size, patch_size) uint8 arrays -- one per band.
        A single-band or 3-band (RGB-able) patch still yields one entry
        per band; callers decide whether to merge 3 bands into an RGB
        PNG or write them separately.
    """
    clipped = np.clip(patch, 0.0, 1.0)
    scaled = np.round(clipped * 255.0).astype(np.uint8)
    if scaled.ndim == 2:
        return [scaled]
    return [scaled[band_idx] for band_idx in range(scaled.shape[0])]


def export_patches(
    patches: List[Patch],
    output_dir: Path,
    prefix: str,
    metadata: Dict[str, Any],
) -> List[Path]:
    """Write each patch to disk as one or more PNG files.

    Naming convention: `{prefix}_r{row}_c{col}.png` for single-band
    patches, or `{prefix}_r{row}_c{col}_b{band}.png` per band for
    multi-band patches -- except when a patch has exactly 3 bands, in
    which case it is written as one RGB PNG (since the common
    multi-polarization/multi-band case worth composing visually is
    3-channel).

    Args:
        patches: List of (patch, (row, col)) tuples, as produced by
            `patchify.patchify`. Patch values are expected in [0, 1]
            (i.e. already normalized).
        output_dir: Directory to write PNGs into; created if it doesn't
            exist.
        prefix: Filename prefix, typically derived from the source
            scene (e.g. dataset name + scene id).
        metadata: The scene's metadata dict. Not written to disk; used
            only for logging context here.

    Returns:
        List of paths written, in the same order as `patches`.
    """
    ensure_dir(output_dir)
    written: List[Path] = []

    for patch, (row, col) in patches:

        # Skip empty or invalid patches
        if (
            patch.size == 0
            or patch.shape[-2] == 0
            or patch.shape[-1] == 0
        ):
            logger.warning(
                "Skipping empty patch at (%d, %d) with shape %s",
                row,
                col,
                patch.shape,
            )
            continue

        bands = _patch_to_uint8_images(patch)

        if len(bands) == 3:
            rgb = np.stack(bands, axis=-1)
            out_path = output_dir / f"{prefix}_r{row}_c{col}.png"
            Image.fromarray(rgb, mode="RGB").save(out_path)
            written.append(out_path)
        elif len(bands) == 1:
            out_path = output_dir / f"{prefix}_r{row}_c{col}.png"
            Image.fromarray(bands[0], mode="L").save(out_path)
            written.append(out_path)
        else:
            for band_idx, band in enumerate(bands):
                out_path = output_dir / f"{prefix}_r{row}_c{col}_b{band_idx}.png"
                Image.fromarray(band, mode="L").save(out_path)
                written.append(out_path)

    logger.info(
        "Exported %d patch(es) (%d PNG file(s)) to %s [dataset=%s]",
        len(patches),
        len(written),
        output_dir,
        metadata.get("dataset"),
    )
    return written
