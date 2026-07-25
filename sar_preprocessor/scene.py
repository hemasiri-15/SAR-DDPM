"""Core data types shared across the sar_preprocessor package.

This module defines the single data structure that flows out of every
reader (`SARScene`) and the common metadata schema that all readers are
required to populate, even when a given dataset does not provide a value
for a particular field (in which case it is set to ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

#: Keys that must be present in every SARScene.metadata dict. Readers may
#: add additional dataset-specific keys on top of these, but downstream
#: code (normalize/patchify/export) is only guaranteed these will exist.
COMMON_METADATA_KEYS = (
    "dataset",
    "sensor",
    "polarization",
    "look_number",
    "crs",
    "transform",
)


def make_metadata(
    dataset: str,
    sensor: Optional[str] = None,
    polarization: Optional[Any] = None,
    look_number: Optional[int] = None,
    crs: Optional[Any] = None,
    transform: Optional[Any] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a metadata dict that satisfies the common schema.

    Args:
        dataset: Short identifier for the dataset type, e.g. "sentinel1".
        sensor: Sensor name/id if known (e.g. "S1A"), else None.
        polarization: A single polarization string (e.g. "VV") or a list
            of polarizations for multi-band scenes. None if unknown.
        look_number: Integer look number if applicable, else None.
        crs: Coordinate reference system object/string if georeferenced,
            else None.
        transform: Affine transform if georeferenced, else None.
        **extra: Any additional dataset-specific metadata fields. These
            are merged in on top of the common keys.

    Returns:
        A dict guaranteed to contain every key in COMMON_METADATA_KEYS,
        plus any extra fields passed in.
    """
    metadata: Dict[str, Any] = {
        "dataset": dataset,
        "sensor": sensor,
        "polarization": polarization,
        "look_number": look_number,
        "crs": crs,
        "transform": transform,
    }
    metadata.update(extra)
    return metadata


@dataclass
class SARScene:
    """A single SAR acquisition/scene as produced by a reader.

    Attributes:
        image: The raster data. Shape is either (H, W) for a single band
            or (num_bands, H, W) for multiple bands (e.g. multiple
            polarizations). Dtype is typically float32.
        metadata: Dict satisfying COMMON_METADATA_KEYS (see
            `make_metadata`). Readers populate this; nothing downstream
            should assume keys beyond the common schema exist without
            checking.
    """

    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError(
                f"SARScene.image must be a numpy.ndarray, got {type(self.image)!r}"
            )
        if self.image.ndim not in (2, 3):
            raise ValueError(
                "SARScene.image must have shape (H, W) or (num_bands, H, W), "
                f"got shape {self.image.shape!r}"
            )
        missing = [k for k in COMMON_METADATA_KEYS if k not in self.metadata]
        if missing:
            raise ValueError(
                f"SARScene.metadata is missing required keys: {missing}. "
                "Use types.make_metadata() to build a compliant dict."
            )
