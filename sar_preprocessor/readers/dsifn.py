"""Reader for DSIFN-formatted datasets.

Detection: DSIFN's one distinguishing filesystem trait, per the project
spec, is the presence of `train/`, `val/`, and `test/` as subdirectories
of the dataset root. This is a pure adapter-level check -- no format
parsing.

Reading -- ASSUMPTION (documented, since there is no sample directory to
validate against and this could not be resolved from the spec alone):
DSIFN is organized as many per-sample images across the train/val/test
splits, but the reader contract in this pipeline returns exactly one
SARScene per call. Rather than block implementation on that mismatch,
this reader treats `path` as pointing at the dataset root and returns
the first image found (path-sorted) by searching `train/`, then falling
back to `val/`, then `test/`. If DSIFN should instead be iterated
sample-by-sample, that iteration belongs in build_dataset.py (per the
"reader returns one scene, orchestration iterates" contract) and can be
added there without changing this reader's interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from PIL import Image

from ..scene import SARScene, make_metadata
from ..utils import get_logger

logger = get_logger(__name__)

#: Subdirectories whose simultaneous presence identifies a DSIFN-style
#: dataset root.
REQUIRED_SPLIT_DIRS = ("train", "val", "test")

#: Image extensions considered when searching a split directory for a
#: representative sample.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def detect(path: Path) -> bool:
    """Return True if `path` contains train/, val/, and test/ subdirs.

    Args:
        path: Candidate dataset root directory.

    Returns:
        True if all three required split directories exist under path.
    """
    if not path.is_dir():
        return False
    return all((path / split).is_dir() for split in REQUIRED_SPLIT_DIRS)


def _find_first_image(split_dir: Path) -> Optional[Path]:
    """Recursively find the first image file (path-sorted) under a split dir.

    Args:
        split_dir: A train/val/test directory.

    Returns:
        Path to the first matching image file, or None if none found.
    """
    if not split_dir.is_dir():
        return None
    candidates = [
        p
        for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(candidates)[0] if candidates else None


def _load_image(image_path: Path):
    """Load a single image file as a float32 array, plus crs/transform.

    Tries rasterio first (handles georeferenced GeoTIFFs), falling back
    to Pillow for plain PNG/JPEG which have no CRS/transform.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (image_array, crs, transform). image_array has shape
        (H, W) for single-band or (num_bands, H, W) for multi-band.
    """
    try:
        with rasterio.open(image_path) as src:
            data = src.read().astype(np.float32)  # (bands, H, W)
            crs = src.crs
            transform = src.transform
        if data.shape[0] == 1:
            data = data[0]
        return data, crs, transform
    except rasterio.errors.RasterioIOError:
        with Image.open(image_path) as img:
            array = np.asarray(img).astype(np.float32)
        if array.ndim == 3:
            # (H, W, C) -> (C, H, W) to match the rest of the pipeline's
            # band-first convention.
            array = np.moveaxis(array, -1, 0)
        return array, None, None


def read(path: Path) -> SARScene:
    """Read a representative image from a DSIFN-style dataset root.

    See module docstring for the documented assumption governing which
    single image is chosen.

    Args:
        path: Dataset root directory (must contain train/val/test).

    Returns:
        A SARScene wrapping the first image found in train/ (falling
        back to val/, then test/).

    Raises:
        FileNotFoundError: If no image file can be found in any split.
    """
    image_path = None
    used_split = None
    for split in REQUIRED_SPLIT_DIRS:
        image_path = _find_first_image(path / split)
        if image_path is not None:
            used_split = split
            break

    if image_path is None:
        raise FileNotFoundError(
            f"No image files ({', '.join(IMAGE_EXTENSIONS)}) found in any of "
            f"{[str(path / s) for s in REQUIRED_SPLIT_DIRS]}"
        )

    image, crs, transform = _load_image(image_path)

    logger.info(
        "Read DSIFN scene from %s (split=%s), shape=%s",
        image_path,
        used_split,
        image.shape,
    )

    metadata = make_metadata(
        dataset="dsifn",
        sensor=None,
        polarization=None,
        look_number=None,
        crs=crs,
        transform=transform,
        split=used_split,
        source_file=str(image_path),
    )
    return SARScene(image=image, metadata=metadata)
