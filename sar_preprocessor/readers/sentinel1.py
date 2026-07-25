"""Reader for Sentinel-1 SAFE products.

Detection: looks for a `manifest.safe` file either directly inside the
given path or inside an immediate `*.SAFE` subfolder, within a bounded
search depth (SAFE products are never nested deeper than that in
practice).

Reading: loads every `measurement/*.tif(f)` band found under the SAFE
root. This is the raw amplitude/DN data -- no radiometric calibration is
applied (that is out of scope for this tool). Known trade-off: bands are
loaded fully into memory rather than windowed, which is fine for typical
scenes but should be revisited if a real product is large enough to blow
out RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from ..scene import SARScene, make_metadata
from ..utils import get_logger

logger = get_logger(__name__)

#: SAFE products are shallow; we never need to look further than this
#: many levels below the input path to find manifest.safe.
MAX_DETECTION_DEPTH = 2


def _find_safe_root(path: Path) -> Optional[Path]:
    """Locate the SAFE product root under `path`, if any.

    A SAFE root is any directory containing a `manifest.safe` file. This
    checks `path` itself first, then any `*.SAFE` directories up to
    MAX_DETECTION_DEPTH levels down.

    Args:
        path: Candidate dataset root.

    Returns:
        Path to the directory containing manifest.safe, or None.
    """
    if (path / "manifest.safe").is_file():
        return path

    if not path.is_dir():
        return None

    stack: List[tuple] = [(path, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_DETECTION_DEPTH:
            continue
        try:
            children = list(current.iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for child in children:
            if not child.is_dir():
                continue
            if (child / "manifest.safe").is_file():
                return child
            if depth < MAX_DETECTION_DEPTH:
                stack.append((child, depth + 1))
    return None


def detect(path: Path) -> bool:
    """Return True if `path` looks like a Sentinel-1 SAFE product.

    Args:
        path: Candidate dataset root directory.

    Returns:
        True if a manifest.safe file is found under `path` within
        MAX_DETECTION_DEPTH levels.
    """
    return _find_safe_root(path) is not None


def _guess_sensor(safe_root: Path) -> Optional[str]:
    """Guess the sensor id (S1A/S1B/S1C/...) from the SAFE folder name.

    This is a cheap filename-prefix heuristic, not manifest XML parsing.

    Args:
        safe_root: Path to the SAFE product root.

    Returns:
        A short sensor string like "S1A", or None if it can't be guessed.
    """
    name = safe_root.name.upper()
    for prefix in ("S1A", "S1B", "S1C", "S1D"):
        if name.startswith(prefix):
            return prefix
    return None


def read(path: Path) -> SARScene:
    """Read a Sentinel-1 SAFE product into a SARScene.

    Args:
        path: Dataset root passed to `detect()`; must contain (or be) a
            valid SAFE product.

    Returns:
        A SARScene whose image is:
          - shape (H, W) if exactly one measurement band is found, or
          - shape (num_pols, H, W) if multiple bands are found, stacked
            in filename-sorted order.
        DN values are cast to float32; no calibration is applied.

    Raises:
        FileNotFoundError: If no SAFE product / measurement bands can be
            located under `path`.
    """
    safe_root = _find_safe_root(path)
    if safe_root is None:
        raise FileNotFoundError(
            f"No Sentinel-1 SAFE product (manifest.safe) found under {path}"
        )

    measurement_dir = safe_root / "measurement"
    if not measurement_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a 'measurement' directory under SAFE root {safe_root}, "
            "found none."
        )

    band_paths = sorted(
        [
            p
            for p in measurement_dir.iterdir()
            if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
        ]
    )
    if not band_paths:
        raise FileNotFoundError(
            f"No .tif/.tiff measurement bands found in {measurement_dir}"
        )

    bands = []
    polarizations: List[str] = []
    crs = None
    transform = None
    for band_path in band_paths:
        with rasterio.open(band_path) as src:
            data = src.read(1).astype(np.float32)
            if crs is None:
                crs = src.crs
                transform = src.transform
        bands.append(data)
        pol = _guess_polarization(band_path.stem)
        polarizations.append(pol if pol is not None else band_path.stem)

    if len(bands) == 1:
        image = bands[0]
        pol_field = polarizations[0]
    else:
        image = np.stack(bands, axis=0)
        pol_field = polarizations

    logger.info(
        "Read Sentinel-1 scene from %s: %d band(s), shape=%s",
        safe_root,
        len(bands),
        image.shape,
    )

    metadata = make_metadata(
        dataset="sentinel1",
        sensor=_guess_sensor(safe_root),
        polarization=pol_field,
        look_number=None,
        crs=crs,
        transform=transform,
        safe_root=str(safe_root),
    )
    return SARScene(image=image, metadata=metadata)


def _guess_polarization(filename_stem: str) -> Optional[str]:
    """Extract a polarization token (VV/VH/HH/HV) from a filename stem.

    Args:
        filename_stem: Filename without extension, e.g.
            "s1a-iw-grd-vv-...".

    Returns:
        One of "VV", "VH", "HH", "HV" if found (case-insensitive), else
        None.
    """
    lowered = filename_stem.lower()
    for pol in ("vv", "vh", "hh", "hv"):
        if pol in lowered:
            return pol.upper()
    return None
