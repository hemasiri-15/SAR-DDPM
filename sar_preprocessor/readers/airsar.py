"""Reader for legacy AIRSAR products.

This is the weakest-grounded reader in the package: there is no
standard raster format for AIRSAR distributions (old JPL/NASA releases
use custom binary layouts such as CEOS leader/image/trailer or
compressed Stokes-matrix formats, while more recent AIRSAR-derived
products are often just GeoTIFF or ENVI). Fully reverse-engineering the
legacy binary layouts is out of scope here, so the pragmatic strategy
is:

1. Detection requires a positive match -- a filename containing
   "airsar" (case-insensitive) or a `.airsar` extension -- so an
   unrelated directory is never silently classified as AIRSAR just
   because nothing else matched. Detection is NOT used as a blind
   fallback by build_dataset.py.
2. Reading tries `rasterio` first on the matched file, which
   transparently covers modern/processed GeoTIFF or ENVI distributions.
3. If rasterio can't open it, falls back to raw binary + a same-basename
   sidecar header (`.hdr` or `.meta`, simple `key: value` or
   `key = value` lines with `width`/`height`/`dtype`), mirroring the
   UAVSAR `.ann`/`.grd` pattern for simple flat-binary exports.
4. Otherwise, fails loudly with a message stating what was tried and
   what's needed, rather than guessing at an unknown legacy layout.

ASSUMPTION, flagged pending a real sample: this covers the two most
plausible cases (already-processed raster, or simple flat binary with a
header) but will not handle raw CEOS/Stokes-matrix legacy layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio

from ..scene import SARScene, make_metadata
from ..utils import get_logger, walk_limited_depth

logger = get_logger(__name__)

#: How deep to search for an airsar-named file, mirroring the bounded
#: search used by the other readers.
MAX_DETECTION_DEPTH = 2

_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint8": np.uint8,
    "complex64": np.complex64,
}


def _find_airsar_file(path: Path) -> Optional[Path]:
    """Find the first file whose name suggests it's an AIRSAR product.

    Args:
        path: Candidate dataset directory (or a direct file path).

    Returns:
        Path to the matched file, or None if nothing matches.
    """
    if path.is_file():
        return path if _looks_like_airsar(path) else None
    if not path.is_dir():
        return None
    for candidate in walk_limited_depth(path, MAX_DETECTION_DEPTH):
        if candidate.is_file() and _looks_like_airsar(candidate):
            return candidate
    return None


def _looks_like_airsar(file_path: Path) -> bool:
    """Check whether a filename positively indicates an AIRSAR product.

    Args:
        file_path: Candidate file.

    Returns:
        True if "airsar" appears in the filename (case-insensitive) or
        the file has a `.airsar` extension.
    """
    name = file_path.name.lower()
    return "airsar" in name or file_path.suffix.lower() == ".airsar"


def detect(path: Path) -> bool:
    """Return True only if a positively-identified AIRSAR file exists.

    This is a real detection test, not a fallback: if nothing in `path`
    is named in a way that suggests AIRSAR, this returns False like any
    other reader would.

    Args:
        path: Candidate dataset directory.

    Returns:
        True if an airsar-named file is found under `path`.
    """
    return _find_airsar_file(path) is not None


def _find_sidecar_header(data_path: Path) -> Optional[Path]:
    """Find a same-basename .hdr or .meta sidecar file next to `data_path`.

    Args:
        data_path: The raw binary data file.

    Returns:
        Path to the sidecar header file, or None if neither exists.
    """
    for ext in (".hdr", ".meta"):
        candidate = data_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _parse_sidecar_header(header_path: Path) -> Dict[str, str]:
    """Parse a simple `key: value` / `key = value` sidecar header file.

    Args:
        header_path: Path to the .hdr/.meta file.

    Returns:
        Dict mapping lowercased key -> stripped value string.
    """
    entries: Dict[str, str] = {}
    for line in header_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
        elif "=" in line:
            key, _, value = line.partition("=")
        else:
            continue
        entries[key.strip().lower()] = value.strip()
    return entries


def _read_raw_binary_with_sidecar(data_path: Path, header_path: Path) -> np.ndarray:
    """Read a flat raw-binary file using width/height/dtype from a sidecar.

    Args:
        data_path: Raw binary data file.
        header_path: Sidecar .hdr/.meta file with width/height/dtype.

    Returns:
        A 2D array of shape (height, width).

    Raises:
        ValueError: If width, height, or dtype can't be determined, or
            if the dtype isn't one of the small set of formats this
            reader supports, or if the file size doesn't match the
            expected width*height*itemsize.
    """
    entries = _parse_sidecar_header(header_path)

    def _get_int(*keys: str) -> Optional[int]:
        for key in keys:
            if key in entries:
                try:
                    return int(float(entries[key]))
                except ValueError:
                    continue
        return None

    width = _get_int("width", "samples", "cols", "columns")
    height = _get_int("height", "lines", "rows")
    dtype_name = entries.get("dtype", entries.get("data type", "float32")).lower()

    if width is None or height is None:
        raise ValueError(
            f"Sidecar header {header_path} does not specify both width and "
            f"height (found width={width}, height={height}). Expected keys "
            "like 'width'/'samples'/'cols' and 'height'/'lines'/'rows'."
        )

    dtype = _DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}' in {header_path}. Supported: "
            f"{sorted(_DTYPE_MAP)}."
        )

    expected_bytes = width * height * np.dtype(dtype).itemsize
    actual_bytes = data_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{data_path} is {actual_bytes} bytes, expected {expected_bytes} "
            f"bytes for width={width}, height={height}, dtype={dtype_name}. "
            "Check the sidecar header values against the actual data file."
        )

    return np.fromfile(data_path, dtype=dtype).reshape(height, width)


def read(path: Path) -> SARScene:
    """Read an AIRSAR product into a SARScene.

    Tries rasterio first, then falls back to raw binary + sidecar
    header. See module docstring for the full strategy and its
    limitations.

    Args:
        path: Dataset directory (or direct file path) matched by
            `detect()`.

    Returns:
        A SARScene for the matched AIRSAR file.

    Raises:
        FileNotFoundError: If no airsar-named file is found.
        ValueError: If the file can't be opened by rasterio and no
            usable sidecar header is present for the raw-binary
            fallback.
    """
    data_path = _find_airsar_file(path)
    if data_path is None:
        raise FileNotFoundError(f"No AIRSAR-identifiable file found under {path}")

    crs = None
    transform = None
    try:
        with rasterio.open(data_path) as src:
            image = src.read().astype(np.float32)  # (bands, H, W)
            crs = src.crs
            transform = src.transform
        if image.shape[0] == 1:
            image = image[0]
        logger.info(
            "Read AIRSAR scene from %s via rasterio: shape=%s",
            data_path,
            image.shape,
        )
    except rasterio.errors.RasterioIOError:
        header_path = _find_sidecar_header(data_path)
        if header_path is None:
            raise ValueError(
                f"{data_path} could not be opened by rasterio (not a "
                "GeoTIFF/ENVI-readable raster) and no sidecar .hdr/.meta "
                "header file was found for a raw-binary fallback. This "
                "AIRSAR reader cannot parse raw legacy CEOS/Stokes-matrix "
                "layouts without a sample to validate against -- add a "
                "sidecar header describing width/height/dtype, or share a "
                "sample so this reader can be extended."
            )
        image = _read_raw_binary_with_sidecar(data_path, header_path).astype(
            np.float32
        )
        logger.info(
            "Read AIRSAR scene from %s via raw-binary fallback (%s): shape=%s",
            data_path,
            header_path.name,
            image.shape,
        )

    metadata = make_metadata(
        dataset="airsar",
        sensor="AIRSAR",
        polarization=_guess_polarization(data_path.stem),
        look_number=None,
        crs=crs,
        transform=transform,
        source_file=str(data_path),
    )
    return SARScene(image=image, metadata=metadata)


def _guess_polarization(filename_stem: str) -> Optional[str]:
    """Extract a polarization token (hh/hv/vh/vv) from a filename stem.

    Args:
        filename_stem: Filename without extension.

    Returns:
        The uppercased polarization token if present, else None.
    """
    lowered = filename_stem.lower()
    for pol in ("hh", "hv", "vh", "vv"):
        if pol in lowered:
            return pol.upper()
    return None
