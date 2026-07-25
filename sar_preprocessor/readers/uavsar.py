"""Reader for UAVSAR .ann/.grd product pairs.

Detection: at least one `*.ann` file with a same-basename `*.grd` file in
the same folder. ASSUMPTION: UAVSAR downloads keep `.ann`/`.grd` pairs
together in one folder (not split across separate folders) -- flagged as
an edge case to revisit if that's wrong for real data.

Reading -- ASSUMPTION (no real UAVSAR sample available to validate
against): `.ann` files are plain-text annotation files with lines of the
form `key (units) = value; comment`. UAVSAR key names vary by product
type and polarization channel, so rather than hardcode one exact key
name, this reader searches the parsed key/value dict for the first key
containing "rows" and the first containing "col" (covers both
"columns" and "cols" spellings) to determine the raster shape. GRD power
products are real-valued float32 by convention, so that is tried first;
if the resulting `rows * cols * 4` byte count doesn't match the actual
`.grd` file size, complex64 (8 bytes/pixel, for SLC/MLC-style complex
products) is tried as a fallback before giving up with a clear error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..scene import SARScene, make_metadata
from ..utils import get_logger

logger = get_logger(__name__)

_KV_LINE_RE = re.compile(r"^\s*([^=;]+?)\s*=\s*([^;]+?)\s*(;.*)?$")


def _find_ann_grd_pair(path: Path) -> Optional[Tuple[Path, Path]]:
    """Find the first .ann file with a matching same-basename .grd file.

    Args:
        path: Candidate dataset directory.

    Returns:
        (ann_path, grd_path) tuple, or None if no matching pair exists.
    """
    if not path.is_dir():
        return None
    for ann_path in sorted(path.glob("*.ann")):
        grd_path = ann_path.with_suffix(".grd")
        if grd_path.is_file():
            return ann_path, grd_path
    return None


def detect(path: Path) -> bool:
    """Return True if `path` contains at least one .ann/.grd pair.

    Args:
        path: Candidate dataset directory.

    Returns:
        True if a matching .ann/.grd pair is found directly in `path`.
    """
    return _find_ann_grd_pair(path) is not None


def _parse_ann_file(ann_path: Path) -> Dict[str, str]:
    """Parse a UAVSAR .ann file into a flat key -> value string dict.

    Lines are expected in the form `key (units) = value; comment`. The
    unit annotation (parentheses) and trailing comment are stripped;
    keys are lowercased for case-insensitive lookup.

    Args:
        ann_path: Path to the .ann file.

    Returns:
        Dict mapping lowercased key -> raw value string.
    """
    entries: Dict[str, str] = {}
    text = ann_path.read_text(errors="replace")
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith(";"):
            continue
        match = _KV_LINE_RE.match(line)
        if not match:
            continue
        raw_key, value, _comment = match.groups()
        # Strip a trailing "(units)" annotation from the key, if present.
        key = re.sub(r"\([^)]*\)", "", raw_key).strip().lower()
        entries[key] = value.strip()
    return entries


def _find_dimension(entries: Dict[str, str], token: str) -> Optional[int]:
    """Find the first integer-valued key containing `token`.

    Args:
        entries: Parsed .ann key/value dict.
        token: Substring to search for in keys, e.g. "rows" or "col".

    Returns:
        The parsed integer value of the first matching key, or None.
    """
    for key, value in entries.items():
        if token in key:
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def read(path: Path) -> SARScene:
    """Read a UAVSAR .ann/.grd pair into a SARScene.

    Args:
        path: Directory containing the .ann/.grd pair.

    Returns:
        A SARScene with a single-band (H, W) float32 or complex64 image,
        depending on which dtype's expected byte count matches the .grd
        file size (see module docstring).

    Raises:
        FileNotFoundError: If no .ann/.grd pair is found.
        ValueError: If row/column counts can't be parsed from the .ann
            file, or if neither float32 nor complex64 matches the .grd
            file size (raster shape can't be determined reliably).
    """
    pair = _find_ann_grd_pair(path)
    if pair is None:
        raise FileNotFoundError(f"No .ann/.grd pair found in {path}")
    ann_path, grd_path = pair

    entries = _parse_ann_file(ann_path)
    rows = _find_dimension(entries, "rows")
    cols = _find_dimension(entries, "col")  # matches "cols" and "columns"

    if rows is None or cols is None:
        raise ValueError(
            f"Could not determine raster dimensions from {ann_path}: "
            f"found rows={rows}, cols={cols}. Expected a key containing "
            "'rows' and one containing 'col'/'columns' with an integer "
            "value, e.g. 'grd_pwr.set_rows (pixels) = 1024'."
        )

    file_size = grd_path.stat().st_size
    pixel_count = rows * cols

    if file_size == pixel_count * 4:
        dtype = np.float32
    elif file_size == pixel_count * 8:
        dtype = np.complex64
    else:
        raise ValueError(
            f"Could not reconcile {grd_path} (size={file_size} bytes) with "
            f"parsed dimensions rows={rows}, cols={cols} from {ann_path}. "
            f"Expected {pixel_count * 4} bytes (float32) or "
            f"{pixel_count * 8} bytes (complex64). Possible fixes: check "
            "that the .ann/.grd pair actually correspond to each other, "
            "or that the parsed row/column keys are the correct ones for "
            "this product type."
        )

    image = np.fromfile(grd_path, dtype=dtype).reshape(rows, cols).astype(
        np.complex64 if dtype == np.complex64 else np.float32
    )

    logger.info(
        "Read UAVSAR scene from %s: shape=%s, dtype=%s",
        grd_path,
        image.shape,
        image.dtype,
    )

    metadata = make_metadata(
        dataset="uavsar",
        sensor="UAVSAR",
        polarization=_guess_polarization(grd_path.stem),
        look_number=None,
        crs=None,
        transform=None,
        ann_file=str(ann_path),
        grd_file=str(grd_path),
    )
    return SARScene(image=image, metadata=metadata)


def _guess_polarization(filename_stem: str) -> Optional[str]:
    """Extract a polarization token from a UAVSAR filename stem.

    UAVSAR product filenames commonly embed a 4-letter polarization code
    (e.g. HHHH, HVHV) among underscore-separated fields.

    Args:
        filename_stem: Filename without extension.

    Returns:
        The polarization token if found, else None.
    """
    for token in re.split(r"[_\-.]", filename_stem):
        if len(token) == 4 and set(token.upper()) <= {"H", "V"}:
            return token.upper()
    return None
