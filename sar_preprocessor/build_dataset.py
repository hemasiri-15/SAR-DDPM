"""Top-level orchestration: detect dataset type, then run the pipeline.

    detect dataset -> reader.read() -> normalize() -> patchify() -> export()

Readers only know how to detect and read; everything else (which
readers exist, what order to try them in, and how to wire the pipeline
stages together) lives here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional

from .export import export_patches
from .normalize import NormalizeMethod, normalize
from .patchify import patchify
from .readers import airsar, dsifn, sentinel1, uavsar
from .utils import get_logger

logger = get_logger(__name__)

#: Readers to try, in order. Each entry is (name, module). AIRSAR is
#: last, but it is NOT a blind fallback -- its own detect() must
#: positively match, same as every other reader.
READERS: List[tuple] = [
    ("sentinel1", sentinel1),
    ("dsifn", dsifn),
    ("uavsar", uavsar),
    ("airsar", airsar),
]

#: One-line description of what each reader's detect() looks for, used
#: only to build a helpful "unknown dataset" message.
_READER_EXPECTATIONS = {
    "sentinel1": "a manifest.safe file (directly, or inside a *.SAFE subfolder)",
    "dsifn": "train/, val/, and test/ subdirectories all present",
    "uavsar": "a *.ann file with a same-basename *.grd file in the same folder",
    "airsar": "a filename containing 'airsar' or a .airsar extension",
}


def detect_dataset(path: Path) -> Optional[ModuleType]:
    """Try each reader's detect() in order and return the first match.

    Args:
        path: Dataset root directory to classify.

    Returns:
        The matching reader module, or None if no reader recognizes the
        directory.
    """
    for name, module in READERS:
        try:
            if module.detect(path):
                logger.info("Detected dataset type '%s' at %s", name, path)
                return module
        except Exception:
            logger.exception("detect() raised an exception for reader '%s'", name)
    return None


def _log_unknown_dataset(path: Path) -> None:
    """Log a friendly, non-crashing message when no reader matches.

    Args:
        path: The dataset root that failed to match any reader.
    """
    lines = [f"Unknown dataset at {path}.", "Checks tried:"]
    for name, _module in READERS:
        expectation = _READER_EXPECTATIONS.get(name, "(no description)")
        lines.append(f"  - {name}: expects {expectation}")
    logger.warning("\n".join(lines))


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    stride: Optional[int] = None,
    pad: bool = False,
    normalize_method: NormalizeMethod = "percentile",
    low: float = 1.0,
    high: float = 99.0,
    prefix: Optional[str] = None,
) -> List[Path]:
    """Run the full detect -> read -> normalize -> patchify -> export pipeline.

    Args:
        input_dir: Root directory of the dataset to process.
        output_dir: Directory PNG patches are written into.
        patch_size: Side length of each square patch, in pixels.
        stride: Patch stride; defaults to `patch_size` (non-overlapping).
        pad: Whether to pad the image up to a multiple of `patch_size`
            before patchifying, so no pixels are dropped.
        normalize_method: One of "percentile", "minmax", "zscore".
        low: Lower percentile bound (only used when
            normalize_method="percentile").
        high: Upper percentile bound (only used when
            normalize_method="percentile").
        prefix: Filename prefix for exported PNGs. Defaults to the
            detected dataset name plus the input directory's name.

    Returns:
        List of paths written by the export stage. Empty if no reader
        matched `input_dir`.
    """
    reader = detect_dataset(input_dir)
    if reader is None:
        _log_unknown_dataset(input_dir)
        return []

    scene = reader.read(input_dir)
    logger.info(
        "Scene ready: dataset=%s shape=%s dtype=%s",
        scene.metadata.get("dataset"),
        scene.image.shape,
        scene.image.dtype,
    )

    normalized = normalize(scene.image, method=normalize_method, low=low, high=high)
    patches = patchify(normalized, patch_size=patch_size, stride=stride, pad=pad)

    if not patches:
        logger.warning(
            "No patches produced for %s (image shape %s, patch_size %d)",
            input_dir,
            normalized.shape,
            patch_size,
        )
        return []

    file_prefix = prefix or f"{scene.metadata.get('dataset', 'scene')}_{input_dir.name}"
    return export_patches(patches, output_dir, file_prefix, scene.metadata)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        A configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="sar_preprocessor",
        description=(
            "Detect a SAR dataset's format, read it, normalize it, split it "
            "into patches, and export the patches as PNGs."
        ),
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Dataset root directory."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Directory to write PNG patches into."
    )
    parser.add_argument(
        "--patch-size", type=int, default=256, help="Square patch side length in pixels."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Patch stride in pixels. Defaults to --patch-size (non-overlapping).",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad the image up to a multiple of --patch-size instead of dropping the remainder.",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["percentile", "minmax", "zscore"],
        default="percentile",
        help="Normalization strategy applied before patchifying.",
    )
    parser.add_argument(
        "--low", type=float, default=1.0, help="Lower percentile for --normalize-method=percentile."
    )
    parser.add_argument(
        "--high", type=float, default=99.0, help="Upper percentile for --normalize-method=percentile."
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Filename prefix for exported PNGs."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to `sys.argv[1:]` when None).

    Returns:
        Process exit code: 0 on success (including "no patches
        produced"), 1 if the input directory doesn't exist.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        logger.error("Input directory does not exist: %s", args.input)
        return 1

    written = process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        patch_size=args.patch_size,
        stride=args.stride,
        pad=args.pad,
        normalize_method=args.normalize_method,
        low=args.low,
        high=args.high,
        prefix=args.prefix,
    )
    logger.info("Done. %d file(s) written.", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
