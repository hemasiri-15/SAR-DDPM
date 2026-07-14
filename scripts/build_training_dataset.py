#!/usr/bin/env python3
"""Build a unified, reproducible train/validation split for SAR-DDPM.

The SAR-DDPM training pipeline (see ``scripts/datasets.py::SynthSARDataset``)
consumes ordinary *clean* optical images and synthesizes Gamma-distributed
speckle on the fly. This script does not touch pixel data at all: it only
walks ``Training_Data/``, verifies that every discovered image is readable,
deduplicates paths, and writes two plain-text manifests

    Training_Data/train.txt
    Training_Data/val.txt

each containing one image path per line, *relative* to ``--root``. Relative
paths keep the manifests portable across machines (laptop, HPC cluster,
Docker container, ...): the dataset loader simply joins the manifest root
with each line.

Usage
-----
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --root Training_Data --seed 123 --train-ratio 0.9

Exit status is non-zero if no valid images are found.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HIDDEN_DIR_PREFIXES = (".", "__")

logger = logging.getLogger("build_training_dataset")


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging format/level.

    Args:
        verbose: If True, emit DEBUG-level messages; otherwise INFO-level.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _is_hidden_dir(dirname: str) -> bool:
    """Return True if a directory name should be skipped (e.g. .git, __pycache__)."""
    return dirname.startswith(HIDDEN_DIR_PREFIXES)


def find_images(root: Path) -> Dict[str, List[Path]]:
    """Recursively discover candidate image files under ``root``.

    Images are grouped by their top-level sub-directory of ``root`` (e.g.
    "AID", "ImageNet", "BSD500", ...) purely for reporting purposes; the
    grouping has no effect on the resulting train/val split. Any image
    found directly inside ``root`` (not under a sub-directory) is grouped
    under the key ``"(root)"``. Hidden/system directories such as ``.git``
    and ``__pycache__`` are skipped entirely.

    Args:
        root: Path to the ``Training_Data`` directory.

    Returns:
        A mapping from top-level group name to the list of absolute image
        paths discovered within that group. Order within each list follows
        ``os.walk`` traversal order (not yet shuffled or deduplicated).

    Raises:
        FileNotFoundError: If ``root`` does not exist or is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"Training data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Training data root is not a directory: {root}")

    root = root.resolve()
    groups: Dict[str, List[Path]] = {}

    top_level_entries = sorted(p for p in root.iterdir() if not _is_hidden_dir(p.name))

    for entry in top_level_entries:
        if entry.is_file():
            if entry.suffix.lower() in IMAGE_EXTENSIONS:
                groups.setdefault("(root)", []).append(entry)
            continue

        if not entry.is_dir():
            continue

        group_name = entry.name
        found: List[Path] = []
        for path in entry.rglob("*"):
            # Skip anything living inside a hidden/system directory.
            if any(_is_hidden_dir(part) for part in path.relative_to(entry).parts[:-1]):
                continue
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                found.append(path)

        groups[group_name] = sorted(found)
        logger.info("Scanned %-15s: %d candidate image(s)", group_name, len(found))

    return groups


def _is_readable_image(path: Path) -> bool:
    """Check whether a file is a valid, openable image.

    Uses a two-pass Pillow check: ``Image.verify()`` catches structurally
    broken files cheaply, and a second ``Image.open().load()`` catches
    truncated-data errors that ``verify()`` alone can miss (verify() closes
    the file handle and cannot be reused to load pixel data).

    Args:
        path: Path to the candidate image file.

    Returns:
        True if the image can be opened and decoded successfully.
    """
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception as exc:  # noqa: BLE001 - any decode failure means "corrupted"
        logger.debug("Corrupted/unreadable image %s: %s", path, exc)
        return False


def verify_images(
    groups: Dict[str, List[Path]]
) -> Tuple[List[Path], int, int, Dict[str, int]]:
    """Verify every discovered image and remove duplicate/corrupted entries.

    Args:
        groups: Mapping of group name to candidate image paths, as returned
            by :func:`find_images`.

    Returns:
        A 4-tuple:
            - valid_images: deduplicated list of verified, readable image paths.
            - num_corrupted: count of images that failed the Pillow check.
            - num_duplicates: count of duplicate paths removed.
            - per_group_counts: mapping of group name to number of valid
              images retained from that group (for the summary report).
    """
    valid_images: List[Path] = []
    seen: set = set()
    num_corrupted = 0
    num_duplicates = 0
    per_group_counts: Dict[str, int] = {}

    all_candidates = [(group, p) for group, paths in groups.items() for p in paths]

    for group, path in tqdm(all_candidates, desc="Verifying images", unit="img"):
        resolved = path.resolve()

        if resolved in seen:
            num_duplicates += 1
            continue
        seen.add(resolved)

        if not _is_readable_image(resolved):
            num_corrupted += 1
            continue

        valid_images.append(resolved)
        per_group_counts[group] = per_group_counts.get(group, 0) + 1

    return valid_images, num_corrupted, num_duplicates, per_group_counts


def split_dataset(
    images: List[Path], seed: int, train_ratio: float
) -> Tuple[List[Path], List[Path]]:
    """Deterministically shuffle and split images into train/validation sets.

    Args:
        images: Deduplicated, verified list of image paths.
        seed: Random seed controlling the shuffle (and thus the split).
        train_ratio: Fraction of images assigned to the training set,
            e.g. 0.9 for a 90/10 train/val split.

    Returns:
        A tuple ``(train_images, val_images)``.

    Raises:
        ValueError: If ``train_ratio`` is not in (0, 1) or ``images`` is empty.
    """
    if not images:
        raise ValueError("Cannot split an empty image list.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train-ratio must be in (0, 1), got {train_ratio}")

    shuffled = sorted(images)  # sort first so the shuffle itself is the only
    # source of randomness (filesystem iteration order is not guaranteed
    # stable across platforms, which would otherwise break reproducibility).
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_idx = round(len(shuffled) * train_ratio)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]
    return train_images, val_images


def write_lists(
    train_images: List[Path], val_images: List[Path], root: Path, output_dir: Path
) -> None:
    """Write train.txt and val.txt, one image path per line, relative to root.

    Args:
        train_images: Absolute paths of images assigned to the training set.
        val_images: Absolute paths of images assigned to the validation set.
        root: The resolved Training_Data root; paths are written relative
            to this directory so the manifests remain portable.
        output_dir: Directory in which to write train.txt / val.txt
            (normally the same as ``root``).

    Raises:
        OSError: If the manifest files cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    try:
        with train_path.open("w", encoding="utf-8") as f:
            for img in train_images:
                f.write(f"{img.relative_to(root).as_posix()}\n")

        with val_path.open("w", encoding="utf-8") as f:
            for img in val_images:
                f.write(f"{img.relative_to(root).as_posix()}\n")
    except OSError as exc:
        raise OSError(f"Failed to write dataset manifests to {output_dir}: {exc}") from exc

    logger.info("Wrote %s (%d lines)", train_path, len(train_images))
    logger.info("Wrote %s (%d lines)", val_path, len(val_images))


def print_summary(
    per_group_counts: Dict[str, int],
    total_candidates: int,
    train_images: List[Path],
    val_images: List[Path],
    num_corrupted: int,
    num_duplicates: int,
    seed: int,
) -> None:
    """Print a human-readable summary of the dataset build.

    Args:
        per_group_counts: Valid image count per top-level dataset group.
        total_candidates: Total number of candidate files discovered
            (before dedup/verification).
        train_images: Final list of training image paths.
        val_images: Final list of validation image paths.
        num_corrupted: Number of images that failed verification.
        num_duplicates: Number of duplicate paths removed.
        seed: Random seed used for the split.
    """
    total_valid = len(train_images) + len(val_images)
    name_width = max((len(name) for name in per_group_counts), default=8)
    name_width = max(name_width, 8)

    lines = [
        "=" * 43,
        "SAR-DDPM Training Dataset Builder",
        "=" * 43,
    ]
    for group_name in sorted(per_group_counts):
        count = per_group_counts[group_name]
        lines.append(f"{group_name:<{name_width}} : {count} images")
    lines.append("-" * 43)
    lines.append(f"{'Total images':<{name_width}} : {total_valid}")
    lines.append(f"{'Train images':<{name_width}} : {len(train_images)}")
    lines.append(f"{'Validation':<{name_width}} : {len(val_images)}")
    lines.append(f"{'Corrupted':<{name_width}} : {num_corrupted}")
    lines.append(f"{'Duplicate paths':<{name_width}} : {num_duplicates}")
    lines.append(f"{'Random seed':<{name_width}} : {seed}")
    lines.append("=" * 43)

    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace with ``root``, ``seed``, and
        ``train_ratio`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Build reproducible train/val manifests for SAR-DDPM training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Training_Data"),
        help="Path to the Training_Data directory to scan.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic shuffling/splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of images assigned to the training split.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: scan, verify, split, and write the dataset manifests.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()
    setup_logging(args.verbose)

    try:
        root = args.root.resolve()
        logger.info("Scanning %s ...", root)
        groups = find_images(root)

        total_candidates = sum(len(paths) for paths in groups.values())
        if total_candidates == 0:
            logger.error("No candidate images found under %s", root)
            return 1

        valid_images, num_corrupted, num_duplicates, per_group_counts = verify_images(groups)
        if not valid_images:
            logger.error("No valid images survived verification.")
            return 1

        train_images, val_images = split_dataset(valid_images, args.seed, args.train_ratio)
        write_lists(train_images, val_images, root, root)

        print_summary(
            per_group_counts=per_group_counts,
            total_candidates=total_candidates,
            train_images=train_images,
            val_images=val_images,
            num_corrupted=num_corrupted,
            num_duplicates=num_duplicates,
            seed=args.seed,
        )
        return 0

    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        logger.error("Dataset build failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors clearly
        logger.exception("Unexpected error during dataset build: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
