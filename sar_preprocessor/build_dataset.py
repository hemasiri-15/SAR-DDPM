"""Top-level orchestration: detect dataset type, then run the pipeline.

    detect dataset -> reader.read() -> normalize() -> patchify() -> export()

Readers only know how to detect and read *a single scene*; everything
else -- which readers exist, what order to try them in, how to walk a
dataset's directory layout into individual samples, how to parallelize
that work, and how to wire the pipeline stages together -- lives here.

Split-aware datasets (e.g. DSIFN's train/val/test layout) are handled by
`iter_dataset_samples`, the one place that knows how to turn a dataset
root into a stream of `Sample(path, split)` pairs. A reader module opts
into split iteration purely by declaring a `SPLITS` tuple (and,
optionally, its own `iter_split_samples` for a non-flat layout); it
never iterates anything itself, and `reader.read()` keeps its existing
single-scene contract (one path in, one `SARScene` out). Readers with no
`SPLITS` attribute (Sentinel-1, UAVSAR, AIRSAR today) fall back to the
original one-scene-per-dataset behavior automatically.

Samples are processed either serially or, with `--workers > 1`, across a
`ProcessPoolExecutor`. Progress is checkpointed to a small state file
under `output_dir` so a run can be resumed with `--skip-existing`
without redoing already-exported samples. A `metadata.json` manifest
summarizing the run (counts, patch settings, timing) is written to
`output_dir` when the run finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, NamedTuple, Optional

from tqdm import tqdm

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

#: Same mapping, keyed for lookup by name. Used by worker processes,
#: which are handed a reader *name* (a picklable string) rather than the
#: module object itself, then resolve it back to the module locally.
_READERS_BY_NAME: Dict[str, ModuleType] = dict(READERS)

#: One-line description of what each reader's detect() looks for, used
#: only to build a helpful "unknown dataset" message.
_READER_EXPECTATIONS = {
    "sentinel1": "a manifest.safe file (directly, or inside a *.SAFE subfolder)",
    "dsifn": "train/, val/, and test/ subdirectories all present",
    "uavsar": "a *.ann file with a same-basename *.grd file in the same folder",
    "airsar": "a filename containing 'airsar' or a .airsar extension",
}

#: Image extensions considered when a reader declares splits but does
#: not provide its own `iter_split_samples` -- i.e. the generic
#: "list every image file directly under this split directory" fallback.
_DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

#: Name of the resume/checkpoint file written under output_dir.
_STATE_FILENAME = ".sar_preprocessor_state.json"

#: Name of the run summary written under output_dir when a run finishes.
_MANIFEST_FILENAME = "metadata.json"


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


class Sample(NamedTuple):
    """One unit of work for the pipeline.

    Attributes:
        path: The path to hand to `reader.read()`. For split-less
            datasets this is the dataset root itself (today's
            behavior); for split datasets it's an individual sample
            path within that split.
        split: Split name ("train"/"val"/"test", or whatever the
            reader declares), or None for datasets with no split
            structure.
    """

    path: Path
    split: Optional[str]

    def key(self) -> str:
        """Stable string identity used for resume/checkpoint bookkeeping."""
        return f"{self.split or '_'}::{self.path}"


def _default_list_split_samples(split_dir: Path) -> List[Path]:
    """Fallback sample listing: every image file directly under `split_dir`.

    Used when a reader declares `SPLITS` but doesn't provide its own
    `iter_split_samples`. Sorted for deterministic ordering/progress
    bars/checkpoints across runs.

    Args:
        split_dir: A single split subdirectory (e.g. `input_dir/train`).

    Returns:
        Sorted list of image file paths directly inside `split_dir`.
    """
    return sorted(
        p
        for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _DEFAULT_IMAGE_EXTENSIONS
    )


def iter_dataset_samples(reader: ModuleType, input_dir: Path) -> Iterator[Sample]:
    """Enumerate every sample that `reader.read()` should be called on.

    This is the single place that knows how to walk a dataset's
    directory layout into individual read()-able samples, so readers
    stay limited to detect() + read(). Two shapes are supported:

    - Split datasets (currently DSIFN): the reader module exposes a
      `SPLITS` tuple, e.g. `SPLITS = ("train", "val", "test")`. For
      each split subdirectory that exists under `input_dir`, samples
      are listed by the reader's own `iter_split_samples(split_dir)`
      if it defines one (for datasets with a non-flat layout, such as
      paired A/B folders), otherwise by the generic image-file listing
      in `_default_list_split_samples`.
    - Single-scene datasets (Sentinel-1, UAVSAR, AIRSAR today): no
      `SPLITS` attribute is present, so `input_dir` itself is yielded
      as the one sample with split=None -- identical to the original
      one-scene-per-dataset behavior.

    To make a future reader split-aware, give its module a `SPLITS`
    tuple (and, optionally, `iter_split_samples`). Nothing here or in
    `process_dataset` needs to change.

    Args:
        reader: The reader module matched by `detect_dataset`.
        input_dir: Dataset root directory.

    Yields:
        `Sample(path, split)` for each unit of work, in split order
        then sample order.
    """
    splits = getattr(reader, "SPLITS", None)
    if not splits:
        yield Sample(path=input_dir, split=None)
        return

    list_split_samples = getattr(reader, "iter_split_samples", _default_list_split_samples)

    for split in splits:
        split_dir = input_dir / split
        if not split_dir.is_dir():
            logger.warning("Split '%s' not found under %s, skipping", split, input_dir)
            continue

        sample_paths = list(list_split_samples(split_dir))
        if not sample_paths:
            logger.warning("Split '%s' at %s has no samples", split, split_dir)
            continue

        for sample_path in sample_paths:
            yield Sample(path=sample_path, split=split)


class SampleResult(NamedTuple):
    """Outcome of processing one `Sample`.

    Attributes:
        sample: The sample that was processed.
        written: Paths written by the export stage (empty if no
            patches were produced).
        error: Exception message if processing failed, else None. A
            failed sample does not stop the run; it's logged and
            skipped so one bad file doesn't lose an entire split.
    """

    sample: Sample
    written: List[Path]
    error: Optional[str]


def _default_prefix(dataset_name: str, sample: Sample, input_dir: Path) -> str:
    """Build the default export filename prefix for a sample.

    Preserves the exact original naming scheme for split-less datasets
    ("<dataset>_<input_dir name>") and extends it per-sample for split
    datasets ("<dataset>_<sample stem>"), so filenames stay unique
    across a split.

    Args:
        dataset_name: `scene.metadata["dataset"]` for this sample.
        sample: The sample being processed.
        input_dir: Dataset root (used only in the split-less case).

    Returns:
        The filename prefix to hand to `export_patches`.
    """
    if sample.split is None:
        return f"{dataset_name}_{input_dir.name}"
    return f"{dataset_name}_{sample.path.stem}"


def _process_sample(
    reader: ModuleType,
    sample: Sample,
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    stride: Optional[int],
    pad: bool,
    normalize_method: NormalizeMethod,
    low: float,
    high: float,
    prefix: Optional[str],
) -> SampleResult:
    """Run read -> normalize -> patchify -> export for a single sample.

    Never raises: any exception from `reader.read()` or the pipeline
    stages is caught and returned as `SampleResult.error` so a single
    corrupt file doesn't abort a whole dataset run (this matters more
    now that a run may cover thousands of samples).

    Args:
        reader: Reader module to call `.read()` on.
        sample: The sample (path + split) to process.
        input_dir: Dataset root, used only to build the default prefix
            for split-less datasets (preserves the original naming
            scheme exactly).
        output_dir: Base output directory. Split samples are written
            under `output_dir/<split>/`; split-less samples are
            written directly under `output_dir`, same as before.
        patch_size, stride, pad, normalize_method, low, high: see
            `process_dataset`.
        prefix: Filename prefix override. If None, defaults per
            `_default_prefix`.

    Returns:
        A `SampleResult` describing what was written (or the error).
    """
    try:
        scene = reader.read(sample.path)
        logger.info(
            "Scene ready: dataset=%s split=%s shape=%s dtype=%s",
            scene.metadata.get("dataset"),
            sample.split,
            scene.image.shape,
            scene.image.dtype,
        )

        normalized = normalize(scene.image, method=normalize_method, low=low, high=high)
        patches = patchify(normalized, patch_size=patch_size, stride=stride, pad=pad)

        if not patches:
            logger.warning(
                "No patches produced for %s (image shape %s, patch_size %d)",
                sample.path,
                normalized.shape,
                patch_size,
            )
            return SampleResult(sample=sample, written=[], error=None)

        dataset_name = scene.metadata.get("dataset", "scene")
        sample_output_dir = output_dir if sample.split is None else output_dir / sample.split
        file_prefix = prefix or _default_prefix(dataset_name, sample, input_dir)

        written = export_patches(patches, sample_output_dir, file_prefix, scene.metadata)
        return SampleResult(sample=sample, written=written, error=None)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.exception("Failed to process sample %s (split=%s)", sample.path, sample.split)
        return SampleResult(sample=sample, written=[], error=str(exc))


def _process_sample_worker(
    reader_name: str,
    sample: Sample,
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    stride: Optional[int],
    pad: bool,
    normalize_method: NormalizeMethod,
    low: float,
    high: float,
    prefix: Optional[str],
) -> SampleResult:
    """`ProcessPoolExecutor`-friendly wrapper around `_process_sample`.

    Workers are handed `reader_name` (a plain string) instead of the
    reader module itself, and re-resolve it via `_READERS_BY_NAME`
    locally -- this avoids relying on module objects surviving pickling
    across the process boundary, which is not guaranteed under every
    multiprocessing start method.

    Args:
        reader_name: Key into `_READERS_BY_NAME` (e.g. "dsifn").
        sample, input_dir, output_dir, patch_size, stride, pad,
        normalize_method, low, high, prefix: see `_process_sample`.

    Returns:
        A `SampleResult`, same as `_process_sample`.
    """
    reader = _READERS_BY_NAME[reader_name]
    return _process_sample(
        reader=reader,
        sample=sample,
        input_dir=input_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        pad=pad,
        normalize_method=normalize_method,
        low=low,
        high=high,
        prefix=prefix,
    )


def _load_state(output_dir: Path) -> Dict[str, List[str]]:
    """Load the resume/checkpoint file, if any.

    Args:
        output_dir: Output directory that may contain a state file from
            a previous, possibly-interrupted run.

    Returns:
        Mapping of `Sample.key()` -> list of output file paths (as
        strings) written for that sample in a prior run. Empty dict if
        no state file exists or it can't be parsed.
    """
    state_path = output_dir / _STATE_FILENAME
    if not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse existing state file %s; ignoring it", state_path)
        return {}


def _save_state(output_dir: Path, state: Dict[str, List[str]]) -> None:
    """Persist the resume/checkpoint file.

    Args:
        output_dir: Output directory to write the state file into.
        state: Mapping of `Sample.key()` -> list of output file paths
            (as strings) successfully written so far.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / _STATE_FILENAME
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state))
    tmp_path.replace(state_path)  # atomic on POSIX, avoids truncated files on crash


def _is_already_done(sample: Sample, state: Dict[str, List[str]]) -> Optional[List[Path]]:
    """Check whether a sample was already exported in a prior run.

    A sample only counts as done if its recorded output files still
    exist on disk -- if they were deleted since, it's reprocessed
    rather than silently reported as complete.

    Args:
        sample: The sample to check.
        state: Checkpoint state loaded by `_load_state`.

    Returns:
        The previously-written paths if the sample is confirmed done,
        else None.
    """
    recorded = state.get(sample.key())
    if recorded is None:
        return None
    paths = [Path(p) for p in recorded]
    if paths and not all(p.is_file() for p in paths):
        return None
    return paths


def _write_manifest(
    output_dir: Path,
    reader_name: str,
    patch_size: int,
    stride: Optional[int],
    normalize_method: NormalizeMethod,
    workers: int,
    samples_by_split: Dict[Optional[str], List[Sample]],
    results: List[SampleResult],
    elapsed_seconds: float,
) -> Path:
    """Write a `metadata.json` summary of the run to `output_dir`.

    Args:
        output_dir: Directory the manifest is written into.
        reader_name: Detected reader/dataset name (e.g. "dsifn").
        patch_size: Patch side length used for this run.
        stride: Patch stride used (resolved to `patch_size` if it was
            None, matching `patchify`'s own default).
        normalize_method: Normalization method used.
        workers: Number of worker processes used (1 = serial).
        samples_by_split: Samples grouped by split, as built in
            `process_dataset`.
        results: Every `SampleResult` produced during this run.
        elapsed_seconds: Wall-clock time spent in the processing loop.

    Returns:
        Path to the written `metadata.json`.
    """
    results_by_key = {r.sample.key(): r for r in results}

    splits_summary: Dict[str, dict] = {}
    total_patches = 0
    total_errors = 0
    for split, samples in samples_by_split.items():
        split_name = split or "default"
        num_patches = 0
        num_errors = 0
        for sample in samples:
            result = results_by_key.get(sample.key())
            if result is None:
                continue  # not processed this run (e.g. skip-existing)
            num_patches += len(result.written)
            if result.error is not None:
                num_errors += 1
        splits_summary[split_name] = {
            "num_images": len(samples),
            "num_patches": num_patches,
            "num_errors": num_errors,
        }
        total_patches += num_patches
        total_errors += num_errors

    total_samples = sum(len(s) for s in samples_by_split.values())
    manifest = {
        "dataset": reader_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patch_size": patch_size,
        "stride": stride if stride is not None else patch_size,
        "normalize_method": normalize_method,
        "workers": workers,
        "splits": splits_summary,
        "total_images": total_samples,
        "total_patches": total_patches,
        "total_errors": total_errors,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "images_per_second": round(total_samples / elapsed_seconds, 3) if elapsed_seconds > 0 else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


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
    workers: int = 1,
    skip_existing: bool = False,
) -> List[Path]:
    """Run the full detect -> read -> normalize -> patchify -> export pipeline.

    For split datasets (DSIFN today), every sample in train/, val/,
    and test/ is processed, with a tqdm progress bar per split and
    outputs kept under split-named subdirectories of `output_dir`. For
    split-less datasets, behavior is unchanged: a single scene is read
    from `input_dir` and processed once.

    Samples are processed serially when `workers <= 1`, or across a
    `ProcessPoolExecutor` with that many worker processes otherwise. A
    checkpoint file under `output_dir` records completed samples so a
    later run with `skip_existing=True` can resume without redoing
    work whose output files are still present. A `metadata.json`
    manifest summarizing the run is always written to `output_dir`.

    Args:
        input_dir: Root directory of the dataset to process.
        output_dir: Directory PNG patches are written into. For split
            datasets, patches for split X land under `output_dir/X/`.
        patch_size: Side length of each square patch, in pixels.
        stride: Patch stride; defaults to `patch_size` (non-overlapping).
        pad: Whether to pad each image up to a multiple of
            `patch_size` before patchifying, so no pixels are dropped.
        normalize_method: One of "percentile", "minmax", "zscore".
        low: Lower percentile bound (only used when
            normalize_method="percentile").
        high: Upper percentile bound (only used when
            normalize_method="percentile").
        prefix: Filename prefix for exported PNGs. Defaults per
            `_default_prefix`.
        workers: Number of worker processes to use. 1 (default) runs
            serially in-process, identical to the original behavior.
        skip_existing: If True, samples whose checkpointed output
            files still exist on disk are skipped instead of
            reprocessed. Combine with a previous, interrupted run's
            `--output` to resume it.

    Returns:
        List of paths written by the export stage during *this* call
        (already-done samples skipped via `skip_existing` are not
        included). Empty if no reader matched `input_dir` or no
        samples/patches were produced.
    """
    reader = detect_dataset(input_dir)
    if reader is None:
        _log_unknown_dataset(input_dir)
        return []

    samples_by_split: Dict[Optional[str], List[Sample]] = {}
    for sample in iter_dataset_samples(reader, input_dir):
        samples_by_split.setdefault(sample.split, []).append(sample)

    if not samples_by_split:
        logger.warning("No samples found for %s", input_dir)
        return []

    reader_name = reader.__name__.rsplit(".", 1)[-1]
    state = _load_state(output_dir) if skip_existing else {}

    written: List[Path] = []
    all_results: List[SampleResult] = []
    skipped_count = 0
    start_time = time.monotonic()

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for split, samples in samples_by_split.items():
            progress_desc = f"{reader_name}:{split}" if split else reader_name

            to_process: List[Sample] = []
            for sample in samples:
                done_paths = _is_already_done(sample, state) if skip_existing else None
                if done_paths is not None:
                    skipped_count += 1
                    continue
                to_process.append(sample)

            if not to_process:
                logger.info("Split '%s': nothing to do (all samples already done)", split)
                continue

            if executor is None:
                results_iter = (
                    _process_sample(
                        reader=reader,
                        sample=sample,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        patch_size=patch_size,
                        stride=stride,
                        pad=pad,
                        normalize_method=normalize_method,
                        low=low,
                        high=high,
                        prefix=prefix,
                    )
                    for sample in to_process
                )
                progress = tqdm(results_iter, total=len(to_process), desc=progress_desc, unit="sample")
                for result in progress:
                    all_results.append(result)
                    written.extend(result.written)
                    state[result.sample.key()] = [str(p) for p in result.written]
            else:
                futures = {
                    executor.submit(
                        _process_sample_worker,
                        reader_name,
                        sample,
                        input_dir,
                        output_dir,
                        patch_size,
                        stride,
                        pad,
                        normalize_method,
                        low,
                        high,
                        prefix,
                    ): sample
                    for sample in to_process
                }
                progress = tqdm(as_completed(futures), total=len(futures), desc=progress_desc, unit="sample")
                for future in progress:
                    result = future.result()
                    all_results.append(result)
                    written.extend(result.written)
                    state[result.sample.key()] = [str(p) for p in result.written]

            # Checkpoint after each split so an interruption only loses
            # progress within the split currently in flight.
            _save_state(output_dir, state)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        # Always leave a checkpoint behind, even on an exception, so a
        # resumed run doesn't redo everything that already succeeded.
        _save_state(output_dir, state)

    elapsed_seconds = time.monotonic() - start_time
    total_samples = sum(len(s) for s in samples_by_split.values())
    error_count = sum(1 for r in all_results if r.error is not None)

    manifest_path = _write_manifest(
        output_dir=output_dir,
        reader_name=reader_name,
        patch_size=patch_size,
        stride=stride,
        normalize_method=normalize_method,
        workers=workers,
        samples_by_split=samples_by_split,
        results=all_results,
        elapsed_seconds=elapsed_seconds,
    )

    images_per_sec = (len(all_results) / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    logger.info(
        "Processed %d/%d sample(s) across %d split(s) for %s (%d skipped, %d error(s)): "
        "%d file(s) written in %.1fs (%.2f images/sec). Manifest: %s",
        len(all_results),
        total_samples,
        len(samples_by_split),
        input_dir,
        skipped_count,
        error_count,
        len(written),
        elapsed_seconds,
        images_per_sec,
        manifest_path,
    )
    return written


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel preprocessing (default: 1, serial).",
    )
    parser.add_argument(
        "--skip-existing",
        "--resume",
        dest="skip_existing",
        action="store_true",
        help=(
            "Skip samples whose output patches were already written in a previous run "
            "to this --output directory (checked via a checkpoint file there)."
        ),
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
        workers=args.workers,
        skip_existing=args.skip_existing,
    )
    logger.info("Done. %d file(s) written.", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
