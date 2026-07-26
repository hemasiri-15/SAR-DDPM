"""Tests for orchestration in build_dataset.py.

These tests exercise the orchestration layer in isolation: they use a
fake reader module (a real, importable module -- required so it can be
looked up by name from a worker process -- rather than a Mock/
SimpleNamespace) with stubbed `detect`/`read`, and rely on the real
`normalize`/`patchify`/`export_patches` stubs installed in this package
so patch counts are deterministic (`patchify` always returns exactly 2
patches per sample, see sar_preprocessor/patchify.py).

Covers, per the review that asked for an expanded suite:
    - iter_dataset_samples(): split-less fallback, default split
      traversal, a reader-provided iter_split_samples override,
      missing split dirs, and empty split dirs.
    - metadata.json generation (counts, patch totals).
    - filename uniqueness across samples in the same split.
    - resume / --skip-existing behavior.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sar_preprocessor import build_dataset
from sar_preprocessor.scene import SARScene, make_metadata


def _make_fake_reader(module_name: str, splits=None, iter_split_samples=None):
    """Create and register a real, importable fake reader module.

    A real module (registered in sys.modules) is used -- rather than a
    Mock or SimpleNamespace -- because `_process_sample_worker` looks
    reader modules up by name via `_READERS_BY_NAME`, which must resolve
    correctly even from a separate worker process (relevant for the
    --workers > 1 tests).

    Args:
        module_name: Unique dotted name to register the module under.
        splits: Optional SPLITS tuple to attach to the module.
        iter_split_samples: Optional callable to attach as the module's
            iter_split_samples.

    Returns:
        The created module, already inserted into build_dataset's
        READERS/_READERS_BY_NAME and sys.modules.
    """
    module = types.ModuleType(module_name)

    def detect(path: Path) -> bool:
        return True

    def read(path: Path) -> SARScene:
        return SARScene(
            image=np.zeros((8, 8), dtype="float32"),
            metadata=make_metadata(dataset=module_name),
        )

    module.detect = detect
    module.read = read
    if splits is not None:
        module.SPLITS = splits
    if iter_split_samples is not None:
        module.iter_split_samples = iter_split_samples

    sys.modules[module_name] = module
    return module


def _register_reader(monkeypatch, reader_name, module):
    """Point build_dataset's reader registry at a fake reader for one test."""
    monkeypatch.setitem(build_dataset._READERS_BY_NAME, reader_name, module)
    monkeypatch.setattr(build_dataset, "READERS", [(reader_name, module)])
    monkeypatch.setattr(build_dataset, "detect_dataset", lambda path: module)


# --------------------------------------------------------------------------
# iter_dataset_samples
# --------------------------------------------------------------------------


def test_iter_dataset_samples_no_splits_falls_back_to_single_sample(tmp_path):
    reader = _make_fake_reader("fake_no_splits")
    samples = list(build_dataset.iter_dataset_samples(reader, tmp_path))
    assert samples == [build_dataset.Sample(path=tmp_path, split=None)]


def test_iter_dataset_samples_default_listing_across_splits(tmp_path):
    for split in ("train", "val", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "b.png").write_bytes(b"")
        (split_dir / "a.png").write_bytes(b"")
        (split_dir / "notes.txt").write_bytes(b"")  # should be ignored

    reader = _make_fake_reader("fake_default_listing", splits=("train", "val", "test"))
    samples = list(build_dataset.iter_dataset_samples(reader, tmp_path))

    by_split = {}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample.path.name)

    assert by_split == {
        "train": ["a.png", "b.png"],
        "val": ["a.png", "b.png"],
        "test": ["a.png", "b.png"],
    }


def test_iter_dataset_samples_uses_reader_provided_iter_split_samples(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "scene_A").mkdir()  # a non-flat, paired-folder layout

    def custom_iter_split_samples(split_dir: Path):
        return [split_dir / "scene_A"]

    reader = _make_fake_reader(
        "fake_custom_split", splits=("train",), iter_split_samples=custom_iter_split_samples
    )
    samples = list(build_dataset.iter_dataset_samples(reader, tmp_path))

    assert len(samples) == 1
    assert samples[0].path.name == "scene_A"
    assert samples[0].split == "train"


def test_iter_dataset_samples_skips_missing_split_dir(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "a.png").write_bytes(b"")
    # val/ and test/ deliberately absent

    reader = _make_fake_reader("fake_missing_split", splits=("train", "val", "test"))
    samples = list(build_dataset.iter_dataset_samples(reader, tmp_path))

    assert [s.split for s in samples] == ["train"]


def test_iter_dataset_samples_skips_empty_split_dir(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "a.png").write_bytes(b"")
    (tmp_path / "val").mkdir()  # exists but empty

    reader = _make_fake_reader("fake_empty_split", splits=("train", "val"))
    samples = list(build_dataset.iter_dataset_samples(reader, tmp_path))

    assert [s.split for s in samples] == ["train"]


# --------------------------------------------------------------------------
# process_dataset: split-aware output, manifest, filename uniqueness
# --------------------------------------------------------------------------


def test_process_dataset_writes_split_output_dirs_and_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    for split, count in (("train", 3), ("val", 2), ("test", 1)):
        split_dir = input_dir / split
        split_dir.mkdir(parents=True)
        for i in range(count):
            (split_dir / f"img_{i}.png").write_bytes(b"")

    reader = _make_fake_reader("fake_manifest_run", splits=("train", "val", "test"))
    _register_reader(monkeypatch, "fake_manifest_run", reader)

    written = build_dataset.process_dataset(input_dir=input_dir, output_dir=output_dir, patch_size=4)

    # 6 total samples * 4 patches/sample = 24 files.
    assert len(written) == 24
    assert (output_dir / "train").is_dir()
    assert (output_dir / "val").is_dir()
    assert (output_dir / "test").is_dir()
    assert len(list((output_dir / "train").glob("*.png"))) == 12  # 3 samples * 2 patches

    manifest = json.loads((output_dir / build_dataset._MANIFEST_FILENAME).read_text())
    assert manifest["dataset"] == "fake_manifest_run"
    assert manifest["splits"]["train"]["num_images"] == 3
    assert manifest["splits"]["train"]["num_patches"] == 12
    assert manifest["splits"]["val"]["num_images"] == 2
    assert manifest["splits"]["test"]["num_images"] == 1
    assert manifest["total_images"] == 6
    assert manifest["total_patches"] == 24
    assert manifest["total_errors"] == 0
    assert manifest["workers"] == 1


def test_process_dataset_filenames_unique_across_samples_in_a_split(tmp_path, monkeypatch):
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    split_dir = input_dir / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "sceneA.png").write_bytes(b"")
    (split_dir / "sceneB.png").write_bytes(b"")

    reader = _make_fake_reader("fake_unique_names", splits=("train",))
    _register_reader(monkeypatch, "fake_unique_names", reader)

    written = build_dataset.process_dataset(input_dir=input_dir, output_dir=output_dir, patch_size=4)

    assert len(written) == len(set(written)) == 8  # no filename collisions
    stems = {
        p.stem.rsplit("_r", 1)[0]
        for p in written
    }
    assert stems == {"fake_unique_names_sceneA", "fake_unique_names_sceneB"}


def test_process_dataset_split_less_dataset_keeps_original_layout(tmp_path, monkeypatch):
    input_dir = tmp_path / "single_scene_dataset"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    reader = _make_fake_reader("fake_no_splits_run")  # no SPLITS
    _register_reader(monkeypatch, "fake_no_splits_run", reader)

    written = build_dataset.process_dataset(input_dir=input_dir, output_dir=output_dir, patch_size=4)

    assert len(written) == 4
    for p in written:
        assert p.parent == output_dir  # not nested under a split subdir
        assert p.name.startswith(f"fake_no_splits_run_{input_dir.name}")


# --------------------------------------------------------------------------
# resume / --skip-existing
# --------------------------------------------------------------------------


def test_process_dataset_skip_existing_avoids_reprocessing(tmp_path, monkeypatch):
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    split_dir = input_dir / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "a.png").write_bytes(b"")
    (split_dir / "b.png").write_bytes(b"")

    read_calls = []
    reader = _make_fake_reader("fake_resume_run", splits=("train",))

    def counting_read(path: Path) -> SARScene:
        read_calls.append(path)
        return SARScene(
            image=np.zeros((8, 8), dtype="float32"),
            metadata=make_metadata(dataset="fake_resume_run"),
        )

    reader.read = counting_read
    _register_reader(monkeypatch, "fake_resume_run", reader)

    first_written = build_dataset.process_dataset(
        input_dir=input_dir, output_dir=output_dir, patch_size=4, skip_existing=True
    )
    assert len(read_calls) == 2
    assert len(first_written) == 8

    # Second run: everything already has output on disk -> nothing reprocessed.
    second_written = build_dataset.process_dataset(
        input_dir=input_dir, output_dir=output_dir, patch_size=4, skip_existing=True
    )
    assert len(read_calls) == 2  # unchanged: no new read() calls
    assert second_written == []  # nothing *newly* written this call

    manifest = json.loads((output_dir / build_dataset._MANIFEST_FILENAME).read_text())
    assert manifest["total_patches"] == 0  # nothing processed on this (second) run


def test_process_dataset_skip_existing_reprocesses_if_output_deleted(tmp_path, monkeypatch):
    input_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    split_dir = input_dir / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "a.png").write_bytes(b"")

    reader = _make_fake_reader("fake_resume_delete_run", splits=("train",))
    _register_reader(monkeypatch, "fake_resume_delete_run", reader)

    build_dataset.process_dataset(input_dir=input_dir, output_dir=output_dir, patch_size=4, skip_existing=True)

    # Delete the exported patches but keep the checkpoint file.
    for f in (output_dir / "train").glob("*.png"):
        f.unlink()

    second_written = build_dataset.process_dataset(
        input_dir=input_dir, output_dir=output_dir, patch_size=4, skip_existing=True
    )
    assert len(second_written) == 4
    # reprocessed since output was missing


# --------------------------------------------------------------------------
# parallel workers
# --------------------------------------------------------------------------


def test_process_dataset_parallel_matches_serial_output(tmp_path, monkeypatch):
    """--workers > 1 should produce the same result set as serial processing.

    Note: ProcessPoolExecutor spawns/forks real subprocesses, so the fake
    reader must be a genuine, importable module (see _make_fake_reader) --
    on start methods other than "fork" a worker cannot see a
    monkeypatched-only object. This test is skipped if the platform's
    default start method can't see module-level state, which is the
    common case on Linux CI (fork).
    """
    import multiprocessing

    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("requires fork start method for the fake reader module to be visible")

    input_dir = tmp_path / "dataset"
    output_dir_serial = tmp_path / "out_serial"
    output_dir_parallel = tmp_path / "out_parallel"
    split_dir = input_dir / "train"
    split_dir.mkdir(parents=True)
    for i in range(5):
        (split_dir / f"img_{i}.png").write_bytes(b"")

    reader = _make_fake_reader("fake_parallel_run", splits=("train",))
    _register_reader(monkeypatch, "fake_parallel_run", reader)

    serial_written = build_dataset.process_dataset(
        input_dir=input_dir, output_dir=output_dir_serial, patch_size=4, workers=1
    )
    parallel_written = build_dataset.process_dataset(
        input_dir=input_dir, output_dir=output_dir_parallel, patch_size=4, workers=2
    )

    assert len(serial_written) == len(parallel_written) == 20
    assert {p.name for p in serial_written} == {p.name for p in parallel_written}
