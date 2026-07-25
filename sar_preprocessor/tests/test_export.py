"""Tests for sar_preprocessor.export."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

from sar_preprocessor.export import export_patches
from sar_preprocessor.patchify import Patch

MINIMAL_METADATA = {
    "dataset": "test",
    "sensor": None,
    "polarization": None,
    "look_number": None,
    "crs": None,
    "transform": None,
}


def _make_patches(images_and_offsets: List[Tuple[np.ndarray, Tuple[int, int]]]) -> List[Patch]:
    return list(images_and_offsets)


# --------------------------------------------------------------------------
# Successful cases
# --------------------------------------------------------------------------


class TestExportSuccess:
    def test_single_band_patch_written_as_grayscale(self, tmp_path: Path) -> None:
        patch = np.full((4, 4), 0.5, dtype=np.float32)
        patches = _make_patches([(patch, (0, 0))])

        written = export_patches(patches, tmp_path, prefix="scene", metadata=MINIMAL_METADATA)

        assert len(written) == 1
        out_path = written[0]
        assert out_path.name == "scene_r0_c0.png"
        assert out_path.exists()

        with Image.open(out_path) as img:
            assert img.mode == "L"
            array = np.asarray(img)
        assert array.shape == (4, 4)
        assert np.all(array == 128)  # round(0.5 * 255) == 128 (banker's rounding)

    def test_three_band_patch_written_as_rgb(self, tmp_path: Path) -> None:
        patch = np.stack(
            [
                np.full((4, 4), 1.0, dtype=np.float32),
                np.full((4, 4), 0.0, dtype=np.float32),
                np.full((4, 4), 0.5, dtype=np.float32),
            ],
            axis=0,
        )
        patches = _make_patches([(patch, (0, 0))])

        written = export_patches(patches, tmp_path, prefix="rgb", metadata=MINIMAL_METADATA)

        assert len(written) == 1
        out_path = written[0]
        assert out_path.name == "rgb_r0_c0.png"

        with Image.open(out_path) as img:
            assert img.mode == "RGB"
            array = np.asarray(img)
        assert array.shape == (4, 4, 3)
        assert np.all(array[:, :, 0] == 255)
        assert np.all(array[:, :, 1] == 0)
        assert np.all(array[:, :, 2] == 128)

    def test_two_band_patch_written_per_band(self, tmp_path: Path) -> None:
        patch = np.stack(
            [
                np.full((4, 4), 1.0, dtype=np.float32),
                np.full((4, 4), 0.0, dtype=np.float32),
            ],
            axis=0,
        )
        patches = _make_patches([(patch, (2, 3))])

        written = export_patches(patches, tmp_path, prefix="dual", metadata=MINIMAL_METADATA)

        assert len(written) == 2
        names = sorted(p.name for p in written)
        assert names == ["dual_r2_c3_b0.png", "dual_r2_c3_b1.png"]
        for out_path in written:
            with Image.open(out_path) as img:
                assert img.mode == "L"

    def test_four_band_patch_written_per_band(self, tmp_path: Path) -> None:
        patch = np.zeros((4, 6, 6), dtype=np.float32)
        patches = _make_patches([(patch, (0, 0))])

        written = export_patches(patches, tmp_path, prefix="quad", metadata=MINIMAL_METADATA)

        assert len(written) == 4
        names = sorted(p.name for p in written)
        assert names == [
            "quad_r0_c0_b0.png",
            "quad_r0_c0_b1.png",
            "quad_r0_c0_b2.png",
            "quad_r0_c0_b3.png",
        ]

    def test_multiple_patches_all_written(self, tmp_path: Path) -> None:
        patches = _make_patches(
            [
                (np.zeros((4, 4), dtype=np.float32), (0, 0)),
                (np.ones((4, 4), dtype=np.float32), (0, 4)),
                (np.full((4, 4), 0.5, dtype=np.float32), (4, 0)),
            ]
        )
        written = export_patches(patches, tmp_path, prefix="multi", metadata=MINIMAL_METADATA)
        assert len(written) == 3
        for out_path in written:
            assert out_path.exists()

    def test_output_directory_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        assert not nested.exists()
        patch = np.zeros((4, 4), dtype=np.float32)
        export_patches([(patch, (0, 0))], nested, prefix="scene", metadata=MINIMAL_METADATA)
        assert nested.is_dir()

    def test_values_outside_unit_range_are_clipped(self, tmp_path: Path) -> None:
        patch = np.array(
            [[-1.0, 2.0], [0.5, 1.5]],
            dtype=np.float32,
        )
        written = export_patches(
            [(patch, (0, 0))], tmp_path, prefix="clip", metadata=MINIMAL_METADATA
        )
        with Image.open(written[0]) as img:
            array = np.asarray(img)
        assert array[0, 0] == 0  # -1.0 clipped to 0.0 -> 0
        assert array[0, 1] == 255  # 2.0 clipped to 1.0 -> 255

    def test_returned_paths_order_matches_input_order(self, tmp_path: Path) -> None:
        patches = _make_patches(
            [
                (np.zeros((2, 2), dtype=np.float32), (0, 0)),
                (np.zeros((2, 2), dtype=np.float32), (0, 2)),
            ]
        )
        written = export_patches(patches, tmp_path, prefix="order", metadata=MINIMAL_METADATA)
        assert written[0].name == "order_r0_c0.png"
        assert written[1].name == "order_r0_c2.png"


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


class TestExportEdgeCases:
    def test_empty_patch_list_returns_empty_and_creates_dir(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "empty_run"
        written = export_patches([], out_dir, prefix="none", metadata=MINIMAL_METADATA)
        assert written == []
        assert out_dir.is_dir()

    def test_metadata_missing_dataset_key_does_not_crash(self, tmp_path: Path) -> None:
        patch = np.zeros((4, 4), dtype=np.float32)
        written = export_patches([(patch, (0, 0))], tmp_path, prefix="scene", metadata={})
        assert len(written) == 1

    def test_zero_size_patch_dimension_skipped_not_crashed(self, tmp_path: Path) -> None:
        # A patch with a zero-length dimension can't be saved as a PNG
        # (Pillow has no representation for a 0x4 image); export_patches
        # should skip it gracefully rather than raising.
        patch = np.zeros((0, 4), dtype=np.float32)
        written = export_patches([(patch, (0, 0))], tmp_path, prefix="zero", metadata=MINIMAL_METADATA)
        assert written == []

    def test_zero_size_patch_mixed_with_valid_patches(self, tmp_path: Path) -> None:
        zero_patch = np.zeros((0, 4), dtype=np.float32)
        valid_patch = np.full((4, 4), 0.5, dtype=np.float32)
        written = export_patches(
            [(zero_patch, (0, 0)), (valid_patch, (0, 4))],
            tmp_path,
            prefix="mixed",
            metadata=MINIMAL_METADATA,
        )
        assert len(written) == 1
        assert written[0].name == "mixed_r0_c4.png"
