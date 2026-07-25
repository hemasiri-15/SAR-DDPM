"""Tests for sar_preprocessor.patchify."""

from __future__ import annotations

import numpy as np
import pytest

from sar_preprocessor.patchify import patchify


# --------------------------------------------------------------------------
# Successful cases
# --------------------------------------------------------------------------


class TestPatchifySuccess:
    def test_non_overlapping_2d_exact_fit(self) -> None:
        image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        patches = patchify(image, patch_size=8)
        assert len(patches) == 4  # 2x2 grid of 8x8 patches
        offsets = {offset for _patch, offset in patches}
        assert offsets == {(0, 0), (0, 8), (8, 0), (8, 8)}

    def test_patch_content_matches_source_slice(self) -> None:
        image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        patches = patchify(image, patch_size=8)
        for patch, (row, col) in patches:
            expected = image[row : row + 8, col : col + 8]
            assert np.array_equal(patch, expected)

    def test_patch_shape_2d(self) -> None:
        image = np.zeros((16, 16), dtype=np.float32)
        patches = patchify(image, patch_size=4)
        for patch, _offset in patches:
            assert patch.shape == (4, 4)

    def test_patch_shape_multiband(self) -> None:
        image = np.zeros((3, 16, 16), dtype=np.float32)
        patches = patchify(image, patch_size=4)
        assert len(patches) == 16  # 4x4 grid
        for patch, _offset in patches:
            assert patch.shape == (3, 4, 4)

    def test_default_stride_equals_patch_size(self) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        default_stride = patchify(image, patch_size=8)
        explicit_stride = patchify(image, patch_size=8, stride=8)
        assert len(default_stride) == len(explicit_stride)
        assert [o for _p, o in default_stride] == [o for _p, o in explicit_stride]

    def test_overlapping_stride_produces_more_patches(self) -> None:
        image = np.zeros((16, 16), dtype=np.float32)
        non_overlapping = patchify(image, patch_size=8, stride=8)
        overlapping = patchify(image, patch_size=8, stride=4)
        assert len(overlapping) > len(non_overlapping)

    def test_drop_remainder_when_not_divisible(self) -> None:
        # 10x10 image, patch_size=4, stride=4 -> only rows/cols 0..3, 4..7
        # fit; the trailing 2 pixels are dropped (pad=False default).
        image = np.zeros((10, 10), dtype=np.float32)
        patches = patchify(image, patch_size=4, pad=False)
        offsets = {offset for _patch, offset in patches}
        assert offsets == {(0, 0), (0, 4), (4, 0), (4, 4)}
        assert 8 not in {r for r, _c in offsets}

    def test_pad_true_covers_all_pixels(self) -> None:
        # 10x10 image padded up to 12x12 with patch_size=4 -> a 3x3 grid.
        image = np.ones((10, 10), dtype=np.float32)
        patches = patchify(image, patch_size=4, pad=True, pad_mode="constant")
        assert len(patches) == 9
        offsets = {offset for _patch, offset in patches}
        assert offsets == {(r, c) for r in (0, 4, 8) for c in (0, 4, 8)}

    def test_pad_produces_more_patches_than_no_pad(self) -> None:
        image = np.zeros((10, 10), dtype=np.float32)
        no_pad = patchify(image, patch_size=4, pad=False)
        with_pad = patchify(image, patch_size=4, pad=True)
        assert len(with_pad) > len(no_pad)

    def test_patches_are_copies_not_views(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        patches = patchify(image, patch_size=4)
        patch, _offset = patches[0]
        patch[0, 0] = 999.0
        assert image[0, 0] == 0.0  # original untouched

    def test_exact_fit_with_no_padding_needed(self) -> None:
        image = np.zeros((16, 16), dtype=np.float32)
        no_pad = patchify(image, patch_size=8, pad=False)
        with_pad = patchify(image, patch_size=8, pad=True)
        # No padding was actually necessary, so results should match.
        assert len(no_pad) == len(with_pad) == 4


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


class TestPatchifyEdgeCases:
    def test_patch_size_larger_than_image_returns_empty(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        patches = patchify(image, patch_size=16)
        assert patches == []

    def test_patch_size_equal_to_image_size(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        patches = patchify(image, patch_size=8)
        assert len(patches) == 1
        assert patches[0][1] == (0, 0)

    def test_empty_image_returns_empty_list(self) -> None:
        image = np.zeros((0, 0), dtype=np.float32)
        patches = patchify(image, patch_size=4)
        assert patches == []

    def test_patch_size_one(self) -> None:
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        patches = patchify(image, patch_size=1)
        assert len(patches) == 9
        for patch, (row, col) in patches:
            assert patch.shape == (1, 1)
            assert patch[0, 0] == image[row, col]

    def test_odd_patch_size_with_pad(self) -> None:
        image = np.ones((10, 10), dtype=np.float32)
        patches = patchify(image, patch_size=3, pad=True, pad_mode="constant")
        # ceil(10/3) = 4 -> padded to 12x12 -> 4x4 grid = 16 patches
        assert len(patches) == 16

    def test_odd_patch_size_without_pad(self) -> None:
        image = np.ones((10, 10), dtype=np.float32)
        patches = patchify(image, patch_size=3, pad=False)
        # floor(10/3) = 3 -> 3x3 grid = 9 patches
        assert len(patches) == 9


# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------


class TestPatchifyFailures:
    def test_zero_patch_size_raises(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="patch_size must be positive"):
            patchify(image, patch_size=0)

    def test_negative_patch_size_raises(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="patch_size must be positive"):
            patchify(image, patch_size=-4)

    def test_zero_stride_raises(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="stride must be positive"):
            patchify(image, patch_size=4, stride=0)

    def test_negative_stride_raises(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="stride must be positive"):
            patchify(image, patch_size=4, stride=-1)

    def test_1d_array_raises(self) -> None:
        image = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            patchify(image, patch_size=4)

    def test_4d_array_raises(self) -> None:
        image = np.zeros((2, 3, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            patchify(image, patch_size=4)
