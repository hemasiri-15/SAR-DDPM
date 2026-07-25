"""Tests for sar_preprocessor.readers (sentinel1, dsifn, uavsar, airsar).

Real Sentinel-1/UAVSAR/AIRSAR products are not required anywhere in this
file. Filesystem-shape detection tests use plain tmp_path directories.
Raster reading is either exercised against small real files we write
ourselves (PNG via Pillow, raw binary via numpy.tofile) or against a
mocked rasterio.open so no actual satellite/airborne data is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import rasterio
from PIL import Image

from sar_preprocessor.readers import airsar, dsifn, sentinel1, uavsar


class _FakeRasterioDataset:
    """Minimal stand-in for a rasterio dataset context manager."""

    def __init__(self, data: np.ndarray, crs: object = "EPSG:4326", transform: object = "fake-transform"):
        self._data = np.asarray(data)
        self.crs = crs
        self.transform = transform

    def __enter__(self) -> "_FakeRasterioDataset":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self, band: Optional[int] = None) -> np.ndarray:
        if band is None:
            # rasterio's no-arg read() always returns (bands, H, W).
            if self._data.ndim == 2:
                return self._data[np.newaxis, ...]
            return self._data
        # rasterio's read(n) returns a single 2D band.
        if self._data.ndim == 2:
            return self._data
        return self._data[band - 1]


# ==========================================================================
# Sentinel-1
# ==========================================================================


class TestSentinel1Detect:
    def test_manifest_directly_in_path(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.safe").write_text("x")
        assert sentinel1.detect(tmp_path) is True

    def test_manifest_inside_safe_subfolder(self, tmp_path: Path) -> None:
        safe = tmp_path / "S1A_IW_GRDH.SAFE"
        safe.mkdir()
        (safe / "manifest.safe").write_text("x")
        assert sentinel1.detect(tmp_path) is True

    def test_no_manifest_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "unrelated.txt").write_text("x")
        assert sentinel1.detect(tmp_path) is False

    def test_empty_directory_returns_false(self, tmp_path: Path) -> None:
        assert sentinel1.detect(tmp_path) is False

    def test_manifest_too_deep_returns_false(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "manifest.safe").write_text("x")
        assert sentinel1.detect(tmp_path) is False

    def test_nonexistent_path_returns_false(self, tmp_path: Path) -> None:
        assert sentinel1.detect(tmp_path / "does_not_exist") is False


class TestSentinel1Read:
    def test_single_band_read(self, tmp_path: Path, mocker) -> None:
        safe = tmp_path / "S1A_TEST.SAFE"
        (safe / "measurement").mkdir(parents=True)
        (safe / "manifest.safe").write_text("x")
        (safe / "measurement" / "s1a-iw-grd-vv-test.tiff").write_bytes(b"fake")

        fake_data = np.arange(16, dtype=np.float32).reshape(4, 4)
        mocker.patch(
            "sar_preprocessor.readers.sentinel1.rasterio.open",
            return_value=_FakeRasterioDataset(fake_data),
        )

        scene = sentinel1.read(safe)
        assert scene.image.shape == (4, 4)
        assert scene.image.dtype == np.float32
        assert np.array_equal(scene.image, fake_data)
        assert scene.metadata["dataset"] == "sentinel1"
        assert scene.metadata["sensor"] == "S1A"
        assert scene.metadata["polarization"] == "VV"
        assert scene.metadata["crs"] == "EPSG:4326"

    def test_multi_band_read_stacked_sorted(self, tmp_path: Path, mocker) -> None:
        safe = tmp_path / "S1B_TEST.SAFE"
        (safe / "measurement").mkdir(parents=True)
        (safe / "manifest.safe").write_text("x")
        (safe / "measurement" / "s1b-iw-grd-vh-test.tiff").write_bytes(b"fake")
        (safe / "measurement" / "s1b-iw-grd-vv-test.tiff").write_bytes(b"fake")

        band = np.ones((2, 2), dtype=np.float32)
        mocker.patch(
            "sar_preprocessor.readers.sentinel1.rasterio.open",
            return_value=_FakeRasterioDataset(band),
        )

        scene = sentinel1.read(safe)
        assert scene.image.shape == (2, 2, 2)  # 2 bands, sorted vh then vv
        assert scene.metadata["polarization"] == ["VH", "VV"]
        assert scene.metadata["sensor"] == "S1B"

    def test_missing_safe_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest.safe"):
            sentinel1.read(tmp_path)

    def test_missing_measurement_dir_raises(self, tmp_path: Path) -> None:
        safe = tmp_path / "S1A_TEST.SAFE"
        safe.mkdir()
        (safe / "manifest.safe").write_text("x")
        with pytest.raises(FileNotFoundError, match="measurement"):
            sentinel1.read(safe)

    def test_empty_measurement_dir_raises(self, tmp_path: Path) -> None:
        safe = tmp_path / "S1A_TEST.SAFE"
        (safe / "measurement").mkdir(parents=True)
        (safe / "manifest.safe").write_text("x")
        with pytest.raises(FileNotFoundError, match="No .tif"):
            sentinel1.read(safe)

    def test_unknown_sensor_prefix_is_none(self, tmp_path: Path, mocker) -> None:
        safe = tmp_path / "UNKNOWN_TEST.SAFE"
        (safe / "measurement").mkdir(parents=True)
        (safe / "manifest.safe").write_text("x")
        (safe / "measurement" / "band.tiff").write_bytes(b"fake")

        mocker.patch(
            "sar_preprocessor.readers.sentinel1.rasterio.open",
            return_value=_FakeRasterioDataset(np.zeros((2, 2), dtype=np.float32)),
        )
        scene = sentinel1.read(safe)
        assert scene.metadata["sensor"] is None


# ==========================================================================
# DSIFN
# ==========================================================================


class TestDsifnDetect:
    def test_all_splits_present(self, tmp_path: Path) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / split).mkdir()
        assert dsifn.detect(tmp_path) is True

    def test_missing_one_split_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        # no "test"
        assert dsifn.detect(tmp_path) is False

    def test_no_splits_returns_false(self, tmp_path: Path) -> None:
        assert dsifn.detect(tmp_path) is False

    def test_nonexistent_path_returns_false(self, tmp_path: Path) -> None:
        assert dsifn.detect(tmp_path / "nope") is False


class TestDsifnRead:
    def test_reads_first_image_via_pillow_fallback(self, tmp_path: Path, mocker) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / split).mkdir()
        img_dir = tmp_path / "train" / "images"
        img_dir.mkdir()
        array = (np.arange(48).reshape(4, 4, 3) % 255).astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(img_dir / "sample.png")

        # Force the rasterio branch to fail so the Pillow fallback path is
        # exercised deterministically.
        mocker.patch(
            "sar_preprocessor.readers.dsifn.rasterio.open",
            side_effect=rasterio.errors.RasterioIOError("not a raster"),
        )

        scene = dsifn.read(tmp_path)
        assert scene.image.shape == (3, 4, 4)  # (C, H, W) after moveaxis
        assert scene.metadata["dataset"] == "dsifn"
        assert scene.metadata["split"] == "train"
        assert scene.metadata["crs"] is None

    def test_reads_via_rasterio_when_available(self, tmp_path: Path, mocker) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / split).mkdir()
        img_dir = tmp_path / "train"
        (img_dir / "sample.tif").write_bytes(b"fake")

        fake_data = np.ones((1, 5, 5), dtype=np.float32)
        mocker.patch(
            "sar_preprocessor.readers.dsifn.rasterio.open",
            return_value=_FakeRasterioDataset(fake_data),
        )

        scene = dsifn.read(tmp_path)
        assert scene.image.shape == (5, 5)  # single band squeezed
        assert scene.metadata["crs"] == "EPSG:4326"

    def test_falls_back_to_val_when_train_empty(self, tmp_path: Path, mocker) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / split).mkdir()
        array = np.zeros((2, 2, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(tmp_path / "val" / "sample.png")

        mocker.patch(
            "sar_preprocessor.readers.dsifn.rasterio.open",
            side_effect=rasterio.errors.RasterioIOError("not a raster"),
        )

        scene = dsifn.read(tmp_path)
        assert scene.metadata["split"] == "val"

    def test_no_images_in_any_split_raises(self, tmp_path: Path) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / split).mkdir()
        with pytest.raises(FileNotFoundError, match="No image files"):
            dsifn.read(tmp_path)


# ==========================================================================
# UAVSAR
# ==========================================================================


class TestUavsarDetect:
    def test_matching_ann_grd_pair(self, tmp_path: Path) -> None:
        (tmp_path / "scene.ann").write_text("x")
        (tmp_path / "scene.grd").write_bytes(b"\x00" * 16)
        assert uavsar.detect(tmp_path) is True

    def test_ann_without_grd_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "scene.ann").write_text("x")
        assert uavsar.detect(tmp_path) is False

    def test_grd_without_ann_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "scene.grd").write_bytes(b"\x00" * 16)
        assert uavsar.detect(tmp_path) is False

    def test_empty_directory_returns_false(self, tmp_path: Path) -> None:
        assert uavsar.detect(tmp_path) is False


class TestUavsarRead:
    def test_float32_grd_read(self, tmp_path: Path) -> None:
        rows, cols = 4, 5
        data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        (tmp_path / "scene_hhhh.ann").write_text(
            f"grd_pwr.set_rows (pixels) = {rows} ; rows\n"
            f"grd_pwr.set_cols (pixels) = {cols} ; cols\n"
        )
        data.tofile(tmp_path / "scene_hhhh.grd")

        scene = uavsar.read(tmp_path)
        assert scene.image.shape == (rows, cols)
        assert scene.image.dtype == np.float32
        assert np.array_equal(scene.image, data)
        assert scene.metadata["dataset"] == "uavsar"
        assert scene.metadata["polarization"] == "HHHH"

    def test_complex64_grd_fallback(self, tmp_path: Path) -> None:
        rows, cols = 3, 3
        data = (np.arange(rows * cols) + 1j * np.arange(rows * cols)).astype(
            np.complex64
        ).reshape(rows, cols)
        (tmp_path / "scene.ann").write_text(
            f"rows (pixels) = {rows}\ncols (pixels) = {cols}\n"
        )
        data.tofile(tmp_path / "scene.grd")

        scene = uavsar.read(tmp_path)
        assert scene.image.shape == (rows, cols)
        assert np.iscomplexobj(scene.image)
        assert np.allclose(scene.image, data)

    def test_missing_pair_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            uavsar.read(tmp_path)

    def test_unparseable_dimensions_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "scene.ann").write_text("some_other_key = 123\n")
        (tmp_path / "scene.grd").write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="Could not determine raster dimensions"):
            uavsar.read(tmp_path)

    def test_size_mismatch_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "scene.ann").write_text(
            "rows (pixels) = 10\ncols (pixels) = 10\n"
        )
        # 10x10 float32 should be 400 bytes; write something else entirely.
        (tmp_path / "scene.grd").write_bytes(b"\x00" * 17)
        with pytest.raises(ValueError, match="Could not reconcile"):
            uavsar.read(tmp_path)


# ==========================================================================
# AIRSAR
# ==========================================================================


class TestAirsarDetect:
    def test_filename_contains_airsar(self, tmp_path: Path) -> None:
        (tmp_path / "flight_airsar_scene.tif").write_bytes(b"fake")
        assert airsar.detect(tmp_path) is True

    def test_airsar_extension(self, tmp_path: Path) -> None:
        (tmp_path / "scene.airsar").write_bytes(b"fake")
        assert airsar.detect(tmp_path) is True

    def test_no_match_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "random_data.tif").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("nothing to see here")
        assert airsar.detect(tmp_path) is False

    def test_empty_directory_returns_false(self, tmp_path: Path) -> None:
        assert airsar.detect(tmp_path) is False

    def test_does_not_blindly_match_unrelated_dataset(self, tmp_path: Path) -> None:
        # Regression guard: AIRSAR must never be a fallback match for an
        # otherwise-unrecognized directory.
        (tmp_path / "manifest.safe").write_text("x")  # a sentinel1 file
        assert airsar.detect(tmp_path) is False

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        (tmp_path / "AIRSAR_Product.dat").write_bytes(b"fake")
        assert airsar.detect(tmp_path) is True


class TestAirsarRead:
    def test_reads_via_rasterio_when_openable(self, tmp_path: Path, mocker) -> None:
        data_path = tmp_path / "flight_airsar_hh.tif"
        data_path.write_bytes(b"fake")

        fake_data = np.full((1, 4, 4), 3.0, dtype=np.float32)
        mocker.patch(
            "sar_preprocessor.readers.airsar.rasterio.open",
            return_value=_FakeRasterioDataset(fake_data),
        )

        scene = airsar.read(tmp_path)
        assert scene.image.shape == (4, 4)
        assert scene.metadata["dataset"] == "airsar"
        assert scene.metadata["polarization"] == "HH"
        assert scene.metadata["crs"] == "EPSG:4326"

    def test_raw_binary_sidecar_fallback(self, tmp_path: Path) -> None:
        data = (np.arange(20, dtype=np.float32)).reshape(4, 5)
        data_path = tmp_path / "airsar_sample.dat"
        data.tofile(data_path)
        (tmp_path / "airsar_sample.hdr").write_text(
            "width: 5\nheight: 4\ndtype: float32\n"
        )

        scene = airsar.read(tmp_path)
        assert scene.image.shape == (4, 5)
        assert np.array_equal(scene.image, data)
        assert scene.metadata["crs"] is None

    def test_no_rasterio_no_sidecar_raises_value_error(self, tmp_path: Path) -> None:
        data_path = tmp_path / "airsar_sample.dat"
        data_path.write_bytes(b"not a real raster and no sidecar")
        with pytest.raises(ValueError, match="sidecar"):
            airsar.read(tmp_path)

    def test_sidecar_missing_dimensions_raises(self, tmp_path: Path) -> None:
        data_path = tmp_path / "airsar_sample.dat"
        data_path.write_bytes(b"\x00" * 16)
        (tmp_path / "airsar_sample.hdr").write_text("dtype: float32\n")
        with pytest.raises(ValueError, match="width and height"):
            airsar.read(tmp_path)

    def test_sidecar_unsupported_dtype_raises(self, tmp_path: Path) -> None:
        data_path = tmp_path / "airsar_sample.dat"
        data_path.write_bytes(b"\x00" * 16)
        (tmp_path / "airsar_sample.hdr").write_text(
            "width: 2\nheight: 2\ndtype: not_a_real_dtype\n"
        )
        with pytest.raises(ValueError, match="Unsupported dtype"):
            airsar.read(tmp_path)

    def test_sidecar_size_mismatch_raises(self, tmp_path: Path) -> None:
        data_path = tmp_path / "airsar_sample.dat"
        data_path.write_bytes(b"\x00" * 10)  # wrong size
        (tmp_path / "airsar_sample.hdr").write_text(
            "width: 4\nheight: 4\ndtype: float32\n"  # expects 64 bytes
        )
        with pytest.raises(ValueError, match="expected"):
            airsar.read(tmp_path)

    def test_no_matching_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            airsar.read(tmp_path)
