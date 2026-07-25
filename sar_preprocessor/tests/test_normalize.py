"""Tests for sar_preprocessor.normalize."""

from __future__ import annotations

import numpy as np
import pytest

from sar_preprocessor.normalize import normalize


# --------------------------------------------------------------------------
# Successful cases
# --------------------------------------------------------------------------


class TestNormalizeSuccess:
    def test_output_shape_and_dtype_2d(self, single_band_image: np.ndarray) -> None:
        result = normalize(single_band_image)
        assert result.shape == single_band_image.shape
        assert result.dtype == np.float32

    def test_output_shape_and_dtype_3d(self, multi_band_image: np.ndarray) -> None:
        result = normalize(multi_band_image)
        assert result.shape == multi_band_image.shape
        assert result.dtype == np.float32

    def test_output_values_bounded_percentile(self, single_band_image: np.ndarray) -> None:
        result = normalize(single_band_image, method="percentile", low=1.0, high=99.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_minmax_maps_extremes_to_0_and_1(self) -> None:
        image = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
        result = normalize(image, method="minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)
        # 0.0 is the array minimum -> should map to 0.0 exactly.
        assert result[0, 0] == pytest.approx(0.0)
        # 10.0 is the array maximum -> should map to 1.0 exactly.
        assert result[1, 0] == pytest.approx(1.0)

    def test_minmax_preserves_relative_order(self) -> None:
        image = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2)
        result = normalize(image, method="minmax")
        assert result[0, 0] < result[0, 1] < result[1, 0] < result[1, 1]

    def test_zscore_constant_image_maps_to_half(self) -> None:
        image = np.full((4, 4), 7.0, dtype=np.float32)
        result = normalize(image, method="zscore")
        # std == 0 is guarded to 1.0, so z == 0 everywhere -> (0+3)/6 == 0.5
        assert np.allclose(result, 0.5)

    def test_percentile_default_low_high(self, single_band_image: np.ndarray) -> None:
        # Should not raise, and should differ from a minmax normalization
        # for data with a genuine spread of outlier-like tails.
        image = single_band_image.copy()
        image[0, 0] = 1e6  # outlier
        pct_result = normalize(image, method="percentile", low=1.0, high=99.0)
        minmax_result = normalize(image, method="minmax")
        assert not np.allclose(pct_result, minmax_result)

    def test_complex_input_uses_magnitude(self) -> None:
        real = np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        imag = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        complex_image = (real + 1j * imag).astype(np.complex64)

        result_complex = normalize(complex_image, method="minmax")
        expected_magnitude = np.abs(complex_image).astype(np.float32)
        result_real = normalize(expected_magnitude, method="minmax")

        assert np.allclose(result_complex, result_real)
        # magnitude of (3+4j) is 5.0, which is the array max -> maps to 1.0
        assert result_complex[0, 0] == pytest.approx(1.0)

    def test_per_channel_true_normalizes_independently(self) -> None:
        image = np.stack(
            [
                np.array([[0.0, 10.0]], dtype=np.float32),
                np.array([[0.0, 1000.0]], dtype=np.float32),
            ],
            axis=0,
        )
        result = normalize(image, method="minmax", per_channel=True)
        # Both channels' max value should map to 1.0 independently.
        assert result[0, 0, 1] == pytest.approx(1.0)
        assert result[1, 0, 1] == pytest.approx(1.0)

    def test_per_channel_false_uses_global_stats(self) -> None:
        image = np.stack(
            [
                np.array([[0.0, 10.0]], dtype=np.float32),
                np.array([[0.0, 1000.0]], dtype=np.float32),
            ],
            axis=0,
        )
        result = normalize(image, method="minmax", per_channel=False)
        # Global max is 1000 (channel 1), so channel 0's max (10) should
        # NOT map to 1.0 when using shared statistics.
        assert result[0, 0, 1] < 1.0
        assert result[1, 0, 1] == pytest.approx(1.0)

    def test_per_channel_true_and_false_differ(self) -> None:
        image = np.stack(
            [
                np.array([[0.0, 10.0]], dtype=np.float32),
                np.array([[0.0, 1000.0]], dtype=np.float32),
            ],
            axis=0,
        )
        per_channel = normalize(image, method="minmax", per_channel=True)
        global_stats = normalize(image, method="minmax", per_channel=False)
        assert not np.allclose(per_channel, global_stats)

    def test_non_finite_values_mapped_to_zero(self) -> None:
        image = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
        result = normalize(image, method="minmax")
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert np.isfinite(result).all()


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


class TestNormalizeEdgeCases:
    def test_empty_2d_array_returns_empty_without_crash(self) -> None:
        image = np.zeros((0, 0), dtype=np.float32)
        result = normalize(image)
        assert result.shape == (0, 0)

    def test_entirely_non_finite_returns_zeros(self) -> None:
        image = np.full((4, 4), np.nan, dtype=np.float32)
        result = normalize(image)
        assert np.array_equal(result, np.zeros((4, 4), dtype=np.float32))

    def test_constant_image_percentile_returns_zeros(self) -> None:
        image = np.full((4, 4), 3.0, dtype=np.float32)
        result = normalize(image, method="percentile")
        assert np.array_equal(result, np.zeros((4, 4), dtype=np.float32))

    def test_single_pixel_image(self) -> None:
        image = np.array([[42.0]], dtype=np.float32)
        result = normalize(image, method="minmax")
        # Degenerate range (single value) -> zeros, no divide-by-zero crash.
        assert result.shape == (1, 1)
        assert np.isfinite(result).all()


# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------


class TestNormalizeFailures:
    def test_unknown_method_raises_value_error(self, single_band_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown normalize method"):
            normalize(single_band_image, method="not_a_real_method")  # type: ignore[arg-type]

    def test_unknown_method_raises_for_global_stats_path(
        self, multi_band_image: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="Unknown normalize method"):
            normalize(multi_band_image, method="bogus", per_channel=False)  # type: ignore[arg-type]

    def test_1d_array_raises_value_error(self) -> None:
        image = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            normalize(image)

    def test_4d_array_raises_value_error(self) -> None:
        image = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            normalize(image)
