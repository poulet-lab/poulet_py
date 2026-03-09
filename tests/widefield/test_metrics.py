"""Tests for widefield metrics module."""

import numpy as np
import pytest

from poulet_py.widefield.metrics import (
    calculate_baseline_movie,
    calculate_deltaff_movie,
    calculate_percentile_movie,
    calculate_spatial_threshold_metrics,
)


class TestCalculateSpatialThresholdMetrics:
    """Tests for calculate_spatial_threshold_metrics function."""

    @pytest.fixture
    def simple_image(self) -> np.ndarray:
        """Create a simple 2D test image with a bright center."""
        image = np.zeros((100, 100), dtype=np.float32)
        image[40:60, 40:60] = 1.0
        return image

    @pytest.fixture
    def gradient_image(self) -> np.ndarray:
        """Create a gradient test image."""
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        return (xx + yy) / 2

    def test_basic_functionality(self, simple_image: np.ndarray):
        """Test basic function call returns expected fields (structured array)."""
        result = calculate_spatial_threshold_metrics(simple_image)

        assert result is not None
        expected_fields = (
            "threshold",
            "threshold_mask",
            "n_pixels_above",
            "total_pixels",
            "percent_above",
            "auc_above_threshold",
            "smoothed_image",
            "max_activity",
            "min_activity",
            "mean_activity",
            "std_activity",
        )
        assert result.dtype.names == expected_fields

    def test_threshold_calculation(self, simple_image: np.ndarray):
        """Test threshold is correctly calculated as percentage of max."""
        result = calculate_spatial_threshold_metrics(
            simple_image, threshold_percent=50.0, smoothing_sigma=None
        )

        assert result["max_activity"] == pytest.approx(1.0)
        assert result["threshold"] == pytest.approx(0.5)

    def test_no_smoothing(self, simple_image: np.ndarray):
        """Test that smoothing_sigma=None preserves original image."""
        result = calculate_spatial_threshold_metrics(simple_image, smoothing_sigma=None)

        assert np.array_equal(result["smoothed_image"], simple_image)

    def test_with_smoothing(self, simple_image: np.ndarray):
        """Test that smoothing changes the image."""
        result = calculate_spatial_threshold_metrics(simple_image, smoothing_sigma=5.0)

        assert not np.array_equal(result["smoothed_image"], simple_image)
        assert result["max_activity"] < 1.0

    def test_pixel_counts(self, simple_image: np.ndarray):
        """Test pixel counting is correct."""
        result = calculate_spatial_threshold_metrics(
            simple_image, threshold_percent=85.0, smoothing_sigma=None
        )

        assert result["total_pixels"] == 100 * 100

        expected_percent = 100.0 * result["n_pixels_above"] / result["total_pixels"]
        assert result["percent_above"] == pytest.approx(expected_percent)

    def test_auc_calculation(self):
        """Test AUC is sum of values above threshold."""
        image = np.array([[0.0, 0.5], [0.8, 1.0]], dtype=np.float32)

        result = calculate_spatial_threshold_metrics(
            image, threshold_percent=50.0, smoothing_sigma=None
        )

        expected_auc = 0.5 + 0.8 + 1.0
        assert result["auc_above_threshold"] == pytest.approx(expected_auc)
        assert result["n_pixels_above"] == 3

    def test_threshold_mask_is_boolean(self, simple_image: np.ndarray):
        """Test that threshold_mask is a boolean array."""
        result = calculate_spatial_threshold_metrics(simple_image)

        assert result["threshold_mask"].dtype == bool
        assert result["threshold_mask"].shape == simple_image.shape

    def test_different_threshold_percentages(self, gradient_image: np.ndarray):
        """Test that higher threshold percentage gives fewer pixels above."""
        result_low = calculate_spatial_threshold_metrics(
            gradient_image, threshold_percent=25.0, smoothing_sigma=None
        )
        result_high = calculate_spatial_threshold_metrics(
            gradient_image, threshold_percent=75.0, smoothing_sigma=None
        )

        assert result_high["n_pixels_above"] < result_low["n_pixels_above"]
        assert result_high["percent_above"] < result_low["percent_above"]

    def test_invalid_input_1d(self):
        """Test that 1D array returns None."""
        result = calculate_spatial_threshold_metrics(np.array([1, 2, 3]))
        assert result is None

    def test_invalid_input_3d(self):
        """Test that 3D array returns None."""
        result = calculate_spatial_threshold_metrics(np.zeros((10, 10, 10)))
        assert result is None

    def test_empty_image(self):
        """Test that empty image returns None."""
        result = calculate_spatial_threshold_metrics(np.array([]).reshape(0, 0))
        assert result is None

    def test_uniform_image(self):
        """Test handling of uniform (constant) image."""
        image = np.ones((50, 50)) * 0.5

        result = calculate_spatial_threshold_metrics(
            image, threshold_percent=85.0, smoothing_sigma=None
        )

        assert result is not None
        assert result["n_pixels_above"] == result["total_pixels"]
        assert result["percent_above"] == pytest.approx(100.0)

    def test_all_zeros(self):
        """Test handling of all-zero image."""
        image = np.zeros((50, 50))

        result = calculate_spatial_threshold_metrics(image, smoothing_sigma=None)

        assert result is not None
        assert result["max_activity"] == 0.0
        assert result["threshold"] == 0.0

    def test_negative_values(self):
        """Test handling of image with negative values."""
        image = np.array([[-1.0, -0.5], [0.0, 0.5]], dtype=np.float32)

        result = calculate_spatial_threshold_metrics(
            image, threshold_percent=50.0, smoothing_sigma=None
        )

        assert result is not None
        assert result["min_activity"] == pytest.approx(-1.0)
        assert result["max_activity"] == pytest.approx(0.5)

    def test_statistics_correctness(self, simple_image: np.ndarray):
        """Test that statistics are calculated correctly."""
        result = calculate_spatial_threshold_metrics(simple_image, smoothing_sigma=None)

        expected_min = float(np.min(simple_image))
        expected_max = float(np.max(simple_image))
        expected_mean = float(np.mean(simple_image))
        expected_std = float(np.std(simple_image))

        assert result["min_activity"] == pytest.approx(expected_min)
        assert result["max_activity"] == pytest.approx(expected_max)
        assert result["mean_activity"] == pytest.approx(expected_mean)
        assert result["std_activity"] == pytest.approx(expected_std)


class TestCalculatePercentileMovie:
    """Tests for calculate_percentile_movie function."""

    def test_basic_percentile(self):
        """Test basic percentile calculation."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_percentile_movie(data, percentile=15.0)

        assert result is not None
        assert result.shape == (50, 50)

    def test_invalid_input(self):
        """Test that 2D input returns None."""
        result = calculate_percentile_movie(np.zeros((50, 50)))
        assert result is None

    def test_from_ms_to_ms_window(self):
        """Test time window via from_ms and to_ms (ms)."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        # 0–500 ms at 10 fps = frames 0–5 (5 frames)
        result = calculate_percentile_movie(
            data, percentile=15.0, from_ms=0.0, to_ms=500.0, fps=10.0
        )
        assert result is not None
        assert result.shape == (50, 50)

    def test_multiple_percentiles_returns_3d(self):
        """Test list of percentiles returns (n_percentiles, H, W)."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_percentile_movie(data, percentile=[15.0, 50.0, 85.0])
        assert result is not None
        assert result.shape == (3, 50, 50)
        assert result[0].shape == (50, 50)
        assert result[1].shape == (50, 50)
        assert result[2].shape == (50, 50)

    def test_from_ms_to_ms_only_one_returns_none(self):
        """Test that only from_ms or only to_ms returns None."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        assert calculate_percentile_movie(data, from_ms=0.0, to_ms=None, fps=10.0) is None
        assert calculate_percentile_movie(data, from_ms=None, to_ms=500.0, fps=10.0) is None

    def test_from_ms_ge_to_ms_returns_none(self):
        """Test that from_ms >= to_ms returns None."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        assert (
            calculate_percentile_movie(data, percentile=15.0, from_ms=500.0, to_ms=200.0, fps=10.0)
            is None
        )
        assert (
            calculate_percentile_movie(data, percentile=15.0, from_ms=300.0, to_ms=300.0, fps=10.0)
            is None
        )

    def test_empty_percentile_list_returns_none(self):
        """Test that empty list for percentile returns None."""
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_percentile_movie(data, percentile=[])
        assert result is None


class TestCalculateDeltaffMovie:
    """Tests for calculate_deltaff_movie function."""

    def test_basic_deltaff(self):
        """Test basic dF/F calculation."""
        data = np.ones((100, 50, 50)) * 2.0
        baseline = np.ones((50, 50))

        result = calculate_deltaff_movie(data, baseline)

        assert result is not None
        assert result.shape == (100, 50, 50)
        assert np.allclose(result, 1.0)

    def test_invalid_data(self):
        """Test that 2D data returns None."""
        result = calculate_deltaff_movie(np.zeros((50, 50)), np.zeros((50, 50)))
        assert result is None

    def test_invalid_baseline(self):
        """Test that 3D baseline returns None."""
        result = calculate_deltaff_movie(np.zeros((100, 50, 50)), np.zeros((100, 50, 50)))
        assert result is None


class TestCalculateBaselineMovie:
    """Tests for calculate_baseline_movie function."""

    def test_basic_baseline(self):
        """Test basic baseline calculation."""
        data = np.ones((200, 50, 50))
        result = calculate_baseline_movie(
            data, stimulus_start_frame=100, baseline_ms=500.0, fps=20.0
        )

        assert result is not None
        assert result.shape == (50, 50)

    def test_invalid_input(self):
        """Test that 2D input returns None."""
        result = calculate_baseline_movie(
            np.zeros((50, 50)), stimulus_start_frame=10, baseline_ms=500.0, fps=20.0
        )
        assert result is None
