from pathlib import Path

import numpy as np
import pytest

from poulet_py.widefield import WidefieldAnalysis
from poulet_py.widefield.io import (
    load_green_reference,
    load_imaging,
    load_sensors,
    load_timestamps,
    tiff_to_numpy,
)
from poulet_py.widefield.masks import (
    apply_circular_mask,
    load_mask_json,
    save_mask_json,
)
from poulet_py.widefield.metrics import (
    calculate_baseline_movie,
    calculate_deltaff_movie,
    calculate_percentile_movie,
)
from poulet_py.widefield.roi import centroid_from_percentile, trace_within_circular_roi


class TestWidefieldAnalysisInit:
    def test_init_valid_path(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        assert wf.trial_path == sample_trial_path
        assert wf.tiff_path == sample_trial_path / "recording.tiff"
        assert wf.csv_path == sample_trial_path / "recording.csv"
        assert wf.h5_path == sample_trial_path / "data.h5"
        assert wf.green_path == sample_trial_path / "green.tiff"

    def test_init_nonexistent_path(self, tmp_path: Path):
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            WidefieldAnalysis(nonexistent)

    def test_init_missing_tiff(self, tmp_path: Path):
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        with pytest.raises(ValueError, match="recording.tiff not found"):
            WidefieldAnalysis(trial_dir)


class TestWidefieldAnalysisLoadData:
    def test_load_data(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()

        assert wf.imaging_data is not None
        assert wf.imaging_data.ndim == 3
        assert wf.green_reference is not None
        assert wf.timestamps is not None
        assert len(wf.sensor_data) > 0

    def test_imaging_data_shape(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()

        assert wf.imaging_data.shape[0] > 0
        assert wf.imaging_data.shape[1] > 0
        assert wf.imaging_data.shape[2] > 0


class TestWidefieldAnalysisMetrics:
    @pytest.fixture
    def loaded_wf(self, sample_trial_path: Path) -> WidefieldAnalysis:
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()
        return wf

    def test_calculate_percentile(self, loaded_wf: WidefieldAnalysis):
        result = loaded_wf.calculate_percentile(percentile=15.0)

        assert result is not None
        assert result.ndim == 2
        assert result.shape == loaded_wf.imaging_data.shape[1:]

    def test_calculate_percentile_with_window(self, loaded_wf: WidefieldAnalysis):
        fps = loaded_wf.get_fps() or 10.0
        result = loaded_wf.calculate_percentile(
            percentile=15.0,
            stimulus_start_frame=50,
            baseline_ms=500.0,
            fps=fps,
        )

        assert result is not None
        assert result.ndim == 2

    def test_calculate_baseline(self, loaded_wf: WidefieldAnalysis):
        fps = loaded_wf.get_fps() or 10.0
        result = loaded_wf.calculate_baseline(
            stimulus_start_frame=50,
            baseline_ms=500.0,
            fps=fps,
        )

        assert result is not None
        assert result.ndim == 2
        assert result.shape == loaded_wf.imaging_data.shape[1:]

    def test_calculate_deltaff(self, loaded_wf: WidefieldAnalysis):
        baseline = loaded_wf.calculate_percentile(percentile=15.0)
        result = loaded_wf.calculate_deltaff(baseline=baseline)

        assert result is not None
        assert result.ndim == 3
        assert result.shape == loaded_wf.imaging_data.shape


class TestWidefieldAnalysisROI:
    @pytest.fixture
    def loaded_wf(self, sample_trial_path: Path) -> WidefieldAnalysis:
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()
        return wf

    def test_set_roi_tuple(self, loaded_wf: WidefieldAnalysis):
        loaded_wf.set_roi((100, 100))
        assert loaded_wf.roi is not None
        assert loaded_wf.roi["center"] == (100, 100)

    def test_set_roi_dict(self, loaded_wf: WidefieldAnalysis):
        loaded_wf.set_roi({"center": (50, 50)})
        assert loaded_wf.roi is not None
        assert loaded_wf.roi["center"] == (50, 50)

    def test_set_roi_invalid(self, loaded_wf: WidefieldAnalysis):
        with pytest.raises(ValueError):
            loaded_wf.set_roi((1, 2, 3))

        with pytest.raises(ValueError):
            loaded_wf.set_roi({"invalid": "data"})

    def test_calculate_percentile_centroid_roi(self, loaded_wf: WidefieldAnalysis):
        test_data = np.zeros((100, 100))
        test_data[40:60, 40:60] = 1.0

        roi_x, roi_y = loaded_wf.calculate_percentile_centroid_roi(test_data, percentile=95.0)

        assert 40 <= roi_x <= 60
        assert 40 <= roi_y <= 60

    def test_calculate_trace_within_roi(self, loaded_wf: WidefieldAnalysis):
        loaded_wf.set_roi((50, 50))
        trace = loaded_wf.calculate_trace_within_roi(diameter=20.0)

        assert trace is not None
        assert trace.ndim == 1
        assert len(trace) == loaded_wf.imaging_data.shape[0]


class TestWidefieldAnalysisMask:
    @pytest.fixture
    def loaded_wf(self, sample_trial_path: Path) -> WidefieldAnalysis:
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()
        return wf

    def test_apply_mask(self, loaded_wf: WidefieldAnalysis):
        mask_data = {
            "center_x": 50.0,
            "center_y": 50.0,
            "radius": 30.0,
        }
        result = loaded_wf.apply_mask(mask_data=mask_data)

        assert result is not None
        assert result.shape == loaded_wf.imaging_data.shape


class TestWidefieldAnalysisCondition:
    def test_set_condition(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        condition = {
            "temperature": 32.0,
            "duration": 2.0,
            "protocol": "test",
        }
        wf.set_condition(condition)

        assert wf.condition == condition
        assert wf.temperature == 32.0
        assert wf.duration == 2.0
        assert wf.protocol == "test"


class TestWidefieldAnalysisUtility:
    def test_get_fps(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()
        fps = wf.get_fps()
        assert fps is None or isinstance(fps, float)

    def test_to_numpy(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()

        result = wf.to_numpy()
        assert result is not None
        assert np.array_equal(result, wf.imaging_data)

        arr = np.zeros((10, 10, 10))
        result = wf.to_numpy(arr)
        assert np.array_equal(result, arr)

    def test_close(self, sample_trial_path: Path):
        wf = WidefieldAnalysis(sample_trial_path)
        wf.load_data()
        assert wf.imaging_data is not None

        wf.close()
        assert wf.imaging_data is None
        assert wf.green_reference is None
        assert wf.timestamps is None
        assert wf.sensor_data == {}


class TestIOModule:
    def test_load_imaging(self, sample_trial_path: Path):
        tiff_path = sample_trial_path / "recording.tiff"
        data = load_imaging(tiff_path)
        assert data is not None
        assert data.ndim == 3

    def test_load_green_reference(self, sample_trial_path: Path):
        green_path = sample_trial_path / "green.tiff"
        data = load_green_reference(green_path)
        assert data is not None
        assert data.ndim == 2

    def test_load_timestamps(self, sample_trial_path: Path):
        csv_path = sample_trial_path / "recording.csv"
        df = load_timestamps(csv_path)
        assert df is not None
        assert len(df) > 0

    def test_load_sensors(self, sample_trial_path: Path):
        h5_path = sample_trial_path / "data.h5"
        sensor_data, sensor_attrs, file_attrs = load_sensors(h5_path)
        assert isinstance(sensor_data, dict)
        assert isinstance(sensor_attrs, dict)
        assert isinstance(file_attrs, dict)

    def test_tiff_to_numpy_array(self):
        arr = np.zeros((10, 10, 10))
        result = tiff_to_numpy(arr)
        assert np.array_equal(result, arr)

    def test_tiff_to_numpy_none(self):
        result = tiff_to_numpy(None)
        assert result is None


class TestMetricsModule:
    def test_calculate_percentile_movie(self):
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_percentile_movie(data, percentile=15.0)

        assert result is not None
        assert result.shape == (50, 50)

    def test_calculate_percentile_movie_with_window(self):
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_percentile_movie(
            data,
            percentile=15.0,
            stimulus_start_frame=50,
            baseline_ms=500.0,
            fps=10.0,
        )

        assert result is not None
        assert result.shape == (50, 50)

    def test_calculate_percentile_movie_invalid_shape(self):
        data = np.random.rand(50, 50).astype(np.float32)
        result = calculate_percentile_movie(data, percentile=15.0)
        assert result is None

    def test_calculate_deltaff_movie(self):
        data = np.random.rand(100, 50, 50).astype(np.float32) + 1
        baseline = np.ones((50, 50), dtype=np.float32)
        result = calculate_deltaff_movie(data, baseline)

        assert result is not None
        assert result.shape == data.shape

    def test_calculate_deltaff_movie_shape_mismatch(self):
        data = np.random.rand(100, 50, 50).astype(np.float32)
        baseline = np.ones((40, 40), dtype=np.float32)
        result = calculate_deltaff_movie(data, baseline)
        assert result is None

    def test_calculate_baseline_movie(self):
        data = np.random.rand(100, 50, 50).astype(np.float32)
        result = calculate_baseline_movie(
            data,
            stimulus_start_frame=50,
            baseline_ms=500.0,
            fps=10.0,
        )

        assert result is not None
        assert result.shape == (50, 50)


class TestROIModule:
    def test_centroid_from_percentile(self):
        data = np.zeros((100, 100))
        data[40:60, 40:60] = 1.0

        roi_x, roi_y = centroid_from_percentile(data, percentile=95.0)

        assert 40 <= roi_x <= 60
        assert 40 <= roi_y <= 60

    def test_centroid_from_percentile_invalid_shape(self):
        data = np.zeros((100, 100, 100))
        with pytest.raises(ValueError):
            centroid_from_percentile(data)

    def test_trace_within_circular_roi(self):
        data = np.random.rand(100, 50, 50).astype(np.float32)
        center = (25, 25)
        trace = trace_within_circular_roi(data, center, diameter=10.0)

        assert trace.ndim == 1
        assert len(trace) == 100

    def test_trace_within_circular_roi_invalid_shape(self):
        data = np.random.rand(50, 50).astype(np.float32)
        with pytest.raises(ValueError):
            trace_within_circular_roi(data, (25, 25))


class TestMasksModule:
    def test_apply_circular_mask(self):
        data = np.ones((100, 50, 50), dtype=np.float32)
        mask_data = {
            "center_x": 25.0,
            "center_y": 25.0,
            "radius": 10.0,
        }
        result = apply_circular_mask(data, mask_data)

        assert result is not None
        assert result.shape == data.shape
        assert result[:, 0, 0].sum() == 0

    def test_save_and_load_mask_json(self, tmp_path: Path):
        mask_data = {
            "center_x": 100.0,
            "center_y": 100.0,
            "radius": 50.0,
        }
        output_path = tmp_path / "test_mask.json"

        saved_path = save_mask_json(mask_data, output_path)
        assert saved_path is not None
        assert saved_path.exists()

        loaded = load_mask_json(saved_path)
        assert loaded is not None
        assert loaded["center_x"] == mask_data["center_x"]
        assert loaded["center_y"] == mask_data["center_y"]
        assert loaded["radius"] == mask_data["radius"]

    def test_load_mask_json_nonexistent(self, tmp_path: Path):
        result = load_mask_json(tmp_path / "nonexistent.json")
        assert result is None
