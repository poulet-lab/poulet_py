import numpy as np
import pytest

from poulet_py.widefield.trace_metrics import TraceMetrics


@pytest.fixture
def simple_trace() -> np.ndarray:
    """Create a simple synthetic trace with a clear peak."""
    n_frames = 200
    trace = np.zeros(n_frames)
    onset_frame = 50
    peak_frame = 80
    decay_end = 150
    peak_value = 0.15

    for i in range(onset_frame, peak_frame):
        progress = (i - onset_frame) / (peak_frame - onset_frame)
        trace[i] = peak_value * progress

    trace[peak_frame] = peak_value

    for i in range(peak_frame + 1, decay_end):
        progress = (i - peak_frame) / (decay_end - peak_frame)
        trace[i] = peak_value * (1 - progress)

    return trace


@pytest.fixture
def weak_trace() -> np.ndarray:
    """Create a trace with peak below threshold."""
    n_frames = 200
    trace = np.zeros(n_frames)
    trace[80] = 0.02
    return trace


@pytest.fixture
def trace_params() -> dict:
    """Common trace parameters."""
    return {
        "fps": 20.0,
        "onset_frame": 50,
        "offset_frame": 120,
    }


class TestTraceMetricsInit:
    def test_init_valid(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        assert tm.trace is simple_trace
        assert tm.fps == trace_params["fps"]
        assert tm.onset_frame == trace_params["onset_frame"]
        assert tm.offset_frame == trace_params["offset_frame"]
        assert tm.peak_threshold == 0.05

    def test_init_custom_threshold(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
            peak_threshold=0.10,
        )
        assert tm.peak_threshold == 0.10

    def test_init_invalid_shape(self, trace_params: dict):
        trace_2d = np.zeros((10, 10))
        with pytest.raises(ValueError, match="Expected 1D trace"):
            TraceMetrics(
                trace=trace_2d,
                fps=trace_params["fps"],
                onset_frame=trace_params["onset_frame"],
                offset_frame=trace_params["offset_frame"],
            )


class TestCalculatePeak:
    def test_calculate_peak(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        peak_value, peak_frame = tm.calculate_peak()

        assert peak_value is not None
        assert peak_frame is not None
        assert peak_value == pytest.approx(0.15, abs=0.01)
        assert peak_frame == 80

    def test_calculate_peak_with_post_stimulus(
        self, simple_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=70,
        )
        peak_value, peak_frame = tm.calculate_peak(post_stimulus_ms=1000)

        assert peak_value is not None
        assert peak_frame == 80

    def test_calculate_peak_empty_window(self, trace_params: dict):
        trace = np.zeros(200)
        tm = TraceMetrics(
            trace=trace,
            fps=trace_params["fps"],
            onset_frame=250,
            offset_frame=300,
        )
        peak_value, peak_frame = tm.calculate_peak()
        assert peak_value is None
        assert peak_frame is None


class TestCalculateAllMetrics:
    def test_calculate_all_metrics_strong_signal(
        self, simple_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        metrics = tm.calculate_all_metrics()

        assert metrics["peak_value"] == pytest.approx(0.15, abs=0.01)
        assert metrics["peak_frame"] == 80
        assert metrics["above_threshold"] is True
        assert "latency" in metrics
        assert "rise_time" in metrics
        assert "half_width" in metrics
        assert "decay_time" in metrics
        assert "time_to_peak" in metrics
        assert "rise_slope" in metrics
        assert "decay_slope" in metrics

    def test_calculate_all_metrics_weak_signal(
        self, weak_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=weak_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        metrics = tm.calculate_all_metrics()

        assert metrics["above_threshold"] is False
        assert metrics["latency"] is None
        assert metrics["rise_time"] is None
        assert metrics["half_width"] is None
        assert metrics["decay_time"] is None

    def test_metrics_property(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        metrics = tm.metrics

        assert isinstance(metrics, dict)
        assert "peak_value" in metrics
        assert metrics["peak_value"] is not None


class TestRiseTime:
    def test_calculate_rise_time(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        tm.calculate_all_metrics()

        assert tm._rise_time is not None
        assert tm._rise_time > 0
        assert tm._frame_20_rise is not None
        assert tm._frame_80_rise is not None
        assert tm._frame_20_rise < tm._frame_80_rise


class TestHalfWidth:
    def test_calculate_half_width(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        tm.calculate_all_metrics()

        assert tm._half_width is not None
        assert tm._half_width > 0
        assert tm._frame_50_rise is not None
        assert tm._frame_50_fall is not None
        assert tm._frame_50_rise < tm._frame_50_fall


class TestDecayTime:
    def test_calculate_decay_time(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        tm.calculate_all_metrics()

        assert tm._decay_time is not None
        assert tm._decay_time > 0
        assert tm._frame_80_decay is not None
        assert tm._frame_20_decay is not None
        assert tm._frame_80_decay < tm._frame_20_decay


class TestSlopes:
    def test_calculate_rise_slope(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        tm.calculate_all_metrics()

        if tm._rise_slope is not None:
            assert tm._rise_slope > 0

    def test_calculate_decay_slope(self, simple_trace: np.ndarray, trace_params: dict):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        tm.calculate_all_metrics()

        if tm._decay_slope is not None:
            assert tm._decay_slope < 0


class TestThresholdCrossing:
    def test_find_threshold_crossing_rising(
        self, simple_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        frame, time = tm._find_threshold_crossing(
            threshold=0.05,
            start_frame=trace_params["onset_frame"],
            end_frame=80,
            direction="rising",
        )

        assert frame is not None
        assert time is not None
        assert frame > trace_params["onset_frame"]
        assert frame < 80

    def test_find_threshold_crossing_falling(
        self, simple_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        frame, time = tm._find_threshold_crossing(
            threshold=0.05,
            start_frame=80,
            end_frame=150,
            direction="falling",
        )

        assert frame is not None
        assert time is not None
        assert frame > 80
        assert frame < 150

    def test_find_threshold_crossing_invalid_window(
        self, simple_trace: np.ndarray, trace_params: dict
    ):
        tm = TraceMetrics(
            trace=simple_trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        frame, time = tm._find_threshold_crossing(
            threshold=0.05,
            start_frame=100,
            end_frame=50,
            direction="rising",
        )
        assert frame is None
        assert time is None


class TestEdgeCases:
    def test_short_trace(self, trace_params: dict):
        trace = np.array([0.0, 0.1, 0.15, 0.1, 0.0])
        tm = TraceMetrics(
            trace=trace,
            fps=trace_params["fps"],
            onset_frame=0,
            offset_frame=4,
        )
        metrics = tm.calculate_all_metrics()

        assert metrics["peak_value"] == pytest.approx(0.15, abs=0.01)
        assert metrics["above_threshold"] is True

    def test_flat_trace(self, trace_params: dict):
        trace = np.zeros(200)
        tm = TraceMetrics(
            trace=trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        metrics = tm.calculate_all_metrics()

        assert metrics["peak_value"] == 0.0
        assert metrics["above_threshold"] is False

    def test_negative_peak(self, trace_params: dict):
        trace = np.zeros(200)
        trace[80] = -0.15
        tm = TraceMetrics(
            trace=trace,
            fps=trace_params["fps"],
            onset_frame=trace_params["onset_frame"],
            offset_frame=trace_params["offset_frame"],
        )
        metrics = tm.calculate_all_metrics()

        assert metrics["peak_value"] == 0.0
        assert metrics["above_threshold"] is False

