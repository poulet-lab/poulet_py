"""
Trace metrics analysis module for widefield imaging data.

Provides the TraceMetrics class for analyzing 1D dF/F traces,
calculating response kinetics (peak, latency, rise time, half width,
decay time, slopes), and generating visualizations.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from poulet_py import LOGGER


class TraceMetrics:
    """
    Analyzer for 1D dF/F trace kinetics.

    Calculates response metrics including peak value, latency, rise time,
    half width, decay time, and slopes from a single trace or mean trace.

    Attributes:
        trace: 1D numpy array of dF/F values.
        fps: Frames per second.
        onset_frame: Stimulus onset frame index.
        offset_frame: Stimulus offset frame index.
        peak_threshold: Minimum peak value for metric calculation (default 0.05).
    """

    peak_threshold: float = 0.05

    def __init__(
        self,
        trace: np.ndarray,
        fps: float,
        onset_frame: int,
        offset_frame: int,
        peak_threshold: float | None = None,
    ) -> None:
        """
        Initialize TraceMetrics with trace data and timing parameters.

        Args:
            trace: 1D numpy array of dF/F values.
            fps: Frames per second.
            onset_frame: Stimulus onset frame index.
            offset_frame: Stimulus offset frame index.
            peak_threshold: Minimum peak value for metric calculation.
                           If None, uses class default (0.05).
        """
        if trace.ndim != 1:
            msg = f"Expected 1D trace array, got shape {trace.shape}"
            raise ValueError(msg)

        self.trace = trace
        self.fps = fps
        self.onset_frame = onset_frame
        self.offset_frame = offset_frame

        if peak_threshold is not None:
            self.peak_threshold = peak_threshold

        self._peak_value: float | None = None
        self._peak_frame: int | None = None
        self._latency: float | None = None
        self._response_onset_frame: int | None = None
        self._value_at_latency: float | None = None
        self._response_amplitude: float | None = None
        self._rise_time: float | None = None
        self._frame_20_rise: int | None = None
        self._frame_80_rise: int | None = None
        self._time_20_rise: float | None = None
        self._time_80_rise: float | None = None
        self._half_width: float | None = None
        self._frame_50_rise: int | None = None
        self._frame_50_fall: int | None = None
        self._time_50_rise: float | None = None
        self._time_50_fall: float | None = None
        self._decay_time: float | None = None
        self._frame_80_decay: int | None = None
        self._frame_20_decay: int | None = None
        self._time_80_decay: float | None = None
        self._time_20_decay: float | None = None
        self._value_80_decay: float | None = None
        self._value_20_decay: float | None = None
        self._threshold_80: float | None = None
        self._threshold_20: float | None = None
        self._time_to_peak: float | None = None
        self._rise_slope: float | None = None
        self._decay_slope: float | None = None
        self._above_threshold: bool = False
        self._metrics_calculated: bool = False

    def calculate_peak(
        self,
        post_stimulus_ms: float | None = None,
    ) -> tuple[float | None, int | None]:
        """
        Calculate peak value and frame index within stimulus window.

        Args:
            post_stimulus_ms: Optional milliseconds after offset to extend search.
                             If None, searches within stimulation window only.

        Returns:
            Tuple of (peak_value, peak_frame_index) or (None, None) if invalid.
        """
        if self.onset_frame is None or self.offset_frame is None:
            return None, None

        if len(self.trace) <= self.onset_frame:
            return None, None

        window_start = max(0, min(self.onset_frame, len(self.trace) - 1))

        if post_stimulus_ms is not None:
            post_stimulus_frames = int(post_stimulus_ms / 1000.0 * self.fps)
            window_end = min(self.offset_frame + post_stimulus_frames, len(self.trace))
        else:
            window_end = max(window_start + 1, min(self.offset_frame, len(self.trace)))

        if window_end <= window_start:
            return None, None

        trace_window = self.trace[window_start:window_end]
        peak_idx_in_window = int(np.argmax(trace_window))
        peak_frame = window_start + peak_idx_in_window
        peak_value = float(self.trace[peak_frame])

        return peak_value, peak_frame

    def calculate_latency(self) -> tuple[float | None, int | None]:
        """
        Calculate latency as time between stimulus onset and response onset.

        Response onset is defined as peak of second derivative in time window
        from stimulus start to peak of first derivative. Only calculated for
        peak values above peak_threshold.

        Returns:
            Tuple of (latency in seconds, response_onset_frame) or (None, None).
        """
        if self._peak_value is None or self._peak_value < self.peak_threshold:
            return None, None

        if self.onset_frame is None or len(self.trace) <= self.onset_frame + 2:
            return None, None

        stim_start = max(0, min(self.onset_frame, len(self.trace) - 3))

        if self.offset_frame is not None:
            stim_end = min(self.offset_frame, len(self.trace))
        else:
            stim_end = len(self.trace)

        if stim_end <= stim_start + 2:
            return None, None

        trace_segment = self.trace[stim_start:stim_end]
        if len(trace_segment) < 3:
            return None, None

        dt = 1.0 / self.fps
        first_derivative = np.gradient(trace_segment, dt)
        second_derivative = np.gradient(first_derivative, dt)

        peak_first_derivative_idx = int(np.argmax(first_derivative))

        negative_indices = np.where(
            (first_derivative < 0) & (np.arange(len(first_derivative)) < peak_first_derivative_idx)
        )[0]
        if len(negative_indices) > 0:
            window_start = max(0, int(negative_indices[-1]))
        else:
            window_start = 0

        window_end = peak_first_derivative_idx

        if window_end <= window_start:
            return None, None

        window_second_derivative = second_derivative[window_start:window_end]
        if len(window_second_derivative) == 0:
            return None, None

        peak_second_derivative_idx = int(np.argmax(window_second_derivative))
        absolute_second_derivative_idx = window_start + peak_second_derivative_idx
        response_onset_frame = stim_start + absolute_second_derivative_idx

        latency = (response_onset_frame - self.onset_frame) / self.fps

        return latency, response_onset_frame

    def _find_threshold_crossing(
        self,
        threshold: float,
        start_frame: int,
        end_frame: int,
        direction: str = "rising",
    ) -> tuple[int | None, float | None]:
        """
        Find frame and interpolated time where trace crosses threshold.

        Args:
            threshold: Threshold value to find.
            start_frame: Start frame for search.
            end_frame: End frame for search.
            direction: 'rising' or 'falling'.

        Returns:
            Tuple of (frame_index, crossing_time_in_seconds) or (None, None).
        """
        if start_frame >= end_frame or start_frame < 0 or end_frame > len(self.trace):
            return None, None

        segment = self.trace[start_frame:end_frame]

        if direction == "rising":
            below_threshold = segment < threshold
            above_threshold = segment >= threshold
            for i in range(len(segment) - 1):
                if below_threshold[i] and above_threshold[i + 1]:
                    frame_before = start_frame + i
                    frame_after = start_frame + i + 1
                    value_before = self.trace[frame_before]
                    value_after = self.trace[frame_after]

                    if abs(value_after - value_before) < 1e-10:
                        crossing_time = frame_after / self.fps
                    else:
                        fraction = (threshold - value_before) / (value_after - value_before)
                        crossing_time = (frame_before + fraction) / self.fps

                    return frame_after, crossing_time
        else:
            above_threshold = segment >= threshold
            below_threshold = segment < threshold
            for i in range(len(segment) - 1):
                if above_threshold[i] and below_threshold[i + 1]:
                    frame_before = start_frame + i
                    frame_after = start_frame + i + 1
                    value_before = self.trace[frame_before]
                    value_after = self.trace[frame_after]

                    if abs(value_after - value_before) < 1e-10:
                        crossing_time = frame_after / self.fps
                    else:
                        fraction = (threshold - value_before) / (value_after - value_before)
                        crossing_time = (frame_before + fraction) / self.fps

                    return frame_after, crossing_time

        if direction == "rising":
            if segment[0] >= threshold:
                return start_frame, start_frame / self.fps
        elif segment[0] < threshold:
            return start_frame, start_frame / self.fps

        return None, None

    def calculate_rise_time(
        self,
    ) -> tuple[float | None, int | None, int | None, float | None, float | None]:
        """
        Calculate rise time as time between 20% and 80% of response amplitude.

        Returns:
            Tuple of (rise_time, frame_20%, frame_80%, time_20%, time_80%)
            or (None, None, None, None, None) if invalid.
        """
        if self._peak_frame is None or self._peak_value is None:
            return None, None, None, None, None

        if self._peak_frame <= self.onset_frame:
            return None, None, None, None, None

        if self._value_at_latency is None or self._response_amplitude is None:
            threshold_20 = self._peak_value * 0.20
            threshold_80 = self._peak_value * 0.80
        else:
            threshold_20 = self._value_at_latency + self._response_amplitude * 0.20
            threshold_80 = self._value_at_latency + self._response_amplitude * 0.80

        frame_20, time_20 = self._find_threshold_crossing(
            threshold_20, self.onset_frame, self._peak_frame, direction="rising"
        )
        frame_80, time_80 = self._find_threshold_crossing(
            threshold_80, self.onset_frame, self._peak_frame, direction="rising"
        )

        if (
            frame_20 is None
            or frame_80 is None
            or time_20 is None
            or time_80 is None
            or time_80 <= time_20
        ):
            return None, None, None, None, None

        rise_time = time_80 - time_20

        return rise_time, frame_20, frame_80, time_20, time_80

    def calculate_half_width(
        self,
    ) -> tuple[float | None, int | None, int | None, float | None, float | None]:
        """
        Calculate half width at 50% of response amplitude.

        Returns:
            Tuple of (half_width, frame_rising, frame_falling, time_rising, time_falling)
            or (None, None, None, None, None) if invalid.
        """
        if self._peak_frame is None or self._peak_value is None:
            return None, None, None, None, None

        if self._value_at_latency is None or self._response_amplitude is None:
            threshold_50 = self._peak_value * 0.50
        else:
            threshold_50 = self._value_at_latency + self._response_amplitude * 0.50

        rising_start = max(0, self.onset_frame) if self.onset_frame is not None else 0
        if rising_start >= self._peak_frame:
            return None, None, None, None, None

        rising_frame, time_rising = self._find_threshold_crossing(
            threshold_50, rising_start, self._peak_frame, direction="rising"
        )

        falling_direction = "falling" if self._peak_value >= 0 else "rising"
        falling_frame, time_falling = self._find_threshold_crossing(
            threshold_50, self._peak_frame, len(self.trace), direction=falling_direction
        )

        if (
            rising_frame is None
            or falling_frame is None
            or time_rising is None
            or time_falling is None
        ):
            return None, None, None, None, None

        if time_falling <= time_rising:
            return None, None, None, None, None

        half_width = time_falling - time_rising

        return half_width, rising_frame, falling_frame, time_rising, time_falling

    def calculate_decay_time(
        self,
    ) -> tuple[
        float | None,
        int | None,
        int | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        """
        Calculate decay time as time between 80% and 20% on decay phase.

        Returns:
            Tuple of (decay_time, frame_80%, frame_20%, time_80%, time_20%,
            value_80_decay, value_20_decay, threshold_80, threshold_20)
            or all None if invalid.
        """
        if self._peak_frame is None or self._peak_value is None or self.onset_frame is None:
            return None, None, None, None, None, None, None, None, None

        if self._peak_frame >= len(self.trace) - 1 or self._peak_frame <= self.onset_frame:
            return None, None, None, None, None, None, None, None, None

        if self._value_at_latency is None or self._response_amplitude is None:
            threshold_80 = self._peak_value * 0.80
            threshold_20 = self._peak_value * 0.20
        else:
            threshold_80 = self._value_at_latency + self._response_amplitude * 0.80
            threshold_20 = self._value_at_latency + self._response_amplitude * 0.20

        decay_segment_start = self._peak_frame + 1
        decay_segment_end = len(self.trace)

        if decay_segment_end <= decay_segment_start:
            return None, None, None, None, None, None, None, None, None

        frame_80 = None
        frame_20 = None
        time_80 = None
        time_20 = None
        value_80_decay = None
        value_20_decay = None

        if self._peak_value > 0:
            for i in range(decay_segment_start, decay_segment_end - 1):
                if frame_80 is None:
                    if self.trace[i] >= threshold_80 and self.trace[i + 1] < threshold_80:
                        frame_before = i
                        frame_after = i + 1
                        value_before = self.trace[frame_before]
                        value_after = self.trace[frame_after]

                        if abs(value_after - value_before) < 1e-10:
                            time_80 = frame_after / self.fps
                        else:
                            fraction = (threshold_80 - value_before) / (value_after - value_before)
                            time_80 = (frame_before + fraction) / self.fps
                        value_80_decay = threshold_80
                        frame_80 = frame_after

                if frame_80 is not None and frame_20 is None:
                    if self.trace[i] >= threshold_20 and self.trace[i + 1] < threshold_20:
                        frame_before = i
                        frame_after = i + 1
                        value_before = self.trace[frame_before]
                        value_after = self.trace[frame_after]

                        if abs(value_after - value_before) < 1e-10:
                            time_20 = frame_after / self.fps
                        else:
                            fraction = (threshold_20 - value_before) / (value_after - value_before)
                            time_20 = (frame_before + fraction) / self.fps
                        value_20_decay = threshold_20
                        frame_20 = frame_after
                        break
        else:
            for i in range(decay_segment_start, decay_segment_end - 1):
                if frame_80 is None:
                    if self.trace[i] <= threshold_80 and self.trace[i + 1] > threshold_80:
                        frame_before = i
                        frame_after = i + 1
                        value_before = self.trace[frame_before]
                        value_after = self.trace[frame_after]

                        if abs(value_after - value_before) < 1e-10:
                            time_80 = frame_after / self.fps
                        else:
                            fraction = (threshold_80 - value_before) / (value_after - value_before)
                            time_80 = (frame_before + fraction) / self.fps
                        value_80_decay = threshold_80
                        frame_80 = frame_after

                if frame_80 is not None and frame_20 is None:
                    if self.trace[i] <= threshold_20 and self.trace[i + 1] > threshold_20:
                        frame_before = i
                        frame_after = i + 1
                        value_before = self.trace[frame_before]
                        value_after = self.trace[frame_after]

                        if abs(value_after - value_before) < 1e-10:
                            time_20 = frame_after / self.fps
                        else:
                            fraction = (threshold_20 - value_before) / (value_after - value_before)
                            time_20 = (frame_before + fraction) / self.fps
                        value_20_decay = threshold_20
                        frame_20 = frame_after
                        break

        if (
            frame_80 is None
            or frame_20 is None
            or time_80 is None
            or time_20 is None
            or value_80_decay is None
            or value_20_decay is None
            or time_20 <= time_80
        ):
            return None, None, None, None, None, None, None, threshold_80, threshold_20

        decay_time = time_20 - time_80

        return (
            decay_time,
            frame_80,
            frame_20,
            time_80,
            time_20,
            value_80_decay,
            value_20_decay,
            threshold_80,
            threshold_20,
        )

    def calculate_rise_slope(self) -> float | None:
        """
        Calculate slope between 20% and 80% points on rising phase.

        Returns:
            Slope in ΔF/F per second or None if invalid.
        """
        if self._frame_20_rise is None or self._frame_80_rise is None:
            return None

        if (
            self._frame_20_rise >= self._frame_80_rise
            or self._frame_20_rise < 0
            or self._frame_80_rise >= len(self.trace)
        ):
            return None

        value_20 = self.trace[self._frame_20_rise]
        value_80 = self.trace[self._frame_80_rise]

        slope_per_frame = (value_80 - value_20) / (self._frame_80_rise - self._frame_20_rise)
        slope_per_second = slope_per_frame * self.fps

        return float(slope_per_second)

    def calculate_decay_slope(self) -> float | None:
        """
        Calculate slope between 80% and 20% points on decay phase.

        Returns:
            Slope in ΔF/F per second (negative for decay) or None if invalid.
        """
        if self._frame_80_decay is None or self._frame_20_decay is None:
            return None

        if (
            self._frame_80_decay >= self._frame_20_decay
            or self._frame_80_decay < 0
            or self._frame_20_decay >= len(self.trace)
        ):
            return None

        value_80 = self.trace[self._frame_80_decay]
        value_20 = self.trace[self._frame_20_decay]

        slope_per_frame = (value_20 - value_80) / (self._frame_20_decay - self._frame_80_decay)
        slope_per_second = slope_per_frame * self.fps

        return float(slope_per_second)

    def calculate_time_to_peak(self) -> float | None:
        """
        Calculate time to peak from response onset.

        Returns:
            Time to peak in seconds or None if invalid.
        """
        if self._peak_frame is None or self._response_onset_frame is None:
            return None

        if self._peak_frame <= self._response_onset_frame:
            return None

        return (self._peak_frame - self._response_onset_frame) / self.fps

    def calculate_all_metrics(
        self,
        post_stimulus_ms: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate all trace metrics.

        Args:
            post_stimulus_ms: Optional milliseconds after offset to extend
                             peak search window.

        Returns:
            Dictionary containing all calculated metrics.
        """
        self._peak_value, self._peak_frame = self.calculate_peak(post_stimulus_ms)

        if self._peak_value is None:
            self._above_threshold = False
        elif self._peak_value < self.peak_threshold:
            self._above_threshold = False
        else:
            self._above_threshold = True

        if not self._above_threshold:
            self._metrics_calculated = True
            return self._build_metrics_dict()

        self._latency, self._response_onset_frame = self.calculate_latency()

        if self._response_onset_frame is not None:
            self._value_at_latency = float(self.trace[self._response_onset_frame])
            if self._peak_value is not None:
                self._response_amplitude = self._peak_value - self._value_at_latency

        (
            self._rise_time,
            self._frame_20_rise,
            self._frame_80_rise,
            self._time_20_rise,
            self._time_80_rise,
        ) = self.calculate_rise_time()

        (
            self._half_width,
            self._frame_50_rise,
            self._frame_50_fall,
            self._time_50_rise,
            self._time_50_fall,
        ) = self.calculate_half_width()

        (
            self._decay_time,
            self._frame_80_decay,
            self._frame_20_decay,
            self._time_80_decay,
            self._time_20_decay,
            self._value_80_decay,
            self._value_20_decay,
            self._threshold_80,
            self._threshold_20,
        ) = self.calculate_decay_time()

        self._time_to_peak = self.calculate_time_to_peak()

        if self._rise_time is not None:
            self._rise_slope = self.calculate_rise_slope()

        if self._decay_time is not None:
            self._decay_slope = self.calculate_decay_slope()

        self._metrics_calculated = True
        return self._build_metrics_dict()

    def _build_metrics_dict(self) -> dict[str, Any]:
        """Build dictionary of all metrics."""
        return {
            "peak_value": self._peak_value,
            "peak_frame": self._peak_frame,
            "above_threshold": self._above_threshold,
            "latency": self._latency,
            "response_onset_frame": self._response_onset_frame,
            "value_at_latency": self._value_at_latency,
            "response_amplitude": self._response_amplitude,
            "rise_time": self._rise_time,
            "frame_20_rise": self._frame_20_rise,
            "frame_80_rise": self._frame_80_rise,
            "time_20_rise": self._time_20_rise,
            "time_80_rise": self._time_80_rise,
            "half_width": self._half_width,
            "frame_50_rise": self._frame_50_rise,
            "frame_50_fall": self._frame_50_fall,
            "time_50_rise": self._time_50_rise,
            "time_50_fall": self._time_50_fall,
            "decay_time": self._decay_time,
            "frame_80_decay": self._frame_80_decay,
            "frame_20_decay": self._frame_20_decay,
            "time_80_decay": self._time_80_decay,
            "time_20_decay": self._time_20_decay,
            "time_to_peak": self._time_to_peak,
            "rise_slope": self._rise_slope,
            "decay_slope": self._decay_slope,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        """Get metrics dictionary. Calculates if not already done."""
        if not self._metrics_calculated:
            self.calculate_all_metrics()
        return self._build_metrics_dict()

    def plot(
        self,
        output_path: Path | None = None,
        title: str | None = None,
        y_limits: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot trace with metric annotations.

        Args:
            output_path: Optional path to save the plot.
            title: Optional title for the plot.
            y_limits: Optional (y_min, y_max) tuple for y-axis limits.
        """
        if not self._metrics_calculated:
            self.calculate_all_metrics()

        time_axis = np.arange(len(self.trace)) / self.fps
        onset_time = self.onset_frame / self.fps
        offset_time = min(self.offset_frame, len(self.trace)) / self.fps

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.axvspan(onset_time, offset_time, alpha=0.2, color="#F18F01", label="Stimulus")

        ax.plot(time_axis, self.trace, color="black", linewidth=2.5, label="Trace")

        if self._peak_frame is not None and self._peak_frame < len(time_axis):
            peak_time = time_axis[self._peak_frame]
            ax.plot(
                peak_time,
                self._peak_value,
                marker="o",
                markersize=15,
                color="black",
                zorder=5,
                label=f"Peak ({self._peak_value:.3f})",
            )

        if y_limits is not None:
            ax.set_ylim(y_limits)
        else:
            ax.set_ylim(-0.1, 0.4)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        y_offset_base = y_range * 0.05

        if self._above_threshold:
            self._plot_rise_time_annotation(ax, time_axis, y_min, y_offset_base)
            self._plot_half_width_annotation(ax, time_axis, y_min, y_offset_base)
            self._plot_latency_annotation(ax, time_axis, y_min, y_max, y_offset_base)
            self._plot_decay_annotation(ax, y_min, y_offset_base)
            self._plot_time_to_peak_annotation(ax, time_axis, y_max, y_offset_base)
            self._plot_threshold_lines(ax, time_axis)

        self._plot_metrics_text(ax)

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("ΔF/F", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            ax.set_title("Trace Metrics", fontsize=14, fontweight="bold")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(output_path),
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            LOGGER.info(f"Saved plot to: {output_path}")
        else:
            plt.show()

        plt.close(fig)

    def _plot_rise_time_annotation(
        self,
        ax: plt.Axes,
        time_axis: np.ndarray,
        y_min: float,
        y_offset_base: float,
    ) -> None:
        """Plot rise time annotation."""
        if (
            self._rise_time is not None
            and self._time_20_rise is not None
            and self._time_80_rise is not None
        ):
            value_20_rise = np.interp(self._time_20_rise, time_axis, self.trace)
            value_80_rise = np.interp(self._time_80_rise, time_axis, self.trace)

            ax.plot(
                [self._time_20_rise, self._time_80_rise],
                [value_20_rise, value_80_rise],
                "b--",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )
            ax.plot(
                self._time_20_rise,
                value_20_rise,
                marker="o",
                markersize=8,
                color="blue",
                zorder=4,
            )
            ax.plot(
                self._time_80_rise,
                value_80_rise,
                marker="o",
                markersize=8,
                color="blue",
                zorder=4,
            )

            arrow = FancyArrowPatch(
                (self._time_20_rise, value_20_rise),
                (self._time_80_rise, value_80_rise),
                arrowstyle="<->",
                mutation_scale=15,
                color="blue",
                linewidth=2,
                zorder=6,
                alpha=0.8,
            )
            ax.add_patch(arrow)

            rise_text = f"Rise time: {self._rise_time:.3f}s"
            if self._rise_slope is not None:
                rise_text += f", Slope: {self._rise_slope:.3f} ΔF/F/s"

            text_x = self._time_20_rise - (self._time_80_rise - self._time_20_rise) * 0.2
            text_y = (value_20_rise + value_80_rise) / 2
            text_y = max(text_y, y_min + y_offset_base)

            ax.text(
                text_x,
                text_y,
                rise_text,
                fontsize=10,
                color="black",
                ha="right",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    def _plot_half_width_annotation(
        self,
        ax: plt.Axes,
        time_axis: np.ndarray,
        y_min: float,
        y_offset_base: float,
    ) -> None:
        """Plot half width annotation."""
        if (
            self._half_width is not None
            and self._time_50_rise is not None
            and self._time_50_fall is not None
        ):
            value_50_rise = np.interp(self._time_50_rise, time_axis, self.trace)
            value_50_fall = np.interp(self._time_50_fall, time_axis, self.trace)

            ax.plot(
                [self._time_50_rise, self._time_50_fall],
                [value_50_rise, value_50_fall],
                "b--",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )
            ax.plot(
                self._time_50_rise,
                value_50_rise,
                marker="|",
                markersize=12,
                color="blue",
                zorder=4,
                markeredgewidth=2,
            )
            ax.plot(
                self._time_50_fall,
                value_50_fall,
                marker="|",
                markersize=12,
                color="blue",
                zorder=4,
                markeredgewidth=2,
            )

            arrow = FancyArrowPatch(
                (self._time_50_rise, value_50_rise),
                (self._time_50_fall, value_50_fall),
                arrowstyle="<->",
                mutation_scale=15,
                color="blue",
                linewidth=2,
                zorder=6,
                alpha=0.8,
            )
            ax.add_patch(arrow)

            mid_time = (self._time_50_rise + self._time_50_fall) / 2
            mid_value = (value_50_rise + value_50_fall) / 2
            text_y = mid_value - y_offset_base * 3
            text_y = max(text_y, y_min + y_offset_base)

            ax.text(
                mid_time,
                text_y,
                f"Half width: {self._half_width:.3f}s",
                fontsize=10,
                color="black",
                ha="center",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    def _plot_latency_annotation(
        self,
        ax: plt.Axes,
        time_axis: np.ndarray,
        y_min: float,
        y_max: float,
        y_offset_base: float,
    ) -> None:
        """Plot latency annotation."""
        if self._latency is not None and self._response_onset_frame is not None:
            response_onset_time = self._response_onset_frame / self.fps
            response_onset_value = self.trace[self._response_onset_frame]

            line_length = (y_max - y_min) * 0.015
            line_dx = line_length * 0.7
            line_dy = line_length * 0.7

            ax.plot(
                [response_onset_time - line_dx, response_onset_time + line_dx],
                [response_onset_value + line_dy, response_onset_value - line_dy],
                color="blue",
                linewidth=2.5,
                zorder=6,
                label="Response onset",
            )

            text_x = response_onset_time + (y_max - y_min) * 0.05
            text_y = response_onset_value

            if text_y > y_max - y_offset_base * 2:
                text_y = y_max - y_offset_base * 2
            elif text_y < y_min + y_offset_base * 2:
                text_y = y_min + y_offset_base * 2

            trace_value_at_text_x = np.interp(text_x, time_axis, self.trace)
            if abs(text_y - trace_value_at_text_x) < (y_max - y_min) * 0.1:
                if text_y > trace_value_at_text_x:
                    text_y = trace_value_at_text_x + (y_max - y_min) * 0.12
                else:
                    text_y = trace_value_at_text_x - (y_max - y_min) * 0.12

            ax.text(
                text_x,
                text_y,
                f"Latency: {self._latency:.3f}s",
                fontsize=10,
                color="black",
                ha="left",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    def _plot_decay_annotation(
        self,
        ax: plt.Axes,
        y_min: float,
        y_offset_base: float,
    ) -> None:
        """Plot decay time annotation."""
        if (
            self._decay_time is not None
            and self._time_80_decay is not None
            and self._time_20_decay is not None
            and self._value_80_decay is not None
            and self._value_20_decay is not None
        ):
            ax.plot(
                self._time_80_decay,
                self._value_80_decay,
                marker="o",
                markersize=8,
                color="blue",
                zorder=4,
            )
            ax.plot(
                self._time_20_decay,
                self._value_20_decay,
                marker="o",
                markersize=8,
                color="blue",
                zorder=4,
            )

            arrow = FancyArrowPatch(
                (self._time_80_decay, self._value_80_decay),
                (self._time_20_decay, self._value_20_decay),
                arrowstyle="<->",
                mutation_scale=15,
                color="blue",
                linewidth=2,
                zorder=6,
                alpha=0.8,
            )
            ax.add_patch(arrow)

            decay_text = f"Decay time: {self._decay_time:.3f}s"
            if self._decay_slope is not None:
                decay_text += f", Slope: {self._decay_slope:.3f} ΔF/F/s"

            text_x = self._time_80_decay + (self._time_20_decay - self._time_80_decay) * 0.6
            mid_value = (self._value_80_decay + self._value_20_decay) / 2
            text_y = mid_value
            text_y = max(text_y, y_min + y_offset_base)

            ax.text(
                text_x,
                text_y,
                decay_text,
                fontsize=10,
                color="black",
                ha="left",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    def _plot_time_to_peak_annotation(
        self,
        ax: plt.Axes,
        time_axis: np.ndarray,
        y_max: float,
        y_offset_base: float,
    ) -> None:
        """Plot time to peak annotation."""
        if (
            self._peak_frame is not None
            and self._peak_frame < len(time_axis)
            and self._time_to_peak is not None
        ):
            peak_time = time_axis[self._peak_frame]
            text_y = self._peak_value + y_offset_base * 1.5
            text_y = min(text_y, y_max - y_offset_base)

            ax.text(
                peak_time,
                text_y,
                f"Time to peak: {self._time_to_peak:.3f}s",
                fontsize=10,
                color="black",
                ha="center",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    def _plot_threshold_lines(self, ax: plt.Axes, time_axis: np.ndarray) -> None:
        """Plot threshold lines."""
        if (
            self._peak_value is not None
            and self._value_at_latency is not None
            and self._response_amplitude is not None
        ):
            threshold_20 = self._value_at_latency + self._response_amplitude * 0.20
            threshold_50 = self._value_at_latency + self._response_amplitude * 0.50
            threshold_80 = self._value_at_latency + self._response_amplitude * 0.80

            blue_label_used = False
            if self._frame_20_rise is not None:
                ax.axhline(
                    y=threshold_20,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=3,
                    label="20/50/80" if not blue_label_used else None,
                )
                blue_label_used = True

            if self._frame_50_rise is not None or self._frame_50_fall is not None:
                ax.axhline(
                    y=threshold_50,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=3,
                    label="20/50/80" if not blue_label_used else None,
                )
                blue_label_used = True

            if self._frame_80_rise is not None:
                ax.axhline(
                    y=threshold_80,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=3,
                    label="20/50/80" if not blue_label_used else None,
                )

    def _plot_metrics_text(self, ax: plt.Axes) -> None:
        """Plot metrics text box."""
        metrics_text = []
        if self._rise_time is not None:
            rise_text = f"Rise time: {self._rise_time:.3f} s"
            if self._rise_slope is not None:
                rise_text += f", Slope: {self._rise_slope:.3f} ΔF/F/s"
            metrics_text.append(rise_text)
        else:
            metrics_text.append("Rise time: N/A")

        if self._half_width is not None:
            metrics_text.append(f"Half width: {self._half_width:.3f} s")
        else:
            metrics_text.append("Half width: N/A")

        if self._time_to_peak is not None:
            metrics_text.append(f"Time to peak: {self._time_to_peak:.3f} s")
        else:
            metrics_text.append("Time to peak: N/A")

        if self._latency is not None:
            metrics_text.append(f"Latency: {self._latency:.3f} s")
        else:
            metrics_text.append("Latency: N/A")

        if self._decay_time is not None:
            decay_text = f"Decay time: {self._decay_time:.3f} s"
            if self._decay_slope is not None:
                decay_text += f", Slope: {self._decay_slope:.3f} ΔF/F/s"
            metrics_text.append(decay_text)
        else:
            metrics_text.append("Decay time: N/A")

        if metrics_text:
            metrics_str = "\n".join(metrics_text)
            ax.text(
                0.02,
                0.98,
                metrics_str,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                color="black",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.95,
                    "edgecolor": "black",
                    "pad": 0.5,
                },
            )

    def plot_derivatives(self, output_path: Path | None = None) -> None:
        """
        Plot trace with first and second derivatives.

        Args:
            output_path: Optional path to save the plot.
        """
        if not self._metrics_calculated:
            self.calculate_all_metrics()

        time_axis = np.arange(len(self.trace)) / self.fps
        onset_time = self.onset_frame / self.fps
        offset_time = min(self.offset_frame, len(self.trace)) / self.fps

        stim_start = max(0, min(self.onset_frame, len(self.trace) - 3))
        stim_end = min(self.offset_frame, len(self.trace))
        trace_segment = self.trace[stim_start:stim_end]

        if len(trace_segment) < 3:
            LOGGER.warning("Trace segment too short for derivatives")
            return

        dt = 1.0 / self.fps
        first_derivative = np.gradient(trace_segment, dt)
        second_derivative = np.gradient(first_derivative, dt)

        first_derivative_time = np.arange(len(first_derivative)) / self.fps + stim_start / self.fps
        second_derivative_time = first_derivative_time

        peak_first_derivative_idx = None
        window_start = None
        window_end = None

        if self._peak_value is not None and self._peak_value >= self.peak_threshold:
            peak_first_derivative_idx = int(np.argmax(first_derivative))

            negative_indices = np.where(
                (first_derivative < 0) & (np.arange(len(first_derivative)) < peak_first_derivative_idx)
            )[0]
            if len(negative_indices) > 0:
                window_start = int(negative_indices[-1])
            else:
                window_start = 0

            window_end = peak_first_derivative_idx

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax_trace, ax_first, ax_second = axes

        ax_trace.plot(
            time_axis,
            self.trace,
            color="black",
            linewidth=2,
            label="Trace",
        )

        if self._peak_value is not None and self._peak_frame is not None:
            peak_time = self._peak_frame / self.fps
            ax_trace.plot(
                peak_time,
                self._peak_value,
                marker="o",
                markersize=10,
                color="black",
                zorder=5,
                label=f"Peak ({self._peak_value:.3f})",
            )

        ax_trace.axvspan(onset_time, offset_time, alpha=0.2, color="#F18F01", label="Stimulus")
        ax_trace.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_trace.set_ylabel("ΔF/F", fontsize=12)
        ax_trace.set_title(
            "Trace and Derivatives",
            fontsize=14,
            fontweight="bold",
        )
        ax_trace.legend(loc="upper right", fontsize=9)
        ax_trace.spines[["top", "right"]].set_visible(False)
        ax_trace.grid(True, alpha=0.3)

        ax_first.plot(
            first_derivative_time, first_derivative, color="blue", linewidth=2, label="First derivative"
        )

        if peak_first_derivative_idx is not None:
            peak_first_derivative_time = (stim_start + peak_first_derivative_idx) / self.fps
            peak_first_derivative_value = first_derivative[peak_first_derivative_idx]
            ax_first.plot(
                peak_first_derivative_time,
                peak_first_derivative_value,
                marker="o",
                markersize=10,
                color="green",
                zorder=5,
                label="Peak of first derivative",
            )
            ax_first.axvline(
                x=peak_first_derivative_time,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
            )

        ax_first.axvspan(onset_time, offset_time, alpha=0.2, color="#F18F01")
        ax_first.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_first.set_ylabel("First derivative", fontsize=12)
        ax_first.legend(loc="upper right", fontsize=9)
        ax_first.spines[["top", "right"]].set_visible(False)
        ax_first.grid(True, alpha=0.3)

        ax_second.plot(
            second_derivative_time,
            second_derivative,
            color="black",
            linewidth=1.5,
            alpha=0.3,
            label="Second derivative (full)",
        )

        if window_start is not None and window_end is not None and window_end > window_start:
            window_second_derivative = second_derivative[window_start:window_end]
            window_second_derivative_time = second_derivative_time[window_start:window_end]
            ax_second.plot(
                window_second_derivative_time,
                window_second_derivative,
                color="red",
                linewidth=3,
                label="Second derivative (search window)",
            )

            window_start_time = second_derivative_time[window_start]
            window_end_time = second_derivative_time[window_end - 1]
            ax_second.axvspan(
                window_start_time,
                window_end_time,
                alpha=0.2,
                color="yellow",
                label="Search window",
            )

        if self._latency is not None and self._response_onset_frame is not None:
            response_onset_time = self._response_onset_frame / self.fps
            derivative_idx = self._response_onset_frame - stim_start
            if 0 <= derivative_idx < len(second_derivative):
                peak_second_derivative_value = second_derivative[derivative_idx]
                ax_second.plot(
                    response_onset_time,
                    peak_second_derivative_value,
                    marker="o",
                    markersize=10,
                    color="blue",
                    zorder=6,
                    label="Response onset (latency)",
                    markeredgecolor="black",
                    markeredgewidth=1,
                )
                ax_second.axvline(
                    x=response_onset_time,
                    color="blue",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2,
                )

        ax_second.axvspan(onset_time, offset_time, alpha=0.2, color="#F18F01")
        ax_second.axvline(
            x=onset_time,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Stimulus onset",
        )
        ax_second.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_second.set_xlabel("Time (s)", fontsize=12)
        ax_second.set_ylabel("Second derivative", fontsize=12)
        ax_second.legend(loc="upper right", fontsize=9)
        ax_second.spines[["top", "right"]].set_visible(False)
        ax_second.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(output_path),
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            LOGGER.info(f"Saved derivative plot to: {output_path}")
        else:
            plt.show()

        plt.close(fig)
