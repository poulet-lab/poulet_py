"""
Functions for detecting stimulus from sensor traces.
"""

import matplotlib.pyplot as plt
import numpy as np

from poulet_py import LOGGER


def detect_stimulus_frames(
    trace: np.ndarray,
    sampling_rate: float,
    camera_fps: float | None = None,
    derivative_threshold_factor: float = 2.0,
    min_pulse_duration_ms: float = 500.0,
    plot: bool = False,
) -> dict[str, int] | None:
    """
    Detect stimulus onset and offset from trace using derivative-based detection.

    Uses the derivative (rate of change) to detect rapid temperature changes,
    which is more robust for small amplitude stimuli than absolute thresholds.

    Args:
        trace: Trace array (any units).
        sampling_rate: Sampling rate of trace (Hz).
        camera_fps: Camera frame rate for frame conversion.
        derivative_threshold_factor: Multiplier for std of derivative to detect
                                    significant rate changes (default: 5.0).
        min_pulse_duration_ms: Minimum pulse duration in milliseconds to
                               consider valid (default: 500.0).
        plot: Whether to plot detection results.

    Returns:
        Dictionary with 'onset_frame', 'offset_frame', 'onset_sample',
        'offset_sample', 'onset_time', 'offset_time', 'direction'
        ('warming' or 'cooling'), or None if detection fails.
    """
    if len(trace) < 100:
        LOGGER.warning("Trace too short for detection")
        return None

    smooth_window = int(sampling_rate * 0.05)
    smooth_window = max(smooth_window, 1)
    smoothed = np.convolve(trace, np.ones(smooth_window) / smooth_window, mode="same")

    deriv_window = int(sampling_rate * 0.1)
    deriv_window = max(deriv_window, 1)
    derivative = np.gradient(smoothed, 1.0 / sampling_rate)
    smooth_deriv = np.convolve(derivative, np.ones(deriv_window) / deriv_window, mode="same")

    search_start = int(sampling_rate * 3)
    search_end = int(sampling_rate * 10)
    search_end = min(search_end, len(smooth_deriv))

    search_region = smooth_deriv[search_start:search_end]
    max_deriv = np.max(np.abs(search_region))

    if max_deriv < 1e-10:
        LOGGER.warning("No significant derivative found in search region")
        return None

    deriv_threshold = max_deriv / derivative_threshold_factor

    min_onset_sample = int(sampling_rate * 3.0)
    min_gap_samples = int(sampling_rate * 0.5)

    rising = smooth_deriv > deriv_threshold
    falling = smooth_deriv < -deriv_threshold

    rising[:min_onset_sample] = False
    falling[:min_onset_sample] = False

    if not np.any(rising) and not np.any(falling):
        LOGGER.warning(
            f"No deviation detected after {min_onset_sample / sampling_rate:.1f}s "
            f"(derivative threshold: ±{deriv_threshold:.6f})"
        )
        return None

    rising_indices = np.where(rising)[0]
    falling_indices = np.where(falling)[0]

    first_rising = rising_indices[0] if len(rising_indices) > 0 else len(trace)
    first_falling = falling_indices[0] if len(falling_indices) > 0 else len(trace)

    if first_rising < first_falling:
        direction = "warming"
        onset_indices = rising_indices
        offset_indices = falling_indices
    else:
        direction = "cooling"
        onset_indices = falling_indices
        offset_indices = rising_indices

    if len(onset_indices) == 0:
        LOGGER.warning("No stimulus onset detected")
        return None

    onset_sample = onset_indices[0]

    offset_indices_after_onset = offset_indices[offset_indices > (onset_sample + min_gap_samples)]

    if len(offset_indices_after_onset) == 0:
        offset_sample = len(trace) - 1
    else:
        offset_sample = offset_indices_after_onset[0]

    min_pulse_samples = int(min_pulse_duration_ms / 1000.0 * sampling_rate)
    pulse_duration_samples = offset_sample - onset_sample

    if pulse_duration_samples < min_pulse_samples:
        for onset_idx in onset_indices:
            offset_after = offset_indices[offset_indices > (onset_idx + min_gap_samples)]
            if len(offset_after) > 0:
                duration = offset_after[0] - onset_idx
                if duration >= min_pulse_samples:
                    onset_sample = onset_idx
                    offset_sample = offset_after[0]
                    pulse_duration_samples = duration
                    break

        if pulse_duration_samples < min_pulse_samples:
            LOGGER.warning(f"No valid pulse found with duration >= {min_pulse_duration_ms:.1f}ms")
            return None

    if camera_fps is None:
        LOGGER.warning("Camera FPS not provided, cannot convert to frames")
        return {
            "onset_sample": int(onset_sample),
            "offset_sample": int(offset_sample),
            "onset_time": onset_sample / sampling_rate,
            "offset_time": offset_sample / sampling_rate,
            "direction": direction,
        }

    onset_frame = int(onset_sample * camera_fps / sampling_rate)
    offset_frame = int(offset_sample * camera_fps / sampling_rate)

    result = {
        "onset_frame": onset_frame,
        "offset_frame": offset_frame,
        "onset_sample": int(onset_sample),
        "offset_sample": int(offset_sample),
        "onset_time": onset_sample / sampling_rate,
        "offset_time": offset_sample / sampling_rate,
        "direction": direction,
    }

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        time_axis = np.arange(len(trace)) / sampling_rate

        ax = axes[0]
        ax.plot(time_axis, trace, linewidth=0.5, alpha=0.3, label="Trace (raw)", color="gray")
        ax.plot(time_axis, smoothed, linewidth=1.0, alpha=0.8, label="Trace (smoothed)")
        ax.axvline(x=result["onset_time"], color="red", linestyle="-", label="Onset", linewidth=2)
        ax.axvline(
            x=result["offset_time"], color="blue", linestyle="-", label="Offset", linewidth=2
        )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Temperature (V)")
        ax.set_title(f"Stimulus Detection - Trace ({direction})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(time_axis, smooth_deriv, linewidth=1.0, alpha=0.8, label="Derivative (smoothed)")
        ax.axhline(
            y=deriv_threshold,
            color="orange",
            linestyle="--",
            label=f"Rising threshold: +{deriv_threshold:.6f}",
            alpha=0.7,
        )
        ax.axhline(
            y=-deriv_threshold,
            color="purple",
            linestyle="--",
            label=f"Falling threshold: -{deriv_threshold:.6f}",
            alpha=0.7,
        )
        ax.axhline(y=0, color="green", linestyle="--", alpha=0.5)
        ax.axvline(x=result["onset_time"], color="red", linestyle="-", label="Onset", linewidth=2)
        ax.axvline(
            x=result["offset_time"], color="blue", linestyle="-", label="Offset", linewidth=2
        )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Rate of change")
        ax.set_title(f"Stimulus Detection - Derivative ({direction})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    LOGGER.info(
        f"Detected stimulus ({direction}): onset={result['onset_time']:.2f}s "
        f"(frame {onset_frame}), offset={result['offset_time']:.2f}s "
        f"(frame {offset_frame}), duration={result['offset_time'] - result['onset_time']:.2f}s"
    )

    return result
