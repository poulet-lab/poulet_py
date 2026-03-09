"""
Metric calculation functions for widefield imaging analysis.

This module provides functions for computing common metrics
on widefield imaging data, including percentile projections,
baseline calculations, delta F/F normalization, and spatial
threshold metrics for activity quantification.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from poulet_py import LOGGER


def calculate_percentile_movie(
    data: np.ndarray,
    percentile: float = 15.0,
    stimulus_start_frame: int | None = None,
    baseline_ms: float | None = None,
    fps: float | None = None,
) -> np.ndarray | None:
    """
    Calculate a percentile projection across time for each pixel.

    Computes the specified percentile value for each pixel across
    all frames or within a specified baseline window. This is commonly
    used to establish a baseline fluorescence value (F0) for delta F/F
    calculations.

    Args:
        data: 3D numpy array with shape (frames, height, width).
        percentile: The percentile to compute (0-100). Default is 15.0.
        stimulus_start_frame: Frame index where stimulus begins.
            If provided along with baseline_ms, only frames before
            the stimulus are used for calculation.
        baseline_ms: Duration of baseline period in milliseconds.
            Used with stimulus_start_frame to define the window.
        fps: Frames per second. Required when using window parameters.

    Returns:
        2D numpy array with shape (height, width) containing the
        percentile value for each pixel. Zero values are replaced
        with 1 to prevent division by zero. Returns None on error.
    """
    if data.ndim != 3:
        LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
        return None

    T, H, W = data.shape
    window_data = data
    window_info = "all frames"

    if stimulus_start_frame is not None and baseline_ms is not None:
        if fps is None:
            LOGGER.error("fps must be provided when using window parameters")
            return None

        baseline_frames = int(baseline_ms / 1000.0 * fps)
        baseline_start = max(0, stimulus_start_frame - baseline_frames)
        baseline_end = stimulus_start_frame

        if baseline_end > T:
            LOGGER.warning(
                f"Stimulus start frame ({baseline_end}) exceeds data "
                f"length ({T}). Using data length instead."
            )
            baseline_end = T

        if baseline_start >= baseline_end:
            LOGGER.error(f"Invalid baseline window: start={baseline_start}, end={baseline_end}")
            return None

        window_data = data[baseline_start:baseline_end]
        window_info = f"frames [{baseline_start}:{baseline_end}] ({len(window_data)} frames)"

    LOGGER.info(f"Calculating {percentile}th percentile for {window_info} ({H}x{W})")

    f_base = np.percentile(window_data, percentile, axis=0)
    f_base = np.where(f_base == 0, 1, f_base)

    LOGGER.info(
        f"Percentile calculated: min={f_base.min():.2f}, "
        f"max={f_base.max():.2f}, mean={f_base.mean():.2f}"
    )
    return f_base


def calculate_deltaff_movie(
    data: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray | None:
    """
    Calculate delta F/F (ΔF/F) for each frame.

    Computes the relative fluorescence change for each pixel
    and frame using the formula: ΔF/F = (F - F0) / F0 = F/F0 - 1

    Args:
        data: 3D numpy array with shape (frames, height, width)
            containing the raw fluorescence values (F).
        baseline: 2D numpy array with shape (height, width)
            containing the baseline fluorescence values (F0).

    Returns:
        3D numpy array with shape (frames, height, width) containing
        the ΔF/F values. Returns None on error.

    Raises:
        Logs error if data is not 3D or baseline is not 2D.
        Logs error if spatial dimensions don't match.
    """
    if data.ndim != 3:
        LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
        return None

    if baseline.ndim != 2:
        LOGGER.error(f"Expected 2D baseline array (H, W), got: {baseline.shape}")
        return None

    T, H, W = data.shape
    baseline_H, baseline_W = baseline.shape

    if H != baseline_H or W != baseline_W:
        LOGGER.error(
            f"Baseline shape ({baseline_H}, {baseline_W}) does not match "
            f"data spatial dimensions ({H}, {W})"
        )
        return None

    dff = data / baseline - 1

    LOGGER.info(
        f"Calculated ΔF/F: shape={dff.shape}, "
        f"min={dff.min():.3f}, max={dff.max():.3f}, "
        f"mean={dff.mean():.3f}"
    )
    return dff


def calculate_baseline_movie(
    data: np.ndarray,
    stimulus_start_frame: int,
    baseline_ms: float,
    fps: float,
) -> np.ndarray | None:
    """
    Calculate mean baseline image from frames before stimulus onset.

    Computes the temporal mean of frames within the baseline period,
    which is defined as the time window before the stimulus starts.

    Args:
        data: 3D numpy array with shape (frames, height, width).
        stimulus_start_frame: Frame index where stimulus begins.
        baseline_ms: Duration of baseline period in milliseconds.
            Frames from (stimulus_start - baseline_frames) to
            stimulus_start are averaged.
        fps: Frames per second, used to convert ms to frames.

    Returns:
        2D numpy array with shape (height, width) containing the
        mean baseline image. Returns None on error.
    """
    if data.ndim != 3:
        LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
        return None

    T, _, _ = data.shape

    baseline_frames = int(baseline_ms / 1000.0 * fps)
    baseline_start = stimulus_start_frame - baseline_frames
    baseline_end = stimulus_start_frame

    if baseline_start < 0:
        LOGGER.warning(
            f"Baseline start frame ({baseline_start}) is negative. Using frame 0 instead."
        )
        baseline_start = 0

    if baseline_end > T:
        LOGGER.warning(
            f"Stimulus start frame ({baseline_end}) exceeds data length "
            f"({T}). Using data length instead."
        )
        baseline_end = T

    if baseline_start >= baseline_end:
        LOGGER.error(f"Invalid baseline period: start={baseline_start}, end={baseline_end}")
        return None

    baseline_period = data[baseline_start:baseline_end]
    baseline_mean = np.mean(baseline_period, axis=0)

    LOGGER.info(
        f"Calculated baseline: period=[{baseline_start}:{baseline_end}], "
        f"duration={baseline_ms}ms, "
        f"baseline shape={baseline_mean.shape}, "
        f"min={baseline_mean.min():.2f}, max={baseline_mean.max():.2f}, "
        f"mean={baseline_mean.mean():.2f}"
    )

    return baseline_mean


def calculate_spatial_threshold_metrics(
    image: np.ndarray,
    threshold_percent: float = 85.0,
    smoothing_sigma: float | None = 20.0,
) -> np.void | None:
    """
    Calculate spatial threshold metrics for a 2D activity image.

    Applies optional Gaussian smoothing, computes a threshold as a percentage
    of maximum activity, creates a threshold mask, and calculates various
    metrics including area under curve (AUC) above threshold.

    Returns a single structured record (numpy void scalar) so fields are
    accessed by name, e.g. result['threshold'], result['threshold_mask'].
    See :ref:`numpy:basics.rec` for structured arrays.

    Args:
        image: 2D numpy array representing activity (e.g., peak dF/F frame).
        threshold_percent: Percentage of maximum activity to use as threshold.
            Default is 85.0 (85% of max).
        smoothing_sigma: Sigma for Gaussian smoothing in pixels. If None,
            no smoothing is applied. Default is 20.0.

    Returns:
        A structured numpy scalar (np.void) with fields:
            - threshold: The computed threshold value (float64)
            - threshold_mask: Boolean 2D mask (pixels above threshold)
            - n_pixels_above: Count of pixels above threshold (int64)
            - total_pixels: Total pixel count in image (int64)
            - percent_above: Percentage of pixels above threshold (float64)
            - auc_above_threshold: Sum of pixel values above threshold (float64)
            - smoothed_image: The smoothed 2D image (float64)
            - max_activity: Maximum value in smoothed image (float64)
            - min_activity: Minimum value in smoothed image (float64)
            - mean_activity: Mean value in smoothed image (float64)
            - std_activity: Standard deviation in smoothed image (float64)
        Returns None if input is invalid.

    Example:
        >>> peak_image = dff_data[peak_frame]
        >>> metrics = calculate_spatial_threshold_metrics(
        ...     image=peak_image, threshold_percent=85.0, smoothing_sigma=20.0
        ... )
        >>> print(f"AUC above threshold: {metrics['auc_above_threshold']:.3f}")
        >>> print(f"Pixels above: {metrics['percent_above']:.1f}%")
    """
    if image.ndim != 2:
        LOGGER.error(f"Expected 2D array (H, W), got shape: {image.shape}")
        return None

    if image.size == 0:
        LOGGER.error("Empty image provided")
        return None

    if smoothing_sigma is not None and smoothing_sigma > 0:
        smoothed_image = gaussian_filter(image, sigma=smoothing_sigma)
        LOGGER.debug(f"Applied Gaussian smoothing with sigma={smoothing_sigma}")
    else:
        smoothed_image = image.copy()

    max_activity = float(np.max(smoothed_image))
    min_activity = float(np.min(smoothed_image))
    mean_activity = float(np.mean(smoothed_image))
    std_activity = float(np.std(smoothed_image))

    if max_activity <= 0:
        LOGGER.warning(
            f"Max activity ({max_activity}) is not positive. "
            "Threshold metrics may not be meaningful."
        )

    threshold = threshold_percent / 100.0 * max_activity

    threshold_mask = smoothed_image >= threshold
    n_pixels_above = int(np.sum(threshold_mask))
    total_pixels = int(image.size)
    percent_above = 100.0 * n_pixels_above / total_pixels

    pixels_above_values = smoothed_image[threshold_mask]
    auc_above_threshold = float(np.sum(pixels_above_values))

    LOGGER.info(
        f"Spatial threshold metrics: threshold={threshold:.4f} "
        f"({threshold_percent:.0f}% of max), "
        f"pixels_above={n_pixels_above}/{total_pixels} ({percent_above:.1f}%), "
        f"AUC={auc_above_threshold:.4f}"
    )

    H, W = smoothed_image.shape
    dtype = np.dtype([
        ("threshold", np.float64),
        ("threshold_mask", bool, (H, W)),
        ("n_pixels_above", np.int64),
        ("total_pixels", np.int64),
        ("percent_above", np.float64),
        ("auc_above_threshold", np.float64),
        ("smoothed_image", np.float64, (H, W)),
        ("max_activity", np.float64),
        ("min_activity", np.float64),
        ("mean_activity", np.float64),
        ("std_activity", np.float64),
    ])
    out = np.empty(1, dtype=dtype)
    out["threshold"] = threshold
    out["threshold_mask"] = threshold_mask
    out["n_pixels_above"] = n_pixels_above
    out["total_pixels"] = total_pixels
    out["percent_above"] = percent_above
    out["auc_above_threshold"] = auc_above_threshold
    out["smoothed_image"] = smoothed_image
    out["max_activity"] = max_activity
    out["min_activity"] = min_activity
    out["mean_activity"] = mean_activity
    out["std_activity"] = std_activity
    return out[0]
