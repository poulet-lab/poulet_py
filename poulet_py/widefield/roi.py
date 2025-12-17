"""
Region of interest (ROI) functions for widefield imaging analysis.

This module provides functions for defining and extracting data
from regions of interest in widefield imaging data.
"""

import numpy as np

from poulet_py import LOGGER


def centroid_from_percentile(
    data: np.ndarray,
    percentile: float = 95.0,
) -> tuple[int, int]:
    """
    Calculate the centroid of pixels above a percentile threshold.

    Identifies pixels with values above the specified percentile
    and computes their geometric center. This is useful for
    automatically finding the center of activity in an image.

    Args:
        data: 2D numpy array containing the image data.
        percentile: The percentile threshold (0-100). Pixels above
            this percentile are included in centroid calculation.
            Default is 95.0 (top 5% of pixels).

    Returns:
        Tuple of (x, y) coordinates representing the centroid.
        If no pixels are above threshold, returns the image center.

    Raises:
        ValueError: If input data is not 2D.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    thr = np.percentile(data.ravel(), percentile)
    points = np.where(data > thr)

    if len(points[0]) == 0:
        LOGGER.warning(
            f"No points above {percentile}th percentile threshold ({thr:.2f})"
        )
        H, W = data.shape
        return (W // 2, H // 2)

    roi_x = int(np.mean(points[1]))
    roi_y = int(np.mean(points[0]))

    LOGGER.info(
        f"Calculated ROI centroid: ({roi_x}, {roi_y}) from "
        f"{len(points[0])} points above {percentile}th percentile "
        f"({thr:.2f})"
    )

    return (roi_x, roi_y)


def trace_within_circular_roi(
    data: np.ndarray,
    center: tuple[int, int],
    diameter: float = 50.0,
) -> np.ndarray:
    """
    Extract mean fluorescence trace from a circular ROI.

    Defines a circular region of interest and computes the mean
    pixel value within that region for each frame of the movie.

    Args:
        data: 3D numpy array with shape (frames, height, width).
        center: Tuple of (x, y) coordinates for the ROI center.
        diameter: Diameter of the circular ROI in pixels.
            Default is 50.0.

    Returns:
        1D numpy array of length (frames) containing the mean
        fluorescence value within the ROI for each frame.

    Raises:
        ValueError: If input data is not 3D.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got: {data.shape}")

    T, H, W = data.shape
    center_x, center_y = center
    radius = diameter / 2.0

    if center_x < 0 or center_x >= W or center_y < 0 or center_y >= H:
        LOGGER.warning(
            f"ROI center ({center_x}, {center_y}) is outside image bounds ({W}, {H})"
        )

    y, x = np.ogrid[:H, :W]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    trace = data[:, mask].mean(axis=1)

    LOGGER.info(
        f"Calculated trace from ROI: center=({center_x}, {center_y}), "
        f"diameter={diameter}, trace length={T}, "
        f"mean={trace.mean():.2f}, std={trace.std():.2f}"
    )

    return trace
