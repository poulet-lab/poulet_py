"""
Mask creation, saving, loading, and application functions.

This module provides functions for working with circular masks
used to isolate regions of interest in widefield imaging data.
"""

import json
from pathlib import Path

import numpy as np

from poulet_py import LOGGER


def apply_circular_mask(
    data: np.ndarray,
    mask_data: dict[str, float],
    reference_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """
    Apply a circular mask to a 3D imaging stack.

    Creates a circular mask based on the provided center and radius,
    then applies it to all frames of the input data. Pixels outside
    the mask are set to zero.

    Args:
        data: 3D numpy array with shape (frames, height, width).
        mask_data: Dictionary containing mask parameters:
            - center_x: X coordinate of mask center.
            - center_y: Y coordinate of mask center.
            - radius: Radius of the circular mask.
        reference_shape: Optional tuple of (height, width) used for
            scaling mask coordinates if the mask was created on
            a different resolution image.

    Returns:
        3D numpy array with the same shape as input, where pixels
        outside the circular mask are set to zero. Returns None
        on error.
    """
    if data.ndim != 3:
        LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
        return None

    try:
        T, H, W = data.shape
        center_x = mask_data["center_x"]
        center_y = mask_data["center_y"]
        radius = mask_data["radius"]

        LOGGER.info(
            f"Data shape: {data.shape}, Mask center: ({center_x}, {center_y}), radius: {radius}"
        )

        if center_x >= W or center_y >= H:
            LOGGER.warning(
                f"Mask center ({center_x}, {center_y}) is outside image bounds ({W}, {H}). "
                f"Scaling mask coordinates to match image size."
            )
            if reference_shape is not None:
                ref_H, ref_W = reference_shape
                scale_x = W / ref_W
                scale_y = H / ref_H
                center_x = center_x * scale_x
                center_y = center_y * scale_y
                radius = radius * min(scale_x, scale_y)
                LOGGER.info(
                    f"Scaled mask: center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}"
                )

        y, x = np.ogrid[:H, :W]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

        masked_data = data.copy()
        masked_data[:, ~mask] = 0

        LOGGER.info(
            f"Applied mask: center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}"
        )
        LOGGER.info(f"Masked data shape: {masked_data.shape}")

        return masked_data

    except Exception:
        LOGGER.exception("Error applying mask")
        return None


def save_mask_json(
    mask_data: dict[str, float],
    output_path: Path,
) -> Path | None:
    """
    Save mask parameters to a JSON file.

    Writes the mask center coordinates and radius to a JSON file
    for later reuse across multiple trials in a session.

    Args:
        mask_data: Dictionary containing mask parameters:
            - center_x: X coordinate of mask center.
            - center_y: Y coordinate of mask center.
            - radius: Radius of the circular mask.
        output_path: Path where the JSON file will be saved.
            Parent directories are created if they don't exist.

    Returns:
        Path to the saved file, or None on error.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(mask_data, f, indent=2)

        LOGGER.info(f"Saved mask to: {output_path}")
        LOGGER.info(
            f"  Center: ({mask_data['center_x']}, "
            f"{mask_data['center_y']}), Radius: {mask_data['radius']}"
        )
        return output_path

    except Exception:
        LOGGER.exception("Error saving mask")
        return None


def load_mask_json(mask_path: Path) -> dict[str, float] | None:
    """
    Load mask parameters from a JSON file.

    Reads previously saved mask parameters for reuse across
    multiple trials.

    Args:
        mask_path: Path to the JSON file containing mask parameters.

    Returns:
        Dictionary containing mask parameters (center_x, center_y,
        radius), or None if the file doesn't exist or cannot be read.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        LOGGER.warning(f"Mask file not found: {mask_path}")
        return None

    try:
        with open(mask_path) as f:
            mask_data = json.load(f)
        LOGGER.info(f"Loaded mask from: {mask_path}")
        return mask_data
    except Exception:
        LOGGER.exception(f"Error loading mask: {mask_path}")
        return None
