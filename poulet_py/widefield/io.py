"""
Input/output functions for widefield imaging data.

This module provides functions for loading various data files
associated with widefield imaging trials, including TIFF stacks,
timestamps, and sensor data from HDF5 files.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from skimage import io as skio

from poulet_py import LOGGER


def load_imaging(tiff_path: Path) -> np.ndarray:
    """
    Load imaging data from a multi-page TIFF file.

    Reads the TIFF stack and returns it as a 3D numpy array
    with dimensions (frames, height, width).

    Args:
        tiff_path: Path to the TIFF file to load.

    Returns:
        3D numpy array containing the imaging stack.

    Raises:
        FileNotFoundError: If the TIFF file does not exist.
        Exception: If the TIFF file cannot be read.
    """
    LOGGER.info(f"Loading imaging data from: {tiff_path.name}")
    data = skio.imread(str(tiff_path))
    LOGGER.info(f"Loaded imaging stack: {data.shape}")
    return data


def load_green_reference(green_path: Path) -> np.ndarray | None:
    """
    Load the green reference image from a TIFF file.

    The green reference image is typically used for anatomical
    reference and mask creation. If the file is a stack, only
    the first frame is returned.

    Args:
        green_path: Path to the green reference TIFF file.

    Returns:
        2D numpy array containing the reference image,
        or None if the file does not exist.
    """
    if not green_path.exists():
        LOGGER.warning(f"Green reference not found: {green_path}")
        return None

    LOGGER.info(f"Loading green reference from: {green_path.name}")
    img = skio.imread(str(green_path))
    if img.ndim == 3:
        img = img[0]
    LOGGER.info(f"Loaded green reference: {img.shape}")
    return img


def load_timestamps(csv_path: Path) -> pd.DataFrame | None:
    """
    Load frame timestamps from a CSV file.

    Reads the semicolon-separated CSV file containing timing
    information for each frame in the recording.

    Args:
        csv_path: Path to the CSV file containing timestamps.

    Returns:
        DataFrame with timestamp data, or None if file not found.
        Unnamed columns are automatically removed.
    """
    if not csv_path.exists():
        LOGGER.warning(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, sep=";")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    LOGGER.info(f"Loaded timestamps: {len(df)} rows")
    return df


def load_sensors(
    h5_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Load sensor data from an HDF5 file.

    Reads all datasets from the HDF5 file along with their
    attributes and file-level attributes. Sensor data typically
    includes temperature readings and camera trigger signals.

    Args:
        h5_path: Path to the HDF5 file containing sensor data.

    Returns:
        A tuple containing:
        - sensor_data: Dictionary mapping dataset names to numpy arrays.
        - sensor_attrs: Dictionary mapping dataset names to their attributes.
        - file_attrs: Dictionary of file-level attributes.

        Returns empty dictionaries if the file does not exist.
    """
    sensor_data: dict[str, np.ndarray] = {}
    sensor_attrs: dict[str, dict[str, Any]] = {}
    file_attrs: dict[str, Any] = {}

    if not h5_path.exists():
        LOGGER.warning(f"H5 not found: {h5_path}")
        return sensor_data, sensor_attrs, file_attrs

    with h5py.File(h5_path, "r") as f:
        file_attrs = dict(f.attrs)

        def _visit_datasets(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                sensor_data[name] = np.array(obj)
                sensor_attrs[name] = dict(obj.attrs)

        f.visititems(_visit_datasets)

    LOGGER.info(f"Loaded {len(sensor_data)} sensor traces from H5")
    return sensor_data, sensor_attrs, file_attrs


def tiff_to_numpy(source: str | Path | np.ndarray | None) -> np.ndarray | None:
    """
    Convert a TIFF file path or array to a numpy array.

    This utility function handles multiple input types for
    flexibility in data processing pipelines.

    Args:
        source: The data source, which can be:
            - A numpy array (returned as-is)
            - A string or Path to a TIFF file
            - None (returns None with a warning)

    Returns:
        Numpy array containing the image data, or None if the
        source is invalid or the file cannot be loaded.
    """
    if source is None:
        LOGGER.warning("No source provided")
        return None

    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, (str, Path)):
        tiff_path = Path(source)
        if not tiff_path.exists():
            LOGGER.error(f"TIFF file not found: {tiff_path}")
            return None
        try:
            LOGGER.info(f"Loading TIFF: {tiff_path.name}")
            data = skio.imread(str(tiff_path))
            LOGGER.info(f"Loaded: {data.shape}")
            return data
        except Exception:
            LOGGER.exception(f"Error loading TIFF: {tiff_path}")
            return None

    LOGGER.error(f"Invalid source type: {type(source)}")
    return None
