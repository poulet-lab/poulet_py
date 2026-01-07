"""
Input/output functions for widefield imaging data.

This module provides functions for loading various data files
associated with widefield imaging trials, including TIFF stacks,
timestamps, and sensor data from HDF5 files.
"""

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from skimage import io as skio

from poulet_py import LOGGER


def load_imaging(imaging_path: Path) -> np.ndarray:
    """
    Load imaging data from a TIFF or NPY file.

    Reads the imaging stack and returns it as a 3D numpy array
    with dimensions (frames, height, width).

    Args:
        imaging_path: Path to the imaging file (.tiff or .npy).

    Returns:
        3D numpy array containing the imaging stack.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    imaging_path = Path(imaging_path)
    LOGGER.info(f"Loading imaging data from: {imaging_path.name}")

    if imaging_path.suffix.lower() == ".npy":
        data = np.load(str(imaging_path))
    elif imaging_path.suffix.lower() in (".tiff", ".tif"):
        data = skio.imread(str(imaging_path))
    else:
        msg = f"Unsupported imaging format: {imaging_path.suffix}"
        raise ValueError(msg)

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


def extract_trial_metadata(h5_path: Path) -> dict[str, Any]:
    """
    Extract all metadata from data.h5 to a dictionary.

    Extracts file-level attributes, dataset information, and parses
    temperature values from the comment field if present.

    Args:
        h5_path: Path to the data.h5 file.

    Returns:
        Dictionary containing:
        - source_h5: Original file path as string
        - trial_name: Name of parent folder
        - file_attributes: All file-level HDF5 attributes
        - datasets: Info about each dataset (shape, dtype, attributes)
        - parsed: Parsed temperature values if found in comment

    Raises:
        FileNotFoundError: If the H5 file does not exist.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        msg = f"H5 file not found: {h5_path}"
        raise FileNotFoundError(msg)

    metadata: dict[str, Any] = {
        "source_h5": str(h5_path),
        "trial_name": h5_path.parent.name,
        "file_attributes": {},
        "datasets": {},
        "parsed": {},
    }

    with h5py.File(h5_path, "r") as f:
        for key, val in f.attrs.items():
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            elif isinstance(val, (np.integer, np.floating)):
                val = val.item()
            metadata["file_attributes"][key] = val

        def _extract_dataset_info(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                ds_info: dict[str, Any] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attributes": {},
                }
                for k, v in obj.attrs.items():
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    elif isinstance(v, (np.integer, np.floating)):
                        v = v.item()
                    ds_info["attributes"][k] = v
                metadata["datasets"][name] = ds_info

        f.visititems(_extract_dataset_info)

    comment = metadata["file_attributes"].get("comment", "")
    match = re.search(r"temp\s+([\d.]+)\s*-\s*([\d.]+)", comment)
    if match:
        baseline = float(match.group(1))
        target = float(match.group(2))
        metadata["parsed"] = {
            "baseline_temperature": baseline,
            "target_temperature": target,
            "amplitude": target - baseline,
        }

    LOGGER.info(f"Extracted metadata from: {h5_path.name}")
    return metadata


def save_trial_metadata(metadata: dict[str, Any], output_path: Path) -> Path:
    """
    Save trial metadata to a JSON file.

    Args:
        metadata: Metadata dictionary from extract_trial_metadata.
        output_path: Path where to save the JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info(f"Saved metadata: {output_path}")
    return output_path


def load_trial_metadata(json_path: Path) -> dict[str, Any] | None:
    """
    Load trial metadata from a JSON file.

    Args:
        json_path: Path to the metadata JSON file.

    Returns:
        Metadata dictionary, or None if file not found.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        LOGGER.warning(f"Metadata file not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        metadata = json.load(f)

    LOGGER.info(f"Loaded metadata from: {json_path.name}")
    return metadata
