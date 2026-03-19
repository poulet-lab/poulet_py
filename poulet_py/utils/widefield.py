"""
Widefield imaging data analysis module.

This module provides a class-based interface for analyzing widefield
imaging TIFF data from body core temperature experiments.

A trial folder contains:
- recording.tiff: Multi-page TIFF stack with imaging data
- recording.csv: Timestamp metadata for frames
- data.h5: Sensor data (temperature, camera triggers)
- green.tiff: Reference/green channel image
"""

from abc import ABC, abstractmethod
from datetime import datetime

from numpy import ndarray


try:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd

    from poulet_py import LOGGER
    from skio import imread
    from pydantic import BaseModel, PrivateAttr, Field

except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

Also ensure: h5py, numpy, pandas, scikit-image, imageio, matplotlib are installed
"""
    raise ImportError(msg) from e


class BaseData(BaseModel, ABC):
    @abstractmethod
    def _load(self): ...


class SeparatedData(BaseData):
    # make restriction of tiff, npy only etc.
    video: Path = Field(..., description="")
    analog_output_data: Path = Field(...)
    reference_image: Path | None = Field(default=None)

    _video: ndarray = PrivateAttr()
    _reference_image: ndarray = PrivateAttr()
    _analog_output_data: ndarray = PrivateAttr()

    self.timestamps: pd.DataFrame | None = None
    self.sensor_data: dict[str, np.ndarray] = {}
    self.sensor_attrs: dict[str, dict[str, Any]] = {}
    self.file_attrs: dict[str, Any] = {}
    self.condition: dict[str, Any] | None = None
    self.roi: dict[str, Any] | None = None

    def _load(self):
        self._load_video()
        self._load_reference_image()

    def _load_video(self) -> np.ndarray:
        LOGGER.info(f"Loading imaging data from: {self.video.name}")

        if self.video.suffix.lower() == ".npy":
            data = np.load(str(self.video))
        elif self.video.suffix.lower() in (".tiff", ".tif"):
            data = imread(str(self.video))
        else:
            msg = f"Unsupported imaging format: {self.video.suffix}"
            raise ValueError(msg)

        LOGGER.info(f"Loaded imaging stack: {data.shape}")
        return data

    def _load_reference_image(self) -> None:
        if isinstance(self.reference_image, Path) and not self.reference_image.exists():
            LOGGER.warning(f"Green reference not found: {self.reference_image}")
            return None

        LOGGER.info(f"Loading green reference from: {self.reference_image.name}")
        img = skio.imread(str(self.reference_image))
        if img.ndim == 3:
            img = img[0]
        LOGGER.info(f"Loaded green reference: {img.shape}")
        return img


class Trial(BaseData):
    # make restriction of tiff, npy only etc.
    path: Path = Field(..., description="")
    data: BaseData = Field(...)

    def load_trial(self):
        self.data._load()


class Session(BaseModel):
    path: Path = Field(..., description="")
    start: datetime = Field()
    end: datetime = Field()

    _trials: list[Trial] = PrivateAttr()


class WidefieldAnalysis(BaseModel):
    session: Session = Field(..., description="")

    def _validate(self) -> None:
        """
        Validate the trial path and locate required files.

        Raises:
            FileNotFoundError: If trial path does not exist.
            ValueError: If recording.tiff is missing.
        """
        if not self.trial_path.exists():
            msg = f"Trial path does not exist: {self.trial_path}"
            raise FileNotFoundError(msg)

        if not self.trial_path.is_dir():
            msg = f"Trial path must be a directory: {self.trial_path}"
            raise ValueError(msg)

        self.video_path = self.trial_path / "recording.tiff"
        self.csv_path = self.trial_path / "recording.csv"
        self.h5_path = self.trial_path / "data.h5"
        self.green_path = self.trial_path / "green.tiff"

        if not self.video_path.exists():
            msg = f"recording.tiff not found in: {self.trial_path}"
            raise ValueError(msg)

    def load_session(self) -> None:
        """
        Load all data files from the trial folder.

        Loads the TIFF imaging stack, green reference image,
        timestamps CSV, and sensor data from the H5 file.
        After loading, prints a summary of the loaded data.
        """
        self._load_video()
        self._load_reference_image()
        self._load_timestamps()
        self._load_sensors()
        self._print_info()

    def _load_timestamps(self) -> None:
        """
        Load frame timestamps from CSV.

        Logs but does not raise exceptions on failure.
        """
        try:
            self.timestamps = wf_io.load_timestamps(self.csv_path)
        except Exception:
            LOGGER.exception(f"Error loading CSV: {self.csv_path}")

    def _load_sensors(self) -> None:
        """
        Load sensor data from HDF5 file.

        Logs but does not raise exceptions on failure.
        """
        try:
            (
                self.sensor_data,
                self.sensor_attrs,
                self.file_attrs,
            ) = wf_io.load_sensors(self.h5_path)
        except Exception:
            LOGGER.exception(f"Error loading H5: {self.h5_path}")

    def _print_info(self) -> None:
        """
        Print a summary of loaded data to the logger.

        Displays information about imaging data dimensions,
        timestamps, sensor traces, and file metadata.
        """
        LOGGER.info("=" * 60)
        LOGGER.info(f"Trial: {self.trial_path.name}")
        LOGGER.info("=" * 60)

        if self.imaging_data is not None:
            n_frames, height, width = self.imaging_data.shape
            LOGGER.info("Imaging data:")
            LOGGER.info(f"  Shape: {self.imaging_data.shape}")
            LOGGER.info(f"  Frames: {n_frames}")
            LOGGER.info(f"  Resolution: {width} x {height}")
            LOGGER.info(f"  Dtype: {self.imaging_data.dtype}")
            LOGGER.info(f"  Value range: [{self.imaging_data.min()}, {self.imaging_data.max()}]")
            size_mb = self.imaging_data.nbytes / (1024 * 1024)
            LOGGER.info(f"  Memory: {size_mb:.1f} MB")

        if self.timestamps is not None:
            LOGGER.info("Timestamps:")
            LOGGER.info(f"  Rows: {len(self.timestamps)}")
            LOGGER.info(f"  Columns: {list(self.timestamps.columns)}")

        if self.sensor_data:
            LOGGER.info("Sensor data:")
            for name, data in self.sensor_data.items():
                attrs = self.sensor_attrs.get(name, {})
                sr = attrs.get("sr", "unknown")
                LOGGER.info(f"  {name}: shape={data.shape}, sr={sr} Hz")

        if self.file_attrs:
            mouse_id = self.file_attrs.get("mouse_id", "unknown")
            protocol = self.file_attrs.get("protocol_name", "unknown")
            comment = self.file_attrs.get("comment", "")
            LOGGER.info("Metadata:")
            LOGGER.info(f"  Mouse: {mouse_id}")
            LOGGER.info(f"  Protocol: {protocol}")
            if comment:
                LOGGER.info(f"  Comment: {comment}")

        LOGGER.info("=" * 60)

    def downscale(
        self,
        target_resolution: tuple[int, int] | None = None,
        factor: int | None = None,
    ) -> np.ndarray | None:
        """
        Downscale the imaging data by averaging pixels.

        Reduces the spatial resolution of the imaging stack by
        averaging blocks of pixels. Useful for reducing memory
        usage and speeding up processing.

        Args:
            target_resolution: Target (height, width) dimensions.
                The downscaling factor is computed automatically.
            factor: Downscaling factor (e.g., 2 means 2x2 blocks).
                Either target_resolution or factor must be provided.

        Returns:
            Downscaled 3D numpy array with dtype uint16, or None
            if imaging data is not loaded.

        Raises:
            ValueError: If neither target_resolution nor factor provided.
        """
        if self.imaging_data is None:
            LOGGER.warning("No imaging data loaded. Call load_data() first.")
            return None

        if target_resolution is None and factor is None:
            msg = "Must provide either target_resolution or factor"
            raise ValueError(msg)

        T, H, W = self.imaging_data.shape
        mov = self.imaging_data.copy()

        if target_resolution is not None:
            target_H, target_W = target_resolution
            factor_H = H / target_H
            factor_W = W / target_W

            if not factor_H.is_integer() or not factor_W.is_integer():
                factor_H_int = int(np.ceil(factor_H))
                factor_W_int = int(np.ceil(factor_W))
                new_H = target_H * factor_H_int
                new_W = target_W * factor_W_int
                pad_H = new_H - H
                pad_W = new_W - W
                LOGGER.warning(
                    f"Target {target_resolution} requires factors "
                    f"({factor_H:.2f}, {factor_W:.2f}). "
                    f"Padding ({pad_H}, {pad_W}) pixels to use factors "
                    f"({factor_H_int}, {factor_W_int})."
                )
                mov = np.pad(
                    mov, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0
                )
                T, H, W = mov.shape
                factor_H = factor_H_int
                factor_W = factor_W_int
            else:
                factor_H = int(factor_H)
                factor_W = int(factor_W)

            if factor_H == factor_W:
                factor = factor_H
                mov = mov.reshape(T, H // factor, factor, W // factor, factor).mean(4).mean(2)
            else:
                LOGGER.info(f"Using different factors: H={factor_H}, W={factor_W}")
                mov = mov.reshape(T, H // factor_H, factor_H, W, 1).mean(2)
                mov = mov.reshape(T, H // factor_H, W // factor_W, factor_W).mean(3)

        elif factor is not None:
            if H % factor != 0 or W % factor != 0:
                pad_H = (factor - (H % factor)) % factor
                pad_W = (factor - (W % factor)) % factor
                LOGGER.warning(
                    f"Dimensions ({H}, {W}) not divisible by factor {factor}. "
                    f"Padding ({pad_H}, {pad_W}) pixels."
                )
                mov = np.pad(
                    mov, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0
                )
                T, H, W = mov.shape

            mov = mov.reshape(T, H // factor, factor, W // factor, factor).mean(4).mean(2)

        result = mov.astype(np.uint16)
        LOGGER.info(f"Downscaled from {self.imaging_data.shape} to {result.shape}")
        return result

    def _view_frame(
        self,
        frame: np.ndarray,
        title: str = "Frame",
        cmap: str = "gray",
    ) -> None:
        """
        Display a single frame using matplotlib.

        Args:
            frame: 2D numpy array to display.
            title: Title for the figure window.
            cmap: Matplotlib colormap name.
        """
        wf_plotting.show_frame(frame, title, cmap)

    def view_reference(self, cmap: str = "gray") -> None:
        """
        Display the green reference image.

        Args:
            cmap: Matplotlib colormap name. Default is "gray".
        """
        if self.green_reference is None:
            LOGGER.warning("Green reference not loaded. Call load_data() first.")
            return

        wf_plotting.show_frame(self.green_reference, title="Green Reference Image", cmap=cmap)

    def get_fps(self) -> float | None:
        """
        Get the camera frame rate from file attributes.

        Returns:
            Frame rate in Hz, or None if not available.
        """
        if "camera_fps" in self.file_attrs:
            fps = float(self.file_attrs["camera_fps"])
            LOGGER.info(f"Found camera_fps: {fps} Hz")
            return fps

        LOGGER.warning("camera_fps not found in file attributes")
        return None

    def get_recording_duration(self) -> float | None:
        """
        Calculate the total recording duration.

        Returns:
            Duration in seconds, or None if FPS or imaging
            data is not available.
        """
        fps = self.get_fps()
        if fps is None:
            LOGGER.warning("Cannot calculate duration: FPS not available")
            return None

        if self.imaging_data is None:
            LOGGER.warning("Cannot calculate duration: imaging data not loaded")
            return None

        n_frames = self.imaging_data.shape[0]
        duration = n_frames / fps
        LOGGER.info(f"Recording duration: {duration:.2f} seconds ({n_frames} frames @ {fps} fps)")
        return duration

    def _get_session_processed_folder(self) -> Path | None:
        """
        Get the session-level processed folder path.

        Returns:
            Path to session processed folder, or None on error.
        """
        return wf_paths.get_session_processed_folder(self.trial_path)

    def _get_processed_folder(self) -> Path | None:
        """
        Get the trial-level processed folder path.

        Returns:
            Path to trial processed folder, or None on error.
        """
        return wf_paths.get_trial_processed_folder(self.trial_path)

    def save_array(self, data: np.ndarray, name: str, file_format: str = "npy") -> Path | None:
        """
        Save a numpy array to the processed folder.

        Args:
            data: Numpy array to save.
            name: Base filename (without extension).
            file_format: Either "npy" or "npz" (compressed).

        Returns:
            Path to saved file, or None on error.
        """
        processed_folder = self._get_processed_folder()
        if processed_folder is None:
            return None

        try:
            if file_format == "npy":
                output_path = processed_folder / f"{name}.npy"
                np.save(str(output_path), data)
            elif file_format == "npz":
                output_path = processed_folder / f"{name}.npz"
                np.savez_compressed(str(output_path), data=data)
            else:
                LOGGER.error(f"Unsupported file format: {file_format}")
                return None

            LOGGER.info(f"Saved array to: {output_path}")
            LOGGER.info(f"  Shape: {data.shape}, dtype: {data.dtype}")
            return output_path

        except Exception:
            LOGGER.exception("Error saving array")
            return None

    def to_numpy(self, source: str | Path | np.ndarray | None = None) -> np.ndarray | None:
        """
        Convert a source to a numpy array.

        Args:
            source: Path to TIFF file, numpy array, or None.
                If None, returns the loaded imaging_data.

        Returns:
            Numpy array, or None if source is invalid.
        """
        if source is None:
            if self.imaging_data is not None:
                return self.imaging_data
            LOGGER.warning("No imaging data loaded and no source provided")
            return None

        return wf_io.tiff_to_numpy(source)

    def create_movie(
        self,
        data: str | Path | np.ndarray | None = None,
        output_path: Path | None = None,
        fps: int = 10,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_callback: Callable | None = None,
    ) -> Path | None:
        """
        Create an MP4 movie from imaging data.

        Args:
            data: Source data (path, array, or None for imaging_data).
            output_path: Output file path. Defaults to trial_path/movie.mp4.
            fps: Frames per second for output video.
            cmap: Matplotlib colormap name.
            vmin: Minimum value for colormap scaling.
            vmax: Maximum value for colormap scaling.
            frame_callback: Optional function called for each frame
                to add custom annotations.

        Returns:
            Path to created movie file, or None on error.
        """
        movie_data = self.to_numpy(data)
        if movie_data is None:
            return None

        if movie_data.ndim != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {movie_data.shape}")
            return None

        if output_path is None:
            output_path = self.trial_path / "movie.mp4"
        else:
            output_path = Path(output_path)

        return wf_movie.create_movie_from_array(
            data=movie_data,
            output_path=output_path,
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_callback=frame_callback,
            wf_analysis=self,
        )

    def create_mask(self, initial_radius: float = 100.0) -> dict[str, float] | None:
        """
        Create a circular mask interactively.

        Opens an interactive window to define a circular region
        of interest on the green reference image.

        Args:
            initial_radius: Starting radius in pixels.

        Returns:
            Dictionary with center_x, center_y, radius keys,
            or None if cancelled or green reference not loaded.
        """
        if self.green_reference is None:
            LOGGER.warning("Green reference not loaded. Call load_data() first.")
            return None

        return wf_plotting.create_mask_interactive(self.green_reference, initial_radius)

    def save_mask(self, mask_data: dict[str, float], name: str = "mask") -> Path | None:
        """
        Save mask parameters to the session processed folder.

        Args:
            mask_data: Dictionary with center_x, center_y, radius.
            name: Base filename (without .json extension).

        Returns:
            Path to saved JSON file, or None on error.
        """
        session_folder = self._get_session_processed_folder()
        if session_folder is None:
            return None

        output_path = session_folder / f"{name}.json"
        return wf_masks.save_mask_json(mask_data, output_path)

    def set_condition(self, condition_dict: dict[str, Any]) -> None:
        """
        Set experimental condition attributes on this instance.

        Stores the condition dictionary and also sets each key-value
        pair as an attribute on this object for convenient access.

        Args:
            condition_dict: Dictionary of condition parameters
                (e.g., stimulus_start_frame, baseline_ms).
        """
        self.condition = condition_dict.copy()

        for key, value in condition_dict.items():
            setattr(self, key, value)

        LOGGER.info(f"Set condition: {list(condition_dict.keys())}")

    def load_mask(self, name: str = "mask") -> dict[str, float] | None:
        """
        Load mask parameters from the session processed folder.

        Args:
            name: Base filename (without .json extension).

        Returns:
            Dictionary with center_x, center_y, radius keys,
            or None if file not found.
        """
        session_folder = self._get_session_processed_folder()
        if session_folder is None:
            return None

        mask_path = session_folder / f"{name}.json"
        return wf_masks.load_mask_json(mask_path)

    def apply_mask(
        self,
        data: np.ndarray | None = None,
        mask_data: dict[str, float] | None = None,
        mask_name: str = "mask",
    ) -> np.ndarray | None:
        """
        Apply a circular mask to imaging data.

        Sets pixels outside the circular region to zero.

        Args:
            data: 3D array to mask. Defaults to imaging_data.
            mask_data: Mask parameters. If None, loads from file.
            mask_name: Name of mask file to load if mask_data is None.

        Returns:
            Masked 3D array with same shape as input,
            or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if mask_data is None:
            mask_data = self.load_mask(mask_name)
            if mask_data is None:
                return None

        reference_shape = None
        if self.green_reference is not None:
            reference_shape = self.green_reference.shape[:2]

        return wf_masks.apply_circular_mask(data, mask_data, reference_shape)

    def calculate_percentile(
        self,
        data: np.ndarray | None = None,
        percentile: float = 15.0,
        stimulus_start_frame: int | None = None,
        baseline_ms: float | None = None,
        fps: float | None = None,
    ) -> np.ndarray | None:
        """
        Calculate percentile projection for baseline (F0).

        Computes the specified percentile for each pixel across time,
        typically used as the baseline for delta F/F calculations.

        Args:
            data: 3D array. Defaults to imaging_data.
            percentile: Percentile value (0-100). Default 15.
            stimulus_start_frame: If provided with baseline_ms,
                only uses frames before this point.
            baseline_ms: Duration of baseline period in ms.
            fps: Frame rate. Auto-detected if not provided.

        Returns:
            2D percentile image, or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if fps is None and stimulus_start_frame is not None and baseline_ms is not None:
            fps = self.get_fps()

        return wf_metrics.calculate_percentile_movie(
            data=data,
            percentile=percentile,
            stimulus_start_frame=stimulus_start_frame,
            baseline_ms=baseline_ms,
            fps=fps,
        )

    def calculate_deltaff(
        self, data: np.ndarray | None = None, baseline: np.ndarray | None = None
    ) -> np.ndarray | None:
        """
        Calculate delta F/F (relative fluorescence change).

        Computes (F - F0) / F0 for each pixel and frame.

        Args:
            data: 3D fluorescence data (F). Defaults to imaging_data.
            baseline: 2D baseline image (F0). Required.

        Returns:
            3D delta F/F array, or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if baseline is None:
            LOGGER.error("baseline must be provided")
            return None

        return wf_metrics.calculate_deltaff_movie(data, baseline)

    def calculate_baseline(
        self,
        data: np.ndarray | None = None,
        stimulus_start_frame: int = 0,
        baseline_ms: float = 500.0,
        fps: float | None = None,
    ) -> np.ndarray | None:
        """
        Calculate mean baseline image from pre-stimulus period.

        Averages frames within the baseline window before stimulus.

        Args:
            data: 3D array. Defaults to imaging_data.
            stimulus_start_frame: Frame index of stimulus onset.
            baseline_ms: Duration of baseline period in ms.
            fps: Frame rate. Auto-detected if not provided.

        Returns:
            2D mean baseline image, or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if fps is None:
            fps = self.get_fps()
            if fps is None:
                LOGGER.error("FPS not available and not provided")
                return None

        return wf_metrics.calculate_baseline_movie(
            data=data,
            stimulus_start_frame=stimulus_start_frame,
            baseline_ms=baseline_ms,
            fps=fps,
        )

    def set_roi(self, roi: tuple[int, int] | dict[str, Any]) -> None:
        """
        Set the region of interest for trace extraction.

        Args:
            roi: Either a tuple (x, y) or a dict with 'center' key.

        Raises:
            ValueError: If tuple doesn't have 2 elements or dict
                missing 'center' key.
            TypeError: If roi is neither tuple nor dict.
        """
        if isinstance(roi, tuple):
            if len(roi) != 2:
                raise ValueError("ROI tuple must have 2 elements (x, y)")
            self.roi = {"center": roi}
        elif isinstance(roi, dict):
            if "center" not in roi:
                raise ValueError("ROI dictionary must contain 'center' key")
            self.roi = roi.copy()
        else:
            raise TypeError(f"ROI must be tuple (x, y) or dict, got {type(roi)}")

        center = self.roi["center"]
        if self.imaging_data is not None:
            _, H, W = self.imaging_data.shape
            if center[0] < 0 or center[0] >= W or center[1] < 0 or center[1] >= H:
                LOGGER.warning(
                    f"ROI center ({center[0]}, {center[1]}) is outside image bounds ({W}, {H})"
                )

        LOGGER.info(f"ROI set: center=({center[0]}, {center[1]})")

    def calculate_percentile_centroid_roi(
        self, data: np.ndarray, percentile: float = 95.0
    ) -> tuple[int, int]:
        """
        Find ROI center from high-intensity pixels.

        Computes the centroid of pixels above the specified
        percentile threshold.

        Args:
            data: 2D image array.
            percentile: Threshold percentile (0-100). Default 95.

        Returns:
            Tuple (x, y) of centroid coordinates.
        """
        return wf_roi.centroid_from_percentile(data, percentile)

    def calculate_trace_within_roi(
        self,
        data: np.ndarray | None = None,
        roi: tuple[int, int] | dict[str, Any] | None = None,
        diameter: float = 50.0,
    ) -> np.ndarray | None:
        """
        Extract mean fluorescence trace from circular ROI.

        Computes the mean pixel value within a circular region
        for each frame.

        Args:
            data: 3D array. Defaults to imaging_data.
            roi: ROI center as tuple (x, y) or dict with 'center'.
                Defaults to self.roi.
            diameter: ROI diameter in pixels. Default 50.

        Returns:
            1D array of mean values per frame, or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if roi is None:
            if self.roi is None:
                LOGGER.error("No ROI provided and self.roi not set")
                return None
            roi = self.roi

        if isinstance(roi, dict):
            center = roi.get("center")
            if center is None:
                LOGGER.error("ROI dictionary must contain 'center' key")
                return None
        elif isinstance(roi, tuple):
            center = roi
        else:
            LOGGER.error(f"ROI must be tuple (x, y) or dict, got {type(roi)}")
            return None

        return wf_roi.trace_within_circular_roi(data, center, diameter)

    def close(self) -> None:
        """
        Release resources and clear loaded data.

        Sets imaging_data, green_reference, timestamps, and
        sensor_data to None/empty to free memory.
        """
        self.imaging_data = None
        self.green_reference = None
        self.timestamps = None
        self.sensor_data = {}
        LOGGER.debug("Resources cleaned up")
