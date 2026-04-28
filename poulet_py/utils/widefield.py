import re
from abc import ABC, abstractmethod
from datetime import datetime

try:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import h5py
    from numpy import array, ceil, load, ndarray, pad, save, savez_compressed, uint16
    from pandas import DataFrame, read_csv
    from pydantic import BaseModel, Field, PrivateAttr
    from skimage.io import imread

    from poulet_py import LOGGER

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
    def _open(self): ...


# find better name # CHECK
class DataStructureV1(BaseData):
    path: Path = Field(..., description="Path to trial folder")
    imaging_path: Path | None = Field(default=None)
    timestamps_path: Path | None = Field(default=None)
    analog_output_data_path: Path | None = Field(default=None)
    reference_image_path: Path | None = Field(default=None)

    # similar to all private attrs # CHECK
    _imaging_data: ndarray[Any, Any] | None = PrivateAttr(default=None)
    _green_reference: ndarray[Any, Any] | None = PrivateAttr(default=None)
    _timestamps: DataFrame | None = PrivateAttr(default=None)
    _analog_output_data: dict[str, ndarray[Any, Any]] = PrivateAttr(default_factory=dict)
    _analog_output_data_attrs: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _analog_output_data_file_attrs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _condition: dict[str, Any] | None = PrivateAttr(default=None)
    _roi: dict[str, Any] | None = PrivateAttr(default=None)

    def trial_open_filter(self, start: datetime | int, end: datetime | int) -> bool:  # CHECK
        if isinstance(start, datetime) and isinstance(end, datetime):
            folder_time = self._folder_datetime()
            return folder_time is not None and start <= folder_time <= end

        if isinstance(start, int) and isinstance(end, int):
            trial_number = self._folder_trial_number()
            return trial_number is not None and start <= trial_number <= end

        msg = "start and end must both be datetime values or both be integer trial numbers"
        raise TypeError(msg)

    def _folder_datetime(self) -> datetime | None:
        folder_name = self.path.name
        for date_format in ("%y%m%d_%H%M%S", "%Y%m%d_%H%M%S"):
            try:
                return datetime.strptime(folder_name, date_format)
            except ValueError:
                continue
        return None

    def _folder_trial_number(self) -> int | None:
        match = re.fullmatch(r"(?:trial[_-]?)?0*(\d+)", self.path.name, flags=re.IGNORECASE)
        if match is None:
            return None
        return int(match.group(1))

    @property
    def imaging_data(self):
        return self._imaging_data

    @property
    def green_reference(self):
        return self._green_reference

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def analog_output_data(self):
        return self._analog_output_data

    @property
    def analog_output_data_attrs(self):
        return self._analog_output_data_attrs

    @property
    def analog_output_data_file_attrs(self):
        return self._analog_output_data_file_attrs

    @property
    def condition(self):
        return self._condition

    @property
    def roi(self):
        return self._roi

    def _resolve_paths(self) -> None:
        if not self.path.exists():
            msg = f"Trial path does not exist: {self.path}"
            raise FileNotFoundError(msg)
        if not self.path.is_dir():
            msg = f"Trial path must be a directory: {self.path}"
            raise ValueError(msg)

        video_candidates = ["recording.tiff", "recording.tif", "recording.npy"]
        self.imaging_path = None
        for candidate in video_candidates:
            candidate_path = self.path / candidate
            if candidate_path.exists():
                self.imaging_path = candidate_path
                break

        if self.imaging_path is None:
            msg = f"recording.tiff/.tif/.npy not found in: {self.path}"
            raise ValueError(msg)

        csv_path = self.path / "recording.csv"
        h5_path = self.path / "data.h5"
        green_path = self.path / "green.tiff"
        self.timestamps_path = csv_path if csv_path.exists() else None
        self.analog_output_data_path = h5_path if h5_path.exists() else None
        self.reference_image_path = green_path if green_path.exists() else None

    def open(self) -> None:
        self._resolve_paths()
        self._imaging_data = self._open_imaging()
        self._green_reference = self._open_reference_image()
        self._timestamps = self._open_timestamps()
        (
            self._analog_output_data,
            self._analog_output_data_attrs,
            self._analog_output_data_file_attrs,
        ) = self._open_analog_output()

    # move all these to separated data or whatever u wannt calla it check commit 4031c9e # CHECK
    def _open_imaging(self) -> ndarray[Any, Any]:
        if self.imaging_path is None:
            raise ValueError("Imaging path is not set")
        LOGGER.info(f"Opening imaging data from: {self.imaging_path.name}")

        if self.imaging_path.suffix.lower() == ".npy":
            data = load(str(self.imaging_path))
        elif self.imaging_path.suffix.lower() in (".tiff", ".tif"):
            data = imread(str(self.imaging_path))
        else:
            msg = f"Unsupported imaging format: {self.imaging_path.suffix}"
            raise ValueError(msg)

        LOGGER.info(f"Opened imaging stack: {data.shape}")
        return data

    def _open_reference_image(self) -> ndarray[Any, Any] | None:
        if self.reference_image_path is None:
            return None
        if not self.reference_image_path.exists():
            LOGGER.warning(f"Green reference not found: {self.reference_image_path}")
            return None

        LOGGER.info(f"Opening green reference from: {self.reference_image_path.name}")
        image = imread(str(self.reference_image_path))
        if image.ndim == 3:
            image = image[0]
        LOGGER.info(f"Opened green reference: {image.shape}")
        return image

    def _open_timestamps(self) -> DataFrame | None:
        """
        Open frame timestamps from a CSV file.

        Reads the semicolon-separated CSV file containing timing
        information for each frame in the recording.
        """
        if self.timestamps_path is None or not self.timestamps_path.exists():
            LOGGER.warning(f"Timestamps path not set or not found: {self.timestamps_path}")
            return None
        try:
            timestamps = read_csv(self.timestamps_path, sep=";")
            timestamps = timestamps.loc[:, ~timestamps.columns.str.contains("^Unnamed")]
            LOGGER.info(f"Opened timestamps: {len(timestamps)} rows")
        except Exception:
            LOGGER.exception(f"Error loading timestamps: {self.timestamps_path}")
            return None
        return timestamps

    def _open_analog_output(
        self,
    ) -> tuple[dict[str, ndarray[Any, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        """
        Open sensor data from an HDF5 file.

        Returns empty dictionaries if the file does not exist.
        """

        analog_output_data: dict[str, ndarray[Any, Any]] = {}
        analog_output_data_attrs: dict[str, dict[str, Any]] = {}
        analog_output_data_file_attrs: dict[str, Any] = {}

        if self.analog_output_data_path is None or not self.analog_output_data_path.exists():
            LOGGER.warning(
                f"Analog output data path not set or not found: {self.analog_output_data_path}"
            )
            return {}, {}, {}
        try:
            with h5py.File(self.analog_output_data_path, "r") as f:
                analog_output_data_file_attrs = dict(f.attrs)

                def _visit_datasets(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        analog_output_data[name] = array(obj)
                        analog_output_data_attrs[name] = dict(obj.attrs)

                f.visititems(_visit_datasets)
        except Exception:
            LOGGER.exception(f"Error loading H5: {self.analog_output_data_path}")
            return {}, {}, {}
        return analog_output_data, analog_output_data_attrs, analog_output_data_file_attrs

    def close(self) -> None:
        self._imaging_data = None
        self._green_reference = None
        self._timestamps = None
        self._analog_output_data = {}
        self._analog_output_data_attrs = {}
        self._analog_output_data_file_attrs = {}

    def summary(self) -> str:
        """Return a human-readable summary of this trial."""
        lines = ["=" * 60, f"Trial: {self.path.name}", "=" * 60]

        if self.imaging_data is not None:
            n_frames, height, width = self.imaging_data.shape
            lines.extend(
                [
                    "Imaging data:",
                    f"  Shape: {self.imaging_data.shape}",
                    f"  Frames: {n_frames}",
                    f"  Resolution: {width} x {height}",
                    f"  Dtype: {self.imaging_data.dtype}",
                    (f"  Value range: [{self.imaging_data.min()}, {self.imaging_data.max()}]"),
                ]
            )
            size_mb = self.imaging_data.nbytes / (1024 * 1024)
            lines.append(f"  Memory: {size_mb:.1f} MB")

        if self.timestamps is not None:
            lines.extend(
                [
                    "Timestamps:",
                    f"  Rows: {len(self.timestamps)}",
                    f"  Columns: {list(self.timestamps.columns)}",
                ]
            )

        if self.analog_output_data:
            lines.append("Analog output data:")
            for name, data in self.analog_output_data.items():
                attrs = self.analog_output_data_attrs.get(name, {})
                sr = attrs.get("sr", "unknown")
                lines.append(f"  {name}: shape={data.shape}, sr={sr} Hz")

        if self.analog_output_data_file_attrs:
            mouse_id = self.analog_output_data_file_attrs.get("mouse_id", "unknown")
            protocol = self.analog_output_data_file_attrs.get("protocol_name", "unknown")
            comment = self.analog_output_data_file_attrs.get("comment", "")
            lines.extend(["Metadata:", f"  Mouse: {mouse_id}", f"  Protocol: {protocol}"])
            if comment:
                lines.append(f"  Comment: {comment}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class Trial(DataStructureV1):
    # make restriction of tiff, npy only etc.
    path: Path = Field(..., description="Path to the trial folder")
    data: BaseData = Field(...)

    def open(self) -> None:
        self.data.open()


class Session(BaseModel):
    path: Path = Field(..., description="Path to the session folder")
    start: datetime | int = Field()  # time or trial number and we see further # CHECK
    end: datetime | int = Field()

    _trials: list[Trial] = PrivateAttr(default_factory=list)

    @property
    def trials(self) -> list[Trial]:
        return self._trials

    def add_trial(self, trial: Trial) -> None:
        self._trials.append(trial)

    def open(
        self,
        start: datetime | int | None = None,
        end: datetime | int | None = None,
    ) -> None:
        start = self.start if start is None else start
        end = self.end if end is None else end
        for trial in self._trials:
            if trial.trial_open_filter(start, end):
                trial.open()

    def close(self) -> None:
        for trial in self._trials:
            trial.close()


class WidefieldAnalysis(BaseModel):
    session: Session = Field(..., description="")
    _active_trial_idx: int = PrivateAttr(default=0)

    @property
    def active_trial(self) -> Trial:
        if not self.session.trials:
            msg = "Session has no trials configured"
            raise ValueError(msg)
        return self.session.trials[self._active_trial_idx]

    @staticmethod
    def _is_trial_folder(path: Path) -> bool:
        if not path.is_dir():
            return False
        return any(
            (path / name).exists() for name in ("recording.tiff", "recording.tif", "recording.npy")
        )

    @classmethod
    def from_trial_path(cls, path: Path | str, load: bool = True) -> "WidefieldAnalysis":
        path = Path(path)
        now = datetime.now()
        session = Session(path=path.parent, start=now, end=now)
        wf = cls(session=session)
        if load:
            wf.load(path)
        else:
            wf.session.add_trial(Trial(path=path))
        return wf

    @classmethod
    def from_session_path(cls, path: Path | str, load: bool = True) -> "WidefieldAnalysis":
        path = Path(path)
        now = datetime.now()
        session = Session(path=path, start=now, end=now)
        wf = cls(session=session)
        if load:
            wf.load(path)
        elif path.exists() and path.is_dir():
            for child in sorted(path.iterdir()):
                if cls._is_trial_folder(child):
                    wf.session.add_trial(Trial(path=child))
        return wf

    def load(self, path: Path | str | None = None) -> None:
        """
        Load trial data using a single entrypoint.

        Behavior:
        - path is a trial folder: add/select that trial and load it.
        - path is a session folder: discover all trial folders and load them.
        - path is None:
          - if trials already exist, load active trial when there is one trial,
            or load all registered trials when there are multiple.
          - if no trials exist, infer from self.session.path (trial vs session folder).
        """
        target = Path(path) if path is not None else self.session.path

        if self._is_trial_folder(target):
            for idx, trial in enumerate(self.session.trials):
                if trial.path == target:
                    self._active_trial_idx = idx
                    trial.open()
                    LOGGER.info(str(trial))
                    return
            trial = Trial(path=target)
            self.session.add_trial(trial)
            self._active_trial_idx = len(self.session.trials) - 1
            trial.open()
            LOGGER.info(str(trial))
            return

        if not target.exists() or not target.is_dir():
            msg = f"Path does not exist or is not a directory: {target}"
            raise ValueError(msg)

        if path is None and self.session.trials:
            if len(self.session.trials) == 1:
                self.active_trial.open()
                LOGGER.info(str(self.active_trial))
                return
            for trial in self.session.trials:
                trial.open()
            return

        discovered_trials = [
            child for child in sorted(target.iterdir()) if self._is_trial_folder(child)
        ]
        if not discovered_trials:
            msg = f"No trial folders found in: {target}"
            raise ValueError(msg)

        for trial_path in discovered_trials:
            existing = next((t for t in self.session.trials if t.path == trial_path), None)
            trial = existing if existing is not None else Trial(path=trial_path)
            if existing is None:
                self.session.add_trial(trial)
            trial.open()

    ###### TO INTEGRATE WITH THE NEW CODEBASE ######
    def downscale(
        self,
        target_resolution: tuple[int, int] | None = None,
        factor: int | None = None,
    ) -> ndarray[Any, Any] | None:
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
                factor_H_int = int(ceil(factor_H))
                factor_W_int = int(ceil(factor_W))
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
                mov = pad(mov, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0)
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
                mov = pad(mov, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0)
                T, H, W = mov.shape

            mov = mov.reshape(T, H // factor, factor, W // factor, factor).mean(4).mean(2)

        result = mov.astype(uint16)
        LOGGER.info(f"Downscaled from {self.imaging_data.shape} to {result.shape}")
        return result

    def _view_frame(
        self,
        frame: ndarray[Any, Any],
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
        if "camera_fps" in self.analog_output_data_file_attrs:
            fps = float(self.analog_output_data_file_attrs["camera_fps"])
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

    def save_array(
        self, data: ndarray[Any, Any], name: str, file_format: str = "npy"
    ) -> Path | None:
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
                save(str(output_path), data)
            elif file_format == "npz":
                output_path = processed_folder / f"{name}.npz"
                savez_compressed(str(output_path), data=data)
            else:
                LOGGER.error(f"Unsupported file format: {file_format}")
                return None

            LOGGER.info(f"Saved array to: {output_path}")
            LOGGER.info(f"  Shape: {data.shape}, dtype: {data.dtype}")
            return output_path

        except Exception:
            LOGGER.exception("Error saving array")
            return None

    def to_numpy(
        self, source: str | Path | ndarray[Any, Any] | None = None
    ) -> ndarray[Any, Any] | None:
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
        data: str | Path | ndarray[Any, Any] | None = None,
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
        data: ndarray[Any, Any] | None = None,
        mask_data: dict[str, float] | None = None,
        mask_name: str = "mask",
    ) -> ndarray[Any, Any] | None:
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
        data: ndarray[Any, Any] | None = None,
        percentile: float = 15.0,
        stimulus_start_frame: int | None = None,
        baseline_ms: float | None = None,
        fps: float | None = None,
    ) -> ndarray[Any, Any] | None:
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
        self, data: ndarray[Any, Any] | None = None, baseline: ndarray[Any, Any] | None = None
    ) -> ndarray[Any, Any] | None:
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
        data: ndarray[Any, Any] | None = None,
        stimulus_start_frame: int = 0,
        baseline_ms: float = 500.0,
        fps: float | None = None,
    ) -> ndarray[Any, Any] | None:
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
        self, data: ndarray[Any, Any], percentile: float = 95.0
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
        data: ndarray[Any, Any] | None = None,
        roi: tuple[int, int] | dict[str, Any] | None = None,
        diameter: float = 50.0,
    ) -> ndarray[Any, Any] | None:
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
