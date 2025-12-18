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

try:
    import json
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Dict, Optional, Tuple, Union

    import h5py
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from skimage import io as skio

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


class WidefieldAnalysis:
    """
    Analysis class for widefield imaging TIFF data.

    Loads and provides access to widefield recording trial data including
    the imaging stack (TIFF), timestamps (CSV), and sensor data (H5).

    Attributes:
        trial_path: Path to the trial folder.
        imaging_data: Loaded TIFF imaging stack as numpy array.
        timestamps: DataFrame with frame timing information.
        sensor_data: Dictionary of sensor traces from H5 file.
    """

    def __init__(self, trial_path: Path):
        """
        Initialize the analyzer with a trial folder path.

        Args:
            trial_path: Path to the trial folder containing recording files.

        Raises:
            FileNotFoundError: If the trial path does not exist.
            ValueError: If required files are missing.
        """
        self.trial_path = Path(trial_path)

        self.tiff_path: Path | None = None
        self.csv_path: Path | None = None
        self.h5_path: Path | None = None
        self.green_path: Path | None = None

        self.imaging_data: np.ndarray | None = None
        self.green_reference: np.ndarray | None = None
        self.timestamps: pd.DataFrame | None = None
        self.sensor_data: dict[str, np.ndarray] = {}
        self.sensor_attrs: dict[str, dict[str, Any]] = {}
        self.file_attrs: dict[str, Any] = {}
        self.condition: dict[str, Any] | None = None
        self.roi: dict[str, Any] | None = None

        self._validate()

        LOGGER.info(f"Initialized WidefieldAnalysis for: {self.trial_path.name}")

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

        self.tiff_path = self.trial_path / "recording.tiff"
        self.csv_path = self.trial_path / "recording.csv"
        self.h5_path = self.trial_path / "data.h5"
        self.green_path = self.trial_path / "green.tiff"

        if not self.tiff_path.exists():
            msg = f"recording.tiff not found in: {self.trial_path}"
            raise ValueError(msg)

    def load_data(self) -> None:
        """
        Load all trial data: imaging stack, timestamps, and sensor data.

        Raises:
            IOError: If files cannot be opened or read.
        """
        self._load_imaging()
        self._load_green_reference()
        self._load_timestamps()
        self._load_sensors()
        self._print_info()

    def _load_imaging(self) -> None:
        """Load the TIFF imaging stack."""
        try:
            LOGGER.info(f"Loading imaging data from: {self.tiff_path.name}")
            self.imaging_data = skio.imread(str(self.tiff_path))
            LOGGER.info(f"Loaded imaging stack: {self.imaging_data.shape}")
        except Exception:
            LOGGER.exception(f"Error loading TIFF: {self.tiff_path}")
            raise

    def _load_green_reference(self) -> None:
        """Load the green reference image."""
        if not self.green_path.exists():
            LOGGER.warning(f"Green reference not found: {self.green_path}")
            return

        try:
            LOGGER.info(f"Loading green reference from: {self.green_path.name}")
            self.green_reference = skio.imread(str(self.green_path))
            if len(self.green_reference.shape) == 3:
                self.green_reference = self.green_reference[0]
            LOGGER.info(f"Loaded green reference: {self.green_reference.shape}")
        except Exception:
            LOGGER.exception(f"Error loading green reference: {self.green_path}")

    def _load_timestamps(self) -> None:
        """Load the CSV timestamp file."""
        if not self.csv_path.exists():
            LOGGER.warning(f"CSV not found: {self.csv_path}")
            return

        try:
            self.timestamps = pd.read_csv(self.csv_path, sep=";")
            self.timestamps = self.timestamps.loc[
                :, ~self.timestamps.columns.str.contains("^Unnamed")
            ]
            LOGGER.info(f"Loaded timestamps: {len(self.timestamps)} rows")
        except Exception:
            LOGGER.exception(f"Error loading CSV: {self.csv_path}")

    def _load_sensors(self) -> None:
        """Load the H5 sensor data."""
        if not self.h5_path.exists():
            LOGGER.warning(f"H5 not found: {self.h5_path}")
            return

        try:
            with h5py.File(self.h5_path, "r") as f:
                self.file_attrs = dict(f.attrs)

                def _visit_datasets(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        self.sensor_data[name] = np.array(obj)
                        self.sensor_attrs[name] = dict(obj.attrs)

                f.visititems(_visit_datasets)

            LOGGER.info(f"Loaded {len(self.sensor_data)} sensor traces from H5")
        except Exception:
            LOGGER.exception(f"Error loading H5: {self.h5_path}")

    def _print_info(self) -> None:
        """Print information about the loaded data."""
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
        Downscale the imaging data using block averaging.

        Supports two modes:
        1. Target resolution: Specify desired output (H, W)
        2. Factor: Specify downscale factor (e.g., 2 means 1024→512)

        Args:
            target_resolution: Target (height, width) in pixels.
                              Calculates factor automatically.
            factor: Downscale factor (e.g., 2 means 1024→512).
                    If both specified, target_resolution takes precedence.

        Returns:
            Downscaled movie array, or None if imaging data not loaded.

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
        Internal function to visualize a single frame.

        Args:
            frame: 2D numpy array representing the frame.
            title: Title for the plot.
            cmap: Colormap to use for visualization.
        """
        try:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(frame, cmap=cmap)
            ax.set_title(title, fontsize=14)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            LOGGER.info(f"Displayed frame: {title}, shape: {frame.shape}")
        except Exception:
            LOGGER.exception(f"Error displaying frame: {title}")

    def view_reference(self, cmap: str = "gray") -> None:
        """
        Visualize the green reference image.

        Args:
            cmap: Colormap to use for visualization (default: "gray").
        """
        if self.green_reference is None:
            LOGGER.warning("Green reference not loaded. Call load_data() first.")
            return

        self._view_frame(self.green_reference, title="Green Reference Image", cmap=cmap)

    def get_fps(self) -> float | None:
        """
        Extract frame rate from H5 file attributes.

        Returns:
            Frame rate in fps, or None if not found.
        """
        if "camera_fps" in self.file_attrs:
            fps = float(self.file_attrs["camera_fps"])
            LOGGER.info(f"Found camera_fps: {fps} Hz")
            return fps

        LOGGER.warning("camera_fps not found in file attributes")
        return None

    def get_recording_duration(self) -> float | None:
        """
        Calculate recording duration from frames and FPS.

        Returns:
            Duration in seconds, or None if cannot calculate.
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
        Get or create the processed data folder at session level.

        Creates folder structure: data/processed/[session]/
        where [session] is the folder name after "raw".

        Returns:
            Path to session processed folder, or None if error.
        """
        try:
            trial_path = Path(self.trial_path)
            parts = trial_path.parts

            raw_idx = None
            for i, part in enumerate(parts):
                if part == "raw":
                    raw_idx = i
                    break

            if raw_idx is None:
                LOGGER.error("Could not find 'raw' in trial path structure")
                return None

            session_folder = parts[raw_idx + 1]

            data_folder = trial_path
            for _ in range(len(parts) - raw_idx - 1):
                data_folder = data_folder.parent

            session_processed_folder = data_folder.parent / "processed" / session_folder

            session_processed_folder.mkdir(parents=True, exist_ok=True)
            return session_processed_folder

        except Exception:
            LOGGER.exception("Error creating session processed folder")
            return None

    def _get_processed_folder(self) -> Path | None:
        """
        Get or create the processed data folder for this trial.

        Creates folder structure: data/processed/[session]/trials/[trial]/
        where [session] is the folder name after "raw" and [trial] is the
        trial folder name.

        Returns:
            Path to processed folder, or None if error.
        """
        try:
            trial_path = Path(self.trial_path)
            parts = trial_path.parts

            raw_idx = None
            for i, part in enumerate(parts):
                if part == "raw":
                    raw_idx = i
                    break

            if raw_idx is None:
                LOGGER.error("Could not find 'raw' in trial path structure")
                return None

            session_folder = parts[raw_idx + 1]
            trial_folder = parts[-1]

            data_folder = trial_path
            for _ in range(len(parts) - raw_idx - 1):
                data_folder = data_folder.parent

            processed_folder = (
                data_folder.parent / "processed" / session_folder / "trials" / trial_folder
            )

            processed_folder.mkdir(parents=True, exist_ok=True)
            return processed_folder

        except Exception:
            LOGGER.exception("Error creating processed folder")
            return None

    def save_array(self, data: np.ndarray, name: str, file_format: str = "npy") -> Path | None:
        """
        Save numpy array to processed data folder.

        Args:
            data: Numpy array to save.
            name: Name for the saved file (without extension).
            file_format: File format to save ("npy" for .npy, "npz" for .npz).

        Returns:
            Path to saved file, or None if error.
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
        Convert TIFF file or return numpy array as-is.

        Args:
            source: Path to TIFF file, numpy array, or None to use
                   loaded imaging_data.

        Returns:
            Numpy array of the imaging data, or None if error.
        """
        if source is None:
            if self.imaging_data is not None:
                return self.imaging_data
            LOGGER.warning("No imaging data loaded and no source provided")
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
        Create an MP4 movie from TIFF file or numpy array.

        Args:
            data: TIFF file path, numpy array, or None to use loaded
                 imaging_data.
            output_path: Path to save MP4 file. If None, saves in trial
                        folder with default name.
            fps: Frames per second for the movie (default: 10).
            cmap: Colormap for visualization (default: "gray").
            vmin: Minimum value for colormap scaling. If None, uses data min.
            vmax: Maximum value for colormap scaling. If None, uses data max.
            frame_callback: Optional function to customize each frame.
                          Called with (fig, ax, frame_idx, frame_data, wf_analysis).
                          Can be used to add subplots, traces, annotations, etc.

        Returns:
            Path to saved MP4 file, or None if error.

        Callback function signature:
            def frame_callback(
                fig: matplotlib.figure.Figure,
                ax: matplotlib.axes.Axes,
                frame_idx: int,
                frame_data: np.ndarray,
                wf_analysis: WidefieldAnalysis
            ) -> None:
                # Customize frame: add subplots, plot traces, add annotations, etc.
                pass
        """
        movie_data = self.to_numpy(data)
        if movie_data is None:
            return None

        if len(movie_data.shape) != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {movie_data.shape}")
            return None

        T, H, W = movie_data.shape
        LOGGER.info(f"Creating movie from {T} frames ({H}x{W})")

        if output_path is None:
            output_path = self.trial_path / "movie.mp4"
        else:
            output_path = Path(output_path)

        if vmin is None:
            vmin = float(movie_data.min())
        if vmax is None:
            vmax = float(movie_data.max())

        try:
            from io import BytesIO

            frames = []
            for frame_idx in range(T):
                frame = movie_data[frame_idx]

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"Frame {frame_idx + 1}/{T}", fontsize=14)
                ax.axis("off")

                if frame_callback is not None:
                    frame_callback(fig, ax, frame_idx, frame, self)
                else:
                    plt.tight_layout()

                buf = BytesIO()
                if frame_callback is not None:
                    fig.savefig(buf, format="png", dpi=100, bbox_inches=None, pad_inches=0.0)
                else:
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                frame_img = skio.imread(buf)
                frames.append(frame_img)
                buf.close()
                plt.close(fig)

            LOGGER.info(f"Saving movie to: {output_path}")
            writer = imageio.get_writer(str(output_path), format="FFMPEG", fps=fps, codec="libx264")
            for frame in frames:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                writer.append_data(frame)
            writer.close()
            LOGGER.info(f"Movie saved: {output_path}")

            return output_path

        except Exception:
            LOGGER.exception("Error creating movie")
            return None

    def _get_session_processed_folder(self) -> Path | None:
        """
        Get or create the processed data folder at session level.

        Creates folder structure: data/processed/[session]/
        where [session] is the folder name after "raw".

        Returns:
            Path to session processed folder, or None if error.
        """
        try:
            trial_path = Path(self.trial_path)
            parts = trial_path.parts

            raw_idx = None
            for i, part in enumerate(parts):
                if part == "raw":
                    raw_idx = i
                    break

            if raw_idx is None:
                LOGGER.error("Could not find 'raw' in trial path structure")
                return None

            session_folder = parts[raw_idx + 1]

            data_folder = trial_path
            for _ in range(len(parts) - raw_idx - 1):
                data_folder = data_folder.parent

            session_processed_folder = data_folder.parent / "processed" / session_folder

            session_processed_folder.mkdir(parents=True, exist_ok=True)
            return session_processed_folder

        except Exception:
            LOGGER.exception("Error creating session processed folder")
            return None

    def create_mask(self, initial_radius: float = 100.0) -> dict[str, float] | None:
        """
        Interactive mask creation tool.

        Displays the green reference image and allows user to:
        - Click to set circle center
        - Press 'B' to increase radius
        - Press 'S' to decrease radius
        - Press 'Enter' to confirm and save

        Args:
            initial_radius: Initial radius in pixels (default: 100.0).

        Returns:
            Dictionary with 'center_x', 'center_y', and 'radius', or None.
        """
        if self.green_reference is None:
            LOGGER.warning("Green reference not loaded. Call load_data() first.")
            return None

        center = [None, None]
        radius = initial_radius

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.green_reference, cmap="gray")
        ax.set_title("Click to set center | B=bigger, S=smaller | Enter=confirm", fontsize=12)
        ax.axis("off")

        circle = None

        def update_circle():
            nonlocal circle
            if circle:
                circle.remove()
            if center[0] is not None and center[1] is not None:
                circle = plt.Circle(
                    (center[1], center[0]), radius, fill=False, color="red", linewidth=2
                )
                ax.add_patch(circle)
                fig.canvas.draw()

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                center[0] = int(event.ydata)
                center[1] = int(event.xdata)
                LOGGER.info(f"Center set to: ({center[1]}, {center[0]})")
                update_circle()

        def on_key(event):
            nonlocal radius
            if event.key == "b" or event.key == "B":
                radius += 10
                LOGGER.info(f"Radius increased to: {radius:.1f}")
                update_circle()
            elif event.key == "s" or event.key == "S":
                radius = max(10, radius - 10)
                LOGGER.info(f"Radius decreased to: {radius:.1f}")
                update_circle()
            elif event.key == "enter":
                if center[0] is not None and center[1] is not None:
                    plt.close(fig)
                else:
                    LOGGER.warning("Please click to set center first")

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.tight_layout()
        plt.show()

        if center[0] is not None and center[1] is not None:
            mask_data = {
                "center_x": float(center[1]),
                "center_y": float(center[0]),
                "radius": float(radius),
            }
            LOGGER.info(
                f"Mask created: center=({mask_data['center_x']}, "
                f"{mask_data['center_y']}), radius={mask_data['radius']}"
            )
            return mask_data

        LOGGER.warning("No mask created")
        return None

    def save_mask(self, mask_data: dict[str, float], name: str = "mask") -> Path | None:
        """
        Save mask data to session-level processed folder as JSON.

        Args:
            mask_data: Dictionary with 'center_x', 'center_y', and 'radius'.
            name: Name for the mask file (without extension).

        Returns:
            Path to saved JSON file, or None if error.
        """
        session_folder = self._get_session_processed_folder()
        if session_folder is None:
            return None

        try:
            output_path = session_folder / f"{name}.json"

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

    def set_condition(self, condition_dict: dict[str, Any]) -> None:
        """
        Set condition information from a dictionary.

        Loops over the dictionary and sets each key-value pair as an attribute
        on the class instance. Also stores the full dictionary in self.condition.

        Args:
            condition_dict: Dictionary with condition information.
                           Keys will be set as attributes on the instance.
        """
        self.condition = condition_dict.copy()

        for key, value in condition_dict.items():
            setattr(self, key, value)

        LOGGER.info(f"Set condition: {list(condition_dict.keys())}")

    def load_mask(self, name: str = "mask") -> dict[str, float] | None:
        """
        Load mask data from session-level processed folder.

        Args:
            name: Name of the mask file (without extension).

        Returns:
            Dictionary with mask data, or None if not found.
        """
        session_folder = self._get_session_processed_folder()
        if session_folder is None:
            return None

        mask_path = session_folder / f"{name}.json"
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

    def apply_mask(
        self,
        data: np.ndarray | None = None,
        mask_data: dict[str, float] | None = None,
        mask_name: str = "mask",
    ) -> np.ndarray | None:
        """
        Apply circular mask to imaging data.

        Sets pixels outside the mask to 0 (black).

        Args:
            data: Numpy array to mask. If None, uses loaded imaging_data.
            mask_data: Mask dictionary with 'center_x', 'center_y', 'radius'.
                      If None, loads from saved mask file.
            mask_name: Name of mask file to load if mask_data not provided.

        Returns:
            Masked numpy array, or None if error.
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
                ref_shape = self.green_reference.shape if self.green_reference is not None else None
                if ref_shape is not None and len(ref_shape) >= 2:
                    scale_x = W / ref_shape[1]
                    scale_y = H / ref_shape[0]
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

    def calculate_percentile(
        self,
        data: np.ndarray | None = None,
        percentile: float = 15.0,
        stimulus_start_frame: int | None = None,
        baseline_ms: float | None = None,
        fps: float | None = None,
    ) -> np.ndarray | None:
        """
        Calculate percentile value for each pixel across frames.

        Computes the percentile of F for each pixel across the time dimension.
        Can be restricted to a pre-stimulus window if parameters are provided.

        Args:
            data: Numpy array with shape (T, H, W). If None, uses loaded
                 imaging_data.
            percentile: Percentile value to calculate (default: 15.0).
            stimulus_start_frame: Optional frame index where stimulus starts.
                                If provided with baseline_ms, restricts
                                calculation to pre-stimulus window.
            baseline_ms: Optional milliseconds before stimulus to use for
                        window. Requires stimulus_start_frame.
            fps: Optional frame rate in Hz for time conversion. If None and
                window parameters provided, uses get_fps().

        Returns:
            2D array with percentile values for each pixel, or None if error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if len(data.shape) != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
            return None

        try:
            T, H, W = data.shape

            window_data = data
            window_info = "all frames"

            if stimulus_start_frame is not None and baseline_ms is not None:
                if fps is None:
                    fps = self.get_fps()
                    if fps is None:
                        LOGGER.error("FPS not available and not provided for window calculation")
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
                    LOGGER.error(
                        f"Invalid baseline window: start={baseline_start}, end={baseline_end}"
                    )
                    return None

                window_data = data[baseline_start:baseline_end]
                window_info = (
                    f"frames [{baseline_start}:{baseline_end}] ({len(window_data)} frames)"
                )

            LOGGER.info(f"Calculating {percentile}th percentile for {window_info} ({H}x{W})")

            f_base = np.zeros((H, W), dtype=data.dtype)

            for r in range(H):
                for c in range(W):
                    trace = window_data[:, r, c]
                    trace_base = np.percentile(trace, percentile)
                    if trace_base == 0:
                        trace_base = 1
                    f_base[r, c] = trace_base

            LOGGER.info(
                f"Percentile calculated: min={f_base.min():.2f}, "
                f"max={f_base.max():.2f}, mean={f_base.mean():.2f}"
            )
            return f_base

        except Exception:
            LOGGER.exception("Error calculating movie percentile")
            return None

    def calculate_deltaff(
        self, data: np.ndarray | None = None, baseline: np.ndarray | None = None
    ) -> np.ndarray | None:
        """
        Calculate delta F over F (ΔF/F) for a movie.

        Uses provided baseline (F0). Formula: ΔF/F = (F - F0) / F0 = F/F0 - 1

        Args:
            data: Numpy array with shape (T, H, W). If None, uses loaded
                 imaging_data.
            baseline: 2D array with shape (H, W) to use as baseline (F0).
                     Can be obtained from calculate_percentile().

        Returns:
            3D array with delta F over F values, or None if error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if len(data.shape) != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
            return None

        if baseline is None:
            LOGGER.error("baseline must be provided")
            return None

        if len(baseline.shape) != 2:
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

        try:
            dff = data / baseline - 1

            LOGGER.info(
                f"Calculated ΔF/F: shape={dff.shape}, "
                f"min={dff.min():.3f}, max={dff.max():.3f}, "
                f"mean={dff.mean():.3f}"
            )
            return dff

        except Exception:
            LOGGER.exception("Error calculating delta F over F")
            return None

    def calculate_baseline(
        self,
        data: np.ndarray | None = None,
        stimulus_start_frame: int = 0,
        baseline_ms: float = 500.0,
        fps: float | None = None,
    ) -> np.ndarray | None:
        """
        Calculate baseline mean from pre-stimulus window.

        Calculates mean of baseline period (before stimulus) for each pixel.
        Returns 2D baseline array that can be used for ΔF/F calculation.

        Args:
            data: Numpy array with shape (T, H, W). If None, uses loaded
                 imaging_data.
            stimulus_start_frame: Frame index where stimulus starts.
            baseline_ms: Milliseconds before stimulus to use for baseline
                        (default: 500.0).
            fps: Frame rate in Hz. If None, uses get_fps().

        Returns:
            2D array (H, W) with baseline mean values for each pixel,
            or None if error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if len(data.shape) != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
            return None

        if fps is None:
            fps = self.get_fps()
            if fps is None:
                LOGGER.error("FPS not available and not provided")
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

        try:
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

        except Exception:
            LOGGER.exception("Error calculating baseline")
            return None

    def set_roi(self, roi: tuple[int, int] | dict[str, Any]) -> None:
        """
        Set Region of Interest (ROI) center coordinates.

        Accepts ROI as tuple (x, y) or dictionary with 'center' key.
        Stores ROI in self.roi as dictionary format.

        Args:
            roi: ROI center as tuple (x, y) or dictionary with 'center' key.
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
        Calculate ROI centroid from percentile threshold.

        Finds pixels above percentile threshold and calculates their centroid.

        Args:
            data: 2D array (map/data to analyze).
            percentile: Percentile threshold (default: 95.0).

        Returns:
            Tuple (roi_x, roi_y) with centroid coordinates.
        """
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        thr = np.percentile(data.ravel(), percentile)
        points = np.where(data > thr)

        if len(points[0]) == 0:
            LOGGER.warning(f"No points above {percentile}th percentile threshold ({thr:.2f})")
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

    def calculate_trace_within_roi(
        self,
        data: np.ndarray | None = None,
        roi: tuple[int, int] | dict[str, Any] | None = None,
        diameter: float = 50.0,
    ) -> np.ndarray | None:
        """
        Calculate mean trace within circular ROI.

        Extracts mean fluorescence from circular region for each frame.

        Args:
            data: 3D imaging data (T, H, W). If None, uses self.imaging_data.
            roi: ROI center as tuple (x, y) or dict. If None, uses self.roi.
            diameter: Diameter of circular ROI in pixels (default: 50.0).

        Returns:
            1D trace array (length = number of frames), or None if error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if len(data.shape) != 3:
            LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
            return None

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

        T, H, W = data.shape
        center_x = center[0]
        center_y = center[1]
        radius = diameter / 2.0

        if center_x < 0 or center_x >= W or center_y < 0 or center_y >= H:
            LOGGER.warning(
                f"ROI center ({center_x}, {center_y}) is outside image bounds ({W}, {H})"
            )

        y, x = np.ogrid[:H, :W]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

        trace = np.zeros(T)
        for frame_idx in range(T):
            frame = data[frame_idx]
            masked_frame = frame[mask]
            if len(masked_frame) > 0:
                trace[frame_idx] = np.mean(masked_frame)
            else:
                trace[frame_idx] = 0.0

        LOGGER.info(
            f"Calculated trace from ROI: center=({center_x}, {center_y}), "
            f"diameter={diameter}, trace length={T}, "
            f"mean={trace.mean():.2f}, std={trace.std():.2f}"
        )

        return trace

    def process_dff_windows(
        self,
        windows: dict[str, tuple[int, int]],
        fps: float = 20.0,
        dff_filename: str = "dff.npy",
        vmin: float | None = None,
        vmax: float | None = None,
        colors: list[str] | None = None,
    ) -> None:
        """
        Process DFF movie for THIS trial over time windows and save average images.

        Single-trial only:
            Infers dataset/session/protocol/trial from self.trial_path, assuming
            a raw path structure like:

                data/raw/<dataset_id>/<session_id>/<protocol_name>/<trial_name>/

            and looks for the corresponding DFF here:

                data/processed/<dataset_id>/<session_id>/<protocol_name>/<trial_name>/<dff_filename>

        Windows are defined in frame indices:
            windows = {label: (start_frame, end_frame)}

        Args:
            windows: Dict mapping label -> (start_frame, end_frame) in frames.
            fps: Frame rate in Hz (used only to define windows upstream; not used internally).
            dff_filename: Name of the DFF file in the processed folder (default: "dff.npy").
            vmin: Colormap minimum. If None, computed from this DFF.
            vmax: Colormap maximum. If None, computed from this DFF.
            colors: Optional list of color strings to build a custom colormap.
                    If None, uses matplotlib's built-in "inferno".

        Example
        -------
        >>> fps = 20
        >>> windows = {
        ...     "5-6s": (5 * fps, 6 * fps),
        ...     "6-7s": (6 * fps, 7 * fps),
        ... }
        >>> wf = WidefieldAnalysis(
        ...     Path("data/raw/JPCM-07991/250818_leica/22_42_interleaved/250818_160439")
        ... )
        >>> wf.process_dff_windows(
        ...     windows=windows,
        ...     fps=fps,
        ... )
        """
        # ------------------------------------------------------------------
        # Infer dataset/session/protocol/trial from self.trial_path
        # ------------------------------------------------------------------
        trial_path = Path(self.trial_path).resolve()
        parts = trial_path.parts

        # Expect something like: (..., "data", "raw", dataset_id, session_id, protocol_name, trial_name)
        try:
            raw_idx = parts.index("raw")
        except ValueError:
            LOGGER.error(f"'raw' not found in trial_path: {trial_path}")
            return

        # We need at least: raw / dataset / session / protocol / trial
        if len(parts) < raw_idx + 5:
            LOGGER.error(
                "trial_path structure too short to infer dataset/session/protocol/trial: "
                f"{trial_path}"
            )
            return

        dataset_id = parts[raw_idx + 1]
        session_id = parts[raw_idx + 2]
        protocol_name = parts[raw_idx + 3]
        trial_name = parts[raw_idx + 4]

        # data_root is the parent of 'raw'
        data_root = Path(*parts[:raw_idx])  # e.g. 'data'

        LOGGER.info(
            f"Inferred dataset/session/protocol/trial from trial_path:\n"
            f"  data_root     = {data_root}\n"
            f"  dataset_id    = {dataset_id}\n"
            f"  session_id    = {session_id}\n"
            f"  protocol_name = {protocol_name}\n"
            f"  trial_name    = {trial_name}"
        )

        # ------------------------------------------------------------------
        # Locate processed DFF for this trial
        # ------------------------------------------------------------------
        processed_trial_folder = (
            data_root
            / "processed"
            / dataset_id
            / session_id
            / protocol_name
            / trial_name
        )

        if not processed_trial_folder.exists():
            LOGGER.error(f"Processed trial folder does not exist: {processed_trial_folder}")
            return

        dff_path = processed_trial_folder / dff_filename
        if not dff_path.exists():
            LOGGER.error(f"DFF file not found: {dff_path}")
            return

        LOGGER.info(f"Loading DFF from: {dff_path}")
        try:
            dff = np.load(dff_path)
        except Exception:
            LOGGER.exception(f"Error loading DFF from: {dff_path}")
            return

        if dff.ndim != 3:
            LOGGER.error(f"Expected 3D DFF array (T, H, W), got {dff.shape} in {dff_path}")
            return

        T, H, W = dff.shape
        LOGGER.info(f"DFF shape: T={T}, H={H}, W={W}")

        # ------------------------------------------------------------------
        # Colormap selection
        # ------------------------------------------------------------------
        if colors is None:
            cmap = plt.get_cmap("inferno")
        else:
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        # ------------------------------------------------------------------
        # Determine vmin / vmax if not provided
        # ------------------------------------------------------------------
        data_min = float(dff.min())
        data_max = float(dff.max())
        vmin_eff = data_min if vmin is None else vmin
        vmax_eff = data_max if vmax is None else vmax

        LOGGER.info(
            f"Color scaling: vmin={vmin_eff:.4f}, vmax={vmax_eff:.4f} "
            f"(data range: [{data_min:.4f}, {data_max:.4f}])"
        )

        # ------------------------------------------------------------------
        # Output directory for window averages (data/analyzed/...)
        # ------------------------------------------------------------------
        analyzed_trial_folder = (
            data_root
            / "analyzed"
            / dataset_id
            / session_id
            / protocol_name
            / f"avg_windows_{trial_name}"
        )
        analyzed_trial_folder.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Saving window averages to: {analyzed_trial_folder}")

        # ------------------------------------------------------------------
        # Loop over windows
        # ------------------------------------------------------------------
        for label, (start, end) in windows.items():
            if start >= end:
                LOGGER.warning(f"Skipping window '{label}': invalid range ({start}, {end})")
                continue

            if start >= T:
                LOGGER.warning(
                    f"Skipping window '{label}': start {start} >= total frames {T}"
                )
                continue

            if end > T:
                LOGGER.warning(
                    f"Window '{label}' end frame {end} exceeds movie length {T}. Clipping."
                )
                end = T

            window_movie = dff[start:end]
            if window_movie.size == 0:
                LOGGER.warning(
                    f"Window '{label}' has no frames after clipping (start={start}, end={end})"
                )
                continue

            # Average in time after rotating to match your original script
            avg_img = np.mean(
                np.rot90(window_movie, k=-1, axes=(1, 2)),
                axis=0,
            )

            safe_label = label.replace("-", "_")

            # Save numpy average image
            np.save(
                analyzed_trial_folder / f"{trial_name}_average_{safe_label}.npy",
                avg_img,
            )

            # Plot & save as images
            fig, ax = plt.subplots()
            im = ax.imshow(avg_img, cmap=cmap, vmin=vmin_eff, vmax=vmax_eff)
            ax.set_title(f"{trial_name} | {label}")
            ax.axis("on")
            plt.colorbar(im, ax=ax, label="dF/F")

            fig.savefig(
                analyzed_trial_folder / f"{trial_name}_average_{safe_label}.svg",
                format="svg",
            )
            fig.savefig(
                analyzed_trial_folder / f"{trial_name}_average_{safe_label}.png",
                format="png",
            )
            plt.close(fig)

            LOGGER.info(
                f"Saved average image for window '{label}' "
                f"to {analyzed_trial_folder} "
                f"(files prefix: {trial_name}_average_{safe_label})"
            )

        LOGGER.info("DFF window processing (single trial) completed.")

    def close(self) -> None:
        """Clean up resources."""
        self.imaging_data = None
        self.green_reference = None
        self.timestamps = None
        self.sensor_data = {}
        LOGGER.debug("Resources cleaned up")
