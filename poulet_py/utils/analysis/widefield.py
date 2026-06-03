from numpy import maximum


try:
    import json
    from collections.abc import Callable
    from math import sqrt
    from pathlib import Path
    from typing import Self

    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from numpy import ascontiguousarray, ceil, ndarray, ogrid, pad, zeros, any, ceil, array, floor
    from pydantic import PrivateAttr

    from poulet_py import Session, WidefieldData
    from poulet_py.config.logging import LOGGER

except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

Also ensure: h5py, numpy, pandas, scikit-image, imageio, matplotlib are installed
"""
    raise ImportError(msg) from e


class WidefieldAnalysis(Session):
    _created_movies: dict[str, Path] = PrivateAttr(default_factory=dict)
    _mask_data: dict[str, float] | None = PrivateAttr(default=None)

    def downscale(self, target: tuple[int, int] | int | float, *, inplace: bool = False) -> Self:
        obj = self if inplace else self.model_copy(deep=True)

        target_dim = zeros(2, dtype="int32")

        if isinstance(target, (tuple, int)):
            target_dim[:] = target
        elif isinstance(target, float):
            if not (0 < target < 1):
                raise ValueError("Float target must be between 0 and 1.")
        else:
            raise ValueError("target must be a tuple, int, or float.")

        if (target_dim <= 0).any():
            raise ValueError("Target dimensions must be positive.")

        # TODO Vik filtering
        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging = trial.data.imaging
            t = imaging.shape[0]
            dim = array(imaging.shape[1:])

            if isinstance(target, float):
                current_target = maximum(1, floor(dim * target))
            else:
                current_target = target_dim

            if (current_target >= dim).any():
                raise ValueError("Target dimensions must be smaller than original.")

            factor = (dim + current_target - 1) // current_target
            padding = current_target * factor - dim

            imaging = pad(
                imaging,
                ((0, 0), (0, padding[0]), (0, padding[1])),
                mode="constant",
                constant_values=0,
            )

            imaging = ascontiguousarray(imaging)
            imaging = imaging.reshape(
                t, current_target[0], factor[0], current_target[1], factor[1]
            ).mean(axis=(2, 4), dtype="float32")

            trial.data._imaging = imaging

        return obj

    def to_numpy(self) -> dict[str, ndarray]:
        data_dict = {}

        for trial in self.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            if trial.data.imaging is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                continue

            data_dict[trial.path.name] = trial.data.imaging

        return data_dict

    #### add the processed folder in the common.py get processed folder
    #### this is important for the saving the array as well if needed
    def create_movie(
        self,
        output_path: Path | None = None,
        fps: int = 10,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_callback: Callable | None = None,
        inplace: bool = False,
    ) -> Self | None:
        """
        Create MP4 movie(s) from imaging data.

        Follows the same concept as downscale():
        - creates obj from self depending on inplace
        - loops over obj.trials
        - reads trial.data.imaging
        - saves one movie per trial
        - stores output paths in obj._created_movies
        - returns obj only when inplace=False
        """
        obj = self if inplace else self.model_copy(deep=True)

        obj._created_movies = {}

        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            movie_data = trial.data.imaging

            if movie_data is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                return None

            if movie_data.ndim != 3:
                LOGGER.error(f"Expected 3D array (T, H, W), got: {movie_data.shape}")
                return None

            if output_path is None:
                trial_output_path = trial.path / "movie.mp4"
            else:
                output_folder = Path(output_path)
                output_folder.mkdir(parents=True, exist_ok=True)
                trial_output_path = output_folder / f"{trial.path.name}_movie.mp4"

            try:
                norm = Normalize(
                    vmin=vmin if vmin is not None else movie_data.min(),
                    vmax=vmax if vmax is not None else movie_data.max(),
                )
                colormap = cm.get_cmap(cmap)

                with imageio.get_writer(str(trial_output_path), fps=fps) as writer:
                    for frame_idx, frame in enumerate(movie_data):
                        rgba_frame = colormap(norm(frame))
                        rgb_frame = (rgba_frame[:, :, :3] * 255).astype("uint8")

                        if frame_callback is not None:
                            callback_result = frame_callback(rgb_frame, frame_idx)

                            if callback_result is not None:
                                rgb_frame = callback_result

                        writer.append_data(rgb_frame)

                obj._created_movies[trial.path.name] = trial_output_path

                LOGGER.info(f"Created movie: {trial_output_path}")

            except Exception:
                LOGGER.exception(f"Error creating movie for trial: {trial.path.name}")
                return None

        if not obj._created_movies:
            LOGGER.warning("No movies were created.")
            return None

        if not inplace:
            return obj

    def window_mask(self, initial_radius: float = 100.0, *, inplace: bool = False) -> Self | None:
        obj = self if inplace else self.model_copy(deep=True)

        reference_frame = None

        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging_data = trial.data.imaging

            if imaging_data is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                return None

            if imaging_data.ndim != 3:
                LOGGER.error(f"Expected 3D array (T, H, W), got: {imaging_data.shape}")
                return None

            reference_frame = imaging_data[0]
            break

        if reference_frame is None:
            LOGGER.warning("No widefield imaging data found.")
            return None

        fig, ax = plt.subplots()
        ax.imshow(reference_frame, cmap="gray")
        ax.set_title(
            "Click mask center, then click mask edge.\n"
            f"Initial radius suggestion: {initial_radius}px"
        )

        points = plt.ginput(2, timeout=0)
        plt.close(fig)

        if len(points) != 2:
            LOGGER.warning("Mask creation cancelled.")
            return None

        center_x, center_y = points[0]
        edge_x, edge_y = points[1]

        radius = sqrt((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2)

        obj._mask_data = {
            "center_x": float(center_x),
            "center_y": float(center_y),
            "radius": float(radius),
        }

        LOGGER.info(f"Created mask: {obj._mask_data}")

        if not inplace:
            return obj

    def apply_mask(
        self,
        mask_data: dict[str, float] | None = None,
        mask_name: str = "mask",
        inplace: bool = False,
    ) -> Self | None:
        """
        Apply a circular mask to the current imaging data.

        If downscale() was called first, this applies the mask to the
        downscaled movie.
        """
        obj = self if inplace else self.copy(deep=True)

        if mask_data is None:
            mask_data = obj._mask_data

        if mask_data is None:
            mask_data = obj.load_mask(mask_name)

        if mask_data is None:
            LOGGER.warning("No mask data available.")
            return None

        center_x = mask_data["center_x"]
        center_y = mask_data["center_y"]
        radius = mask_data["radius"]

        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging_data = trial.data.imaging

            if imaging_data is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                return None

            if imaging_data.ndim != 3:
                LOGGER.error(f"Expected 3D array (T, H, W), got: {imaging_data.shape}")
                return None

            _, height, width = imaging_data.shape

            y, x = ogrid[:height, :width]
            circular_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

            masked_data = imaging_data.copy()
            masked_data[:, ~circular_mask] = 0

            trial.data._imaging = masked_data.astype(imaging_data.dtype)

        if not inplace:
            return obj

    #### save mask in processed?
    def save_mask(
        self,
        mask_data: dict[str, float] | None = None,
        name: str = "mask",
    ) -> Path | None:
        """
        Save mask parameters to the session processed folder.

        Args:
            mask_data: Dictionary with center_x, center_y, radius.
                If None, uses self._mask_data.
            name: Base filename without .json extension.

        Returns:
            Path to saved JSON file, or None on error.
        """
        if mask_data is None:
            mask_data = self._mask_data

        if mask_data is None:
            LOGGER.warning("No mask data available to save.")
            return None

        processed_folder = self._get_session_processed_folder()

        if processed_folder is None:
            return None

        output_path = processed_folder / f"{name}.json"

        try:
            with open(output_path, "w") as file:
                json.dump(mask_data, file, indent=4)

            LOGGER.info(f"Saved mask to: {output_path}")
            return output_path

        except Exception:
            LOGGER.exception("Error saving mask")
            return None

    def load_mask(self, name: str = "mask") -> dict[str, float] | None:
        """
        Load mask parameters from the session processed folder.

        Args:
            name: Base filename without .json extension.

        Returns:
            Dictionary with center_x, center_y, radius keys,
            or None if file not found.
        """
        processed_folder = self._get_session_processed_folder()

        if processed_folder is None:
            return None

        mask_path = processed_folder / f"{name}.json"

        if not mask_path.exists():
            LOGGER.warning(f"Mask file not found: {mask_path}")
            return None

        try:
            with open(mask_path, "r") as file:
                mask_data = json.load(file)

            self._mask_data = mask_data

            LOGGER.info(f"Loaded mask from: {mask_path}")
            return mask_data

        except Exception:
            LOGGER.exception("Error loading mask")
            return None
