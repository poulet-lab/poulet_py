try:
    from typing import Callable, Self
    from pathlib import Path
    from numpy import ceil, ndarray, pad
    from pydantic import PrivateAttr
    import json
    from math import sqrt # type: ignore
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from numpy import ceil, ndarray, ogrid, pad
    from matplotlib import cm
    from matplotlib.colors import Normalize
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
    _downscaled_imaging: dict[int, ndarray] = PrivateAttr(default_factory=dict)
    _created_movies: dict[str, Path] = PrivateAttr(default_factory=dict)
    _mask_data: dict[str, float] | None = PrivateAttr(default=None)

    def downscale(
        self, target_resolution: tuple[int, int], factor: int, inplace: bool = False
    ) -> Self | None:
        obj = self if inplace else self.copy(deep=True)

        # TODO filtering before (vik)
        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging_data = trial.data.imaging

            if imaging_data is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                return None

            T, H, W = imaging_data.shape
            mov = imaging_data.copy()

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

                    mov = pad(
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
                    mov = pad(
                        mov, ((0, 0), (0, pad_H), (0, pad_W)), mode="constant", constant_values=0
                    )
                    T, H, W = mov.shape

                mov = mov.reshape(T, H // factor, factor, W // factor, factor).mean(4).mean(2)

                trial.data._imaging = mov.astype("uint16")
                # write this also in metadata table

        if not inplace:
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
        obj = self if inplace else self.copy(deep=True)

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
#### i made the mask to be created on the downscaled movie already 
    def create_mask(
        self,
        initial_radius: float = 100.0,
        inplace: bool = False,
    ) -> Self | None:
        """
        Create a circular mask on the current imaging data.

        If downscale() was called before this, the mask is created on the
        downscaled imaging frame.
        """
        obj = self if inplace else self.copy(deep=True)

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