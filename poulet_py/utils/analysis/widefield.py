try:
    import json
    from collections.abc import Callable
    from math import sqrt
    from pathlib import Path
    from typing import Literal, Self

    import matplotlib.pyplot as plt
    from cv2 import COLOR_RGB2BGR, VideoWriter, VideoWriter_fourcc, cvtColor
    from h5py import File
    from matplotlib import cm
    from numpy import (
        array,
        ascontiguousarray,
        clip,
        floor,
        load,
        maximum,
        ndarray,
        ogrid,
        pad,
        savez_compressed,
        zeros,
    )
    from prompt_toolkit.shortcuts import ProgressBar
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
        with ProgressBar("Downscaling movies") as pb:
            for trial in pb(obj.trials, label="Trials", total=len(obj.trials)):
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

    def apply_window_mask(
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

    def movie(
        self,
        path: Path,
        mode: Literal["append", "overwrite"] = "append",
        *,
        fps: int = 10,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_callback: Callable[[ndarray, int], ndarray | None] | None = None,
        codec: str = "mp4v",
    ) -> Self:
        #### add the processed folder in the common.py get processed folder
        #### this is important for the saving the array as well if needed
        path.mkdir(parents=True, exist_ok=True)

        with ProgressBar("Rendering movies") as pb:
            for trial in pb(self.trials, label="Trials", total=len(self.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue

                try:
                    imaging = trial.data.imaging
                    if imaging.ndim != 3:
                        raise ValueError(
                            f"Expected imaging shape (frames, height, width), got {imaging.shape}"
                        )
                    height, width = imaging[0].shape

                    local_vmin = vmin if vmin is not None else float(imaging.min())
                    local_vmax = vmax if vmax is not None else float(imaging.max())

                    if local_vmax <= local_vmin:
                        local_vmax = local_vmin + 1.0

                    normalized = clip((imaging - local_vmin) / (local_vmax - local_vmin), 0, 1)

                    colormap = cm.get_cmap(cmap)

                    colored = colormap(normalized)[..., :3]
                    colored = (colored * 255).astype("uint8")

                    trial_output_path = path / f"{trial.path.stem}.mp4"

                    if trial_output_path.exists():
                        if mode == "append":
                            LOGGER.info(f"Skipping existing movie: {trial_output_path.name}")
                            continue
                        elif mode == "overwrite":
                            LOGGER.info(f"Overwriting existing movie: {trial_output_path.name}")

                    trial_output_path.unlink()
                    writer = VideoWriter(
                        str(trial_output_path),
                        VideoWriter_fourcc(*codec),
                        fps,
                        (width, height),
                        isColor=True,
                    )

                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {trial_output_path}")

                    try:
                        for frame_idx, frame in pb(
                            enumerate(imaging), label="Frames", total=len(imaging)
                        ):
                            if frame_callback is not None:
                                result = frame_callback(frame, frame_idx)
                                if result is not None:
                                    frame = result

                            writer.write(cvtColor(frame, COLOR_RGB2BGR))
                    finally:
                        writer.release()

                except Exception as e:
                    raise RuntimeError(f"Error creating movie for trial: {trial.path.name}") from e
        return self

    def save(self, file: Path, mode: Literal["append", "overwrite"] = "append") -> Self:
        file.parent.mkdir(parents=True, exist_ok=True)

        if file.exists():
            if file.is_dir():
                raise ValueError(f"Path must be a file: {file}")

            if mode == "overwrite":
                file.unlink()
            elif mode != "append":
                # TODO
                raise ValueError(f"Unsupported mode: {mode}")

        try:
            if file.suffix == ".npz":
                self._to_npz(file)
            elif file.suffix in {".h5", ".hdf5"}:
                self._to_h5(file)
            else:
                raise ValueError(f"Unsupported file extension: {file.suffix}")

            LOGGER.info(f"Saved widefield data to: {file}")

        except Exception as e:
            raise RuntimeError(f"Error saving file: {file}") from e

        return self

    def _to_npz(self, file: Path) -> None:
        trials_dict = self._trials_to_dict()

        if file.exists():
            # now keep existing data and add new trials
            with load(file, allow_pickle=False) as existing:
                trials_dict.update({k: existing[k] for k in existing.files})

        savez_compressed(file, **trials_dict)

    def _to_h5(self, file: Path) -> None:
        # TODO more options for compression, chunking, etc.
        trials_dict = self._trials_to_dict()
        with File(file, "a") as h5file:
            for trial_name, data in trials_dict.items():
                if trial_name in h5file:
                    continue  # TODO handle existing datasets based on mode

                h5file.create_dataset(trial_name, data=data, compression="gzip")

    def _trials_to_dict(self) -> dict[str, ndarray]:
        data_dict: dict[str, ndarray] = {}
        with ProgressBar("Converting trials to dict") as pb:
            for trial in pb(self.trials, label="Trials", total=len(self.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue

                data_dict[trial.path.name] = trial.data.imaging

        return data_dict

    #TODO trials_to_df, trials_to_array? for faster processing.