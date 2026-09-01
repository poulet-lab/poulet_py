try:
    from collections.abc import Callable
    from math import sqrt
    from pathlib import Path
    from typing import Any, Literal, Self, overload

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
    from numpy import (
        percentile as percentile_fn,
    )
    from prompt_toolkit.shortcuts import ProgressBar
    from pydantic import PrivateAttr

    from poulet_py import LOGGER, Session, WidefieldData
    from poulet_py.io.data_structures.widefield import WidefieldMaskMetaData
    from poulet_py.utils.analysis.motion import apply_motion_correction

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
    _motion_vectors: dict[str, ndarray] = PrivateAttr(default_factory=dict)
    _motion_reference_images: dict[str, ndarray] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        trials = []
        with ProgressBar("Widefield Analysis Init") as pb:
            for trial in pb(self.trials, label="Trials", total=len(self.trials)):
                if isinstance(trial.data, WidefieldData):
                    trials.append(trial)
        self._trials._data = trials

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
                trial.data.metadata.level = "processed"

        return obj

    def motion_correction(
        self,
        motion_region: tuple | None = None,
        shift_method: str = "integer",
        n_parallel_workers: int | None = 4,
        reference_image: ndarray | None = None,
        *,
        inplace: bool = False,
    ) -> Self | None:
        obj = self if inplace else self.model_copy(deep=True)

        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging_data = trial.data.imaging

            if imaging_data is None:
                LOGGER.warning("No imaging data loaded. Call load_data() first.")
                return None

            corrected_movie, motion_vectors, motion_reference_image = apply_motion_correction(
                imaging_data,
                motion_region=motion_region,
                shift_method=shift_method,
                n_parallel_workers=n_parallel_workers,
                reference_image=reference_image,
            )
            trial.data._imaging = corrected_movie
            trial_key = str(trial.path)
            obj._motion_vectors[trial_key] = motion_vectors
            obj._motion_reference_images[trial_key] = motion_reference_image

        if not inplace:
            return obj

    def calculate_percentile(
        self,
        percentile: float = 15.0,
        stimulus_start_frame: int | None = None,
        baseline_ms: float | None = None,
        *,
        inplace: bool = False,
    ) -> Self:
        obj = self if inplace else self.model_copy(deep=True)

        with ProgressBar("Calculating percentile projection") as pb:
            for trial in pb(obj.trials, label="Trials", total=len(obj.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue

                imaging = trial.data.imaging  # (frames, height, width)
                data_window = imaging

                if stimulus_start_frame is not None and baseline_ms is not None:
                    fps = trial.data.metadata.fps  # TODO: confirm attribute path
                    if fps is None:
                        LOGGER.warning(
                            f"FPS not available for trial {trial.path.name}; using full movie"
                        )
                    else:
                        baseline_frames = int(round((baseline_ms / 1000.0) * fps))
                        start = max(0, stimulus_start_frame - baseline_frames)
                        end = stimulus_start_frame

                        if end <= start:
                            LOGGER.warning(
                                f"Invalid baseline window for trial {trial.path.name}; using full movie"
                            )
                        else:
                            data_window = imaging[start:end]

                if data_window.shape[0] == 0:
                    LOGGER.warning(f"Empty data window for trial {trial.path.name}; skipping")
                    continue

                # TODO: confirm attribute name on WidefieldData
                trial.data._imaging = percentile_fn(data_window, percentile, axis=0)
                trial.data.metadata.level = "processed"

        return obj

    def calculate_baseline(
        self,
        stimulus_start_frame: int = 0,
        baseline_ms: float = 500.0,
        *,
        inplace: bool = False,
    ) -> Self:
        obj = self if inplace else self.model_copy(deep=True)

        with ProgressBar("Calculating baseline") as pb:
            for trial in pb(obj.trials, label="Trials", total=len(obj.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue

                fps = trial.data.metadata.fps  # TODO: confirm attribute path
                if fps is None:
                    LOGGER.error(
                        f"Skipping baseline for trial {trial.path.name}: no FPS in metadata"
                    )
                    continue

                imaging = trial.data.imaging  # (frames, height, width)

                baseline_frames = round((baseline_ms / 1000.0) * fps)
                start = max(0, stimulus_start_frame - baseline_frames)
                end = stimulus_start_frame

                if end <= start:
                    LOGGER.warning(f"Invalid baseline window for trial {trial.path.name}; skipping")
                    continue

                data_window = imaging[start:end]
                if data_window.shape[0] == 0:
                    LOGGER.warning(f"Empty baseline window for trial {trial.path.name}; skipping")
                    continue

                # TODO: confirm attribute name on WidefieldData
                trial.data._imaging = data_window.mean(axis=0, dtype="float32")
                trial.data.metadata.level = "processed"

        return obj

    def calculate_deltaff(self, *, inplace: bool = False) -> Self:
        obj = self if inplace else self.model_copy(deep=True)

        with ProgressBar("Calculating delta F/F") as pb:
            for trial in pb(obj.trials, label="Trials", total=len(obj.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue

                baseline = getattr(trial.data, "baseline", None)  # TODO: confirm attribute name
                if baseline is None:
                    LOGGER.warning(
                        f"No baseline for trial {trial.path.name}; run calculate_baseline() first"
                    )
                    continue

                imaging = trial.data.imaging.astype("float32")
                safe_baseline = baseline.astype("float32")
                safe_baseline[safe_baseline == 0] = 1e-6  # avoid divide-by-zero

                # TODO: confirm attribute name on WidefieldData
                trial.data._imaging = (imaging - safe_baseline) / safe_baseline
                trial.data.metadata.level = "processed"

        return obj

    def window_mask(self, *, inplace: bool = False) -> Self | None:
        obj = self if inplace else self.model_copy(deep=True)

        for trial in obj.trials:
            if not isinstance(trial.data, WidefieldData):
                continue

            imaging_data = trial.data.imaging
            _, height, width = imaging_data.shape
            reference_frame = imaging_data[0]

            mask_data = WidefieldMaskMetaData()
            points = 0
            while points != 2:
                fig, ax = plt.subplots()
                ax.imshow(reference_frame, cmap="gray")
                ax.set_title("Click mask center, then click mask edge.")

                points = plt.ginput(2, timeout=0)
                plt.close(fig)

                if len(points) != 2:
                    LOGGER.warning("Please select two points")
                    continue

                center = points[0]
                edge = points[1]
                radius = sqrt((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2)

                # TODO better prompt
                LOGGER.info(f"Mask data {mask_data}")
                prompt = input("Correct? [Y,n]")
                if prompt.lower() == "y":
                    mask_data.center = center
                    mask_data.radius = radius
                    break

            trial.data.metadata.mask_data = mask_data

            if inplace:
                y, x = ogrid[:height, :width]
                circular_mask = (x - mask_data.center[0]) ** 2 + (
                    y - mask_data.center[1]
                ) ** 2 <= mask_data.radius**2

                trial.data._imaging[:, ~circular_mask] = 0
                trial.data.metadata.level = "processed"

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

    @overload
    def save(
        self,
        file: Path,
        mode: Literal["append", "overwrite"] = "append",
    ) -> Self: ...

    @overload
    def save(
        self,
        file: Path,
        mode: Literal["append", "overwrite"] = "append",
        *,
        chunks: tuple[int, ...] | int | bool | None = None,
        compression: Literal["gzip", "lzf", "szip"] | None = "gzip",
        compression_opts: Any = 4,
        fletcher32: bool = False,
    ) -> Self: ...

    @overload
    def save(
        self,
        file: Path,
        mode: Literal["append", "overwrite"] = "append",
        *,
        fps: int = 10,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_callback: Callable[[ndarray, int], ndarray | None] | None = None,
        codec: str = "mp4v",
    ) -> Self: ...

    # TODO save metadata both from trials and data structures
    def save(self, file: Path, mode: Literal["append", "overwrite"] = "append", **kwargs) -> Self:
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
                self._to_npz(file, mode, **kwargs)
            elif file.suffix in {".h5", ".hdf5"}:
                self._to_h5(file, mode, **kwargs)
            elif file.suffix in {"mp4", ".avi"}:
                self._to_movie(file, mode, **kwargs)
            else:
                raise ValueError(f"Unsupported file extension: {file.suffix}")

            LOGGER.info(f"Saved widefield data to: {file}")

        except Exception as e:
            raise RuntimeError(f"Error saving file: {file}") from e

        return self

    # TODO trials_to_df, trials_to_array? for faster processing. keep in cache?
    def _trials_to_dict(self) -> dict[str, ndarray]:
        data_dict: dict[str, ndarray] = {}
        with ProgressBar("Converting trials to dict") as pb:
            for trial in pb(self.trials, label="Trials", total=len(self.trials)):
                if not isinstance(trial.data, WidefieldData):
                    continue
                # TODO not only imaging
                data_dict[trial.path.name] = trial.data.imaging

        return data_dict

    def _to_npz(self, file: Path, mode: Literal["append", "overwrite"] = "append") -> None:
        trials_dict = self._trials_to_dict()

        if file.exists() and mode == "append":
            with load(file, allow_pickle=False) as existing:
                trials_dict.update({k: existing[k] for k in existing.files})

        savez_compressed(file, allow_pickle=False, **trials_dict)

    def _to_h5(
        self,
        file: Path,
        mode: Literal["append", "overwrite"] = "append",
        *,
        chunks: tuple[int, ...] | int | bool | None = None,
        compression: Literal["gzip", "lzf", "szip"] | None = "gzip",
        compression_opts: Any = 4,
        fletcher32: bool = False,
    ) -> None:
        # TODO more options for compression, chunking, etc.
        trials_dict = self._trials_to_dict()
        with File(file, "a") as h5file:
            for trial_name, data in trials_dict.items():
                if trial_name in h5file:
                    if mode == "append":
                        continue
                    else:
                        h5file["trial_name"] = data

                h5file.create_dataset(
                    trial_name,
                    data=data,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                    fletcher32=fletcher32,
                )

    def _to_movie(
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
        trials_dict = self._trials_to_dict()
        with ProgressBar("Rendering movies") as pb:
            for trial_name, data in trials_dict.items():
                try:
                    if data.ndim != 3:
                        raise ValueError(
                            f"Expected imaging shape (frames, height, width), got {data.shape}"
                        )
                    height, width = data[0].shape

                    local_vmin = vmin if vmin is not None else float(data.min())
                    local_vmax = vmax if vmax is not None else float(data.max())

                    if local_vmax <= local_vmin:
                        local_vmax = local_vmin + 1.0

                    normalized = clip((data - local_vmin) / (local_vmax - local_vmin), 0, 1)

                    colormap = cm.get_cmap(cmap)

                    colored = colormap(normalized)[..., :3]
                    colored = (colored * 255).astype("uint8")

                    trial_output_path = path.parent / (trial_name + path.suffix)

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
                            enumerate(data), label="Frames", total=len(data)
                        ):
                            if frame_callback is not None:
                                result = frame_callback(frame, frame_idx)
                                if result is not None:
                                    frame = result

                            writer.write(cvtColor(frame, COLOR_RGB2BGR))
                    finally:
                        writer.release()

                except Exception as e:
                    raise RuntimeError(f"Error creating movie for trial: {trial_name}") from e
        return self
