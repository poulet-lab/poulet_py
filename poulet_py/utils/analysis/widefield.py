try:
    from typing import Self

    from numpy import ceil, ndarray, pad
    from pydantic import PrivateAttr

    from poulet_py import Session, WidefieldData
    from poulet_py.config.logging import LOGGER
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

    def motion_correction(
        self,
        motion_region: tuple | None = None,
        shift_method: str = "integer",
        n_parallel_workers: int | None = 4,
        reference_image: ndarray | None = None,
        *,
        inplace: bool = False,
    ) -> Self | None:
        obj = self if inplace else self.copy(deep=True)

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
