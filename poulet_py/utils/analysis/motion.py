try:
    from datetime import datetime, timezone
    from io import BytesIO
    from multiprocessing import Pool
    from pathlib import Path

    from imageio import get_writer, imsave
    from imageio.v2 import imread
    from matplotlib.pyplot import close, colorbar, subplots, tight_layout
    from numpy import (
        abs,
        arange,
        arctan2,
        argmax,
        argsort,
        array,
        ceil,
        exp,
        fix,
        float64,
        int16,
        int32,
        load,
        mean,
        ndarray,
        outer,
        reshape,
        roll,
        round,
        save,
        savez,
        sort,
        sqrt,
        sum,
        tensordot,
        uint16,
        unravel_index,
        vstack,
    )
    from numpy.random import default_rng
    from scipy.fft import fftfreq, fftn, ifftn
    from scipy.ndimage import fourier_shift

    from poulet_py.config.logging import LOGGER
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

Also ensure: imageio, matplotlib, numpy, scipy are installed
"""
    raise ImportError(msg) from e


MAX_REFERENCE_FRAMES = 500
REFERENCE_FRAME_COUNT = 30
FRAME_BATCH_SIZE = 50
UPSAMPLE_FACTOR = 20
RGBA_CHANNELS = 4


def _upsampled_dft(
    frequency_domain_data: ndarray,
    upsampled_region_size: int | tuple,
    upsample_factor: int = 1,
    axis_offsets: tuple | None = None,
) -> ndarray:
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * frequency_domain_data.ndim
    elif len(upsampled_region_size) != frequency_domain_data.ndim:
        msg = (
            "shape of upsampled region sizes must be equal "
            "to input data's number of dimensions."
        )
        raise ValueError(msg)

    if axis_offsets is None:
        axis_offsets = [0] * frequency_domain_data.ndim
    elif len(axis_offsets) != frequency_domain_data.ndim:
        msg = "number of axis offsets must be equal to input data's number of dimensions."
        raise ValueError(msg)

    imaginary_2pi = 1j * 2 * 3.141592653589793
    dimension_properties = list(
        zip(frequency_domain_data.shape, upsampled_region_size, axis_offsets, strict=False)
    )

    for n_items, ups_size, ax_offset in dimension_properties[::-1]:
        kernel = (arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = exp(-imaginary_2pi * kernel)
        frequency_domain_data = tensordot(kernel, frequency_domain_data, axes=(1, -1))

    return frequency_domain_data


def _compute_phase_difference(cross_correlation_max: complex) -> float:
    return arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_registration_error(
    cross_correlation_max: complex,
    source_amplitude: float,
    target_amplitude: float,
) -> float:
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / (
        source_amplitude * target_amplitude
    )
    return sqrt(abs(error))


def estimate_image_shift(
    reference_image: ndarray,
    target_image: ndarray,
    upsample_factor: int = 1,
    space: str = "real",
    *,
    return_error: bool = True,
) -> tuple | ndarray:
    if reference_image.shape != target_image.shape:
        msg = "Error: images must be same size for estimate_image_shift"
        raise ValueError(msg)

    if space.lower() == "fourier":
        reference_frequency = reference_image
        target_frequency = target_image
    elif space.lower() == "real":
        reference_frequency = fftn(reference_image)
        target_frequency = fftn(target_image)
    else:
        msg = (
            'Error: estimate_image_shift only knows the "real" '
            'and "fourier" values for the ``space`` argument.'
        )
        raise ValueError(msg)

    image_shape = reference_frequency.shape
    image_product = reference_frequency * target_frequency.conj()
    cross_correlation = ifftn(image_product)
    maxima_indices = unravel_index(argmax(abs(cross_correlation)), cross_correlation.shape)
    midpoints = array([fix(axis_size / 2) for axis_size in image_shape])

    shift_vector = array(maxima_indices, dtype=float64)
    shift_vector[shift_vector > midpoints] -= array(image_shape)[
        shift_vector > midpoints
    ]

    if upsample_factor == 1:
        if return_error:
            source_amplitude = (
                sum(abs(reference_frequency) ** 2) / reference_frequency.size
            )
            target_amplitude = (
                sum(abs(target_frequency) ** 2) / target_frequency.size
            )
            cross_correlation_max = cross_correlation[maxima_indices]
    else:
        shift_vector = round(shift_vector * upsample_factor) / upsample_factor
        upsampled_region_size = ceil(upsample_factor * 1.5)
        dft_shift = fix(upsampled_region_size / 2.0)
        upsample_factor_float = array(upsample_factor, dtype=float64)
        normalization = reference_frequency.size * upsample_factor_float**2
        sample_region_offset = dft_shift - shift_vector * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        cross_correlation /= normalization
        maxima_indices = unravel_index(argmax(abs(cross_correlation)), cross_correlation.shape)
        cross_correlation_max = cross_correlation[maxima_indices]
        maxima_indices = array(maxima_indices, dtype=float64) - dft_shift
        shift_vector = shift_vector + maxima_indices / upsample_factor

        if return_error:
            source_amplitude = _upsampled_dft(
                reference_frequency * reference_frequency.conj(), 1, upsample_factor
            )[0, 0]
            source_amplitude /= normalization
            target_amplitude = _upsampled_dft(
                target_frequency * target_frequency.conj(), 1, upsample_factor
            )[0, 0]
            target_amplitude /= normalization

    for dim in range(reference_frequency.ndim):
        if image_shape[dim] == 1:
            shift_vector[dim] = 0

    if return_error:
        return (
            shift_vector,
            _compute_registration_error(
                cross_correlation_max,
                source_amplitude,
                target_amplitude,
            ),
            _compute_phase_difference(cross_correlation_max),
        )

    return shift_vector


def find_similar_frames(initial_frames: ndarray) -> tuple:
    frames_flat = reshape(initial_frames, (initial_frames.shape[0], -1)).astype("float32")
    frames_flat = frames_flat - reshape(frames_flat.mean(axis=1), (frames_flat.shape[0], 1))
    correlation_matrix = frames_flat @ frames_flat.T
    standard_deviations = sqrt(correlation_matrix.diagonal())
    correlation_matrix = correlation_matrix / outer(standard_deviations, standard_deviations)
    correlation_sorted = -sort(-correlation_matrix, axis=1)
    correlation_top_mean = mean(correlation_sorted[:, 1:150], axis=1)
    index_max_correlation = argmax(correlation_top_mean)
    top_frame_indices = argsort(-correlation_matrix[index_max_correlation, :])

    return top_frame_indices, correlation_matrix


def _worker_cross_correlation(
    frame_batch: ndarray,
    reference_frequency: ndarray,
) -> list:
    motion_vectors = []
    for frame_idx in range(len(frame_batch)):
        motion_vectors.append(
            estimate_image_shift(
                reference_frequency,
                fftn(frame_batch[frame_idx]),
                upsample_factor=UPSAMPLE_FACTOR,
                space="fourier",
                return_error=False,
            )
        )
    return motion_vectors


def estimate_motion_vectors(
    movie: ndarray,
    motion_region: tuple | None = None,
    n_parallel_workers: int | None = 4,
    reference_image: ndarray | None = None,
) -> tuple:
    movie_copy = movie.copy()

    if motion_region is None:
        row_slice = slice(None)
        col_slice = slice(None)
    else:
        row_slice = slice(motion_region[0][0], motion_region[1][0])
        col_slice = slice(motion_region[0][1], motion_region[1][1])

    if reference_image is None:
        n_frames = movie_copy.shape[0]
        if n_frames > MAX_REFERENCE_FRAMES:
            random_generator = default_rng()
            frame_indices = random_generator.choice(
                n_frames, MAX_REFERENCE_FRAMES, replace=False
            )
        else:
            frame_indices = slice(None)

        initial_reference_frames = movie_copy[frame_indices, row_slice, col_slice]
        similar_frame_indices, _ = find_similar_frames(initial_reference_frames)
        reference_image = mean(
            initial_reference_frames[similar_frame_indices[:REFERENCE_FRAME_COUNT]],
            axis=0,
        )

    reference_frequency = fftn(reference_image)

    if n_parallel_workers is None:
        motion_vectors = []
        for frame_idx in range(len(movie_copy)):
            motion_vectors.append(
                estimate_image_shift(
                    reference_frequency,
                    fftn(movie_copy[frame_idx, row_slice, col_slice]),
                    upsample_factor=UPSAMPLE_FACTOR,
                    space="fourier",
                    return_error=False,
                )
            )
    else:
        frame_batches = []
        for frame_idx in range(0, movie_copy.shape[0], FRAME_BATCH_SIZE):
            frame_batches.append(
                (
                    movie_copy[
                        frame_idx : (frame_idx + FRAME_BATCH_SIZE),
                        row_slice,
                        col_slice,
                    ],
                    reference_frequency,
                )
            )

        with Pool(processes=n_parallel_workers) as pool:
            motion_vectors = array(pool.starmap(_worker_cross_correlation, frame_batches))

    motion_vectors = vstack(motion_vectors)

    return motion_vectors, reference_image


def apply_motion_correction(
    movie: ndarray,
    motion_region: tuple | None = None,
    motion_vectors: ndarray | None = None,
    shift_method: str = "integer",
    n_parallel_workers: int | None = 4,
    reference_image: ndarray | None = None,
) -> tuple:
    movie_copy = movie.copy()

    if motion_vectors is None:
        motion_vectors, reference_image = estimate_motion_vectors(
            movie_copy,
            motion_region=motion_region,
            n_parallel_workers=n_parallel_workers,
            reference_image=reference_image,
        )

    if shift_method == "fourier":
        for frame_idx in range(movie_copy.shape[0]):
            shifted_frame_frequency = fourier_shift(
                fftn(movie_copy[frame_idx]), motion_vectors[frame_idx]
            )
            movie_copy[frame_idx] = ifftn(shifted_frame_frequency).real
    elif shift_method == "integer":
        integer_motion = round(motion_vectors).astype(int32)
        for frame_idx in range(movie_copy.shape[0]):
            movie_copy[frame_idx] = roll(
                movie_copy[frame_idx], integer_motion[frame_idx], axis=(0, 1)
            )
    else:
        LOGGER.error(f"shift_method '{shift_method}' not recognized. Use 'fourier' or 'integer'.")
        msg = f"shift_method must be 'fourier' or 'integer', got '{shift_method}'"
        raise ValueError(msg)

    return movie_copy, motion_vectors, reference_image


def load_motion_vectors(motion_vectors_path: Path) -> ndarray | None:
    motion_vectors_path = Path(motion_vectors_path)

    if motion_vectors_path.exists():
        return load(motion_vectors_path) / 100

    return None


def save_motion_vectors(motion_vectors: ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    motion_vectors_scaled = (100 * motion_vectors).astype(int16)
    save(output_path, motion_vectors_scaled)

    return output_path


def save_corrected_recording(corrected_movie: ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save(output_path, corrected_movie)

    return output_path


def save_reference_image(reference_image: ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imsave(output_path, reference_image.astype(uint16), prefer_uint8=False)

    return output_path


def save_motion_correction_metadata(
    output_path: Path,
    shift_method: str,
    n_frames: int,
    image_shape: tuple,
    raw_trial_path: Path | None = None,
    upsample_factor: int = 20,
    n_reference_frames: int = 30,
    motion_region: tuple | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shift_method_description = {
        "integer": "Integer pixel shifts using np.roll (fast, no interpolation)",
        "fourier": "Subpixel shifts using Fourier phase shifting (slower, interpolated)",
    }
    metadata_lines = [
        "MOTION CORRECTION METADATA",
        "=" * 50,
        f"Date processed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]

    if raw_trial_path is not None:
        metadata_lines.append(f"Raw trial path: {raw_trial_path}")

    metadata_lines.extend(
        [
            "",
            "CORRECTION PARAMETERS",
            "-" * 50,
            f"Shift method: {shift_method}",
            f"  Description: {shift_method_description.get(shift_method, 'Unknown')}",
            f"Upsample factor: {upsample_factor} (1/{upsample_factor} pixel precision)",
            f"Reference frames: {n_reference_frames} most similar frames averaged",
            f"Motion region: {motion_region if motion_region else 'Full frame'}",
            "",
            "DATA DIMENSIONS",
            "-" * 50,
            f"Number of frames: {n_frames}",
            f"Frame shape: {image_shape[0]} x {image_shape[1]} pixels",
            "",
            "ALGORITHM DETAILS",
            "-" * 50,
            "Registration: FFT-based cross-correlation with upsampled DFT",
            "Reference image: Mean of N most temporally stable frames",
            "Motion vectors: [row_shift, col_shift] in pixels per frame",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(metadata_lines))

    return output_path


def save_motion_analysis(motion_vectors: ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_shift = motion_vectors[:, 0]
    col_shift = motion_vectors[:, 1]
    displacement = sqrt(row_shift**2 + col_shift**2)

    savez(
        output_path,
        row_shift=row_shift,
        col_shift=col_shift,
        displacement=displacement,
        motion_vectors=motion_vectors,
    )

    return output_path


def _plot_shift_traces(
    motion_vectors_raw: ndarray,
    motion_vectors_corrected: ndarray,
    output_path: Path,
) -> None:
    row_shift_raw = motion_vectors_raw[:, 0]
    col_shift_raw = motion_vectors_raw[:, 1]
    displacement_raw = sqrt(row_shift_raw**2 + col_shift_raw**2)
    row_shift_corrected = motion_vectors_corrected[:, 0]
    col_shift_corrected = motion_vectors_corrected[:, 1]
    displacement_corrected = sqrt(row_shift_corrected**2 + col_shift_corrected**2)
    fig, axes = subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(row_shift_raw, linewidth=0.8, label="Raw", alpha=0.7)
    axes[0].plot(row_shift_corrected, linewidth=0.8, label="Corrected", alpha=0.8)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel("Row shift (px)")
    axes[0].set_title("Estimated motion per frame")
    axes[0].legend()
    axes[1].plot(col_shift_raw, linewidth=0.8, label="Raw", alpha=0.7)
    axes[1].plot(col_shift_corrected, linewidth=0.8, label="Corrected", alpha=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_ylabel("Col shift (px)")
    axes[1].legend()
    axes[2].plot(displacement_raw, linewidth=0.8, label="Raw", alpha=0.7)
    axes[2].plot(displacement_corrected, linewidth=0.8, label="Corrected", alpha=0.8)
    axes[2].set_ylabel("Displacement (px)")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path / "motion_over_time.png", dpi=150, bbox_inches="tight")
    close(fig)
    LOGGER.info(f"Saved: {output_path / 'motion_over_time.png'}")


def _plot_motion_trajectory(
    motion_vectors_raw: ndarray,
    motion_vectors_corrected: ndarray,
    output_path: Path,
) -> None:
    row_shift_raw = motion_vectors_raw[:, 0]
    col_shift_raw = motion_vectors_raw[:, 1]
    row_shift_corrected = motion_vectors_corrected[:, 0]
    col_shift_corrected = motion_vectors_corrected[:, 1]
    fig, axes = subplots(1, 2, figsize=(12, 6))
    n_frames = len(col_shift_raw)
    axes[0].scatter(
        col_shift_raw,
        row_shift_raw,
        s=5,
        alpha=0.5,
        c=range(n_frames),
        cmap="viridis",
    )
    axes[0].axhline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[0].axvline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[0].set_xlabel("Col shift (px)")
    axes[0].set_ylabel("Row shift (px)")
    axes[0].set_title("Motion trajectory - Raw")
    axes[0].set_aspect("equal")
    colorbar(axes[0].collections[0], ax=axes[0], label="Frame")
    axes[1].scatter(
        col_shift_corrected,
        row_shift_corrected,
        s=5,
        alpha=0.5,
        c=range(n_frames),
        cmap="viridis",
    )
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[1].axvline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[1].set_xlabel("Col shift (px)")
    axes[1].set_ylabel("Row shift (px)")
    axes[1].set_title("Motion trajectory - Corrected")
    axes[1].set_aspect("equal")
    colorbar(axes[1].collections[0], ax=axes[1], label="Frame")
    fig.tight_layout()
    fig.savefig(output_path / "02_motion_trajectory.png", dpi=150, bbox_inches="tight")
    close(fig)
    LOGGER.info(f"Saved: {output_path / '02_motion_trajectory.png'}")


def plot_motion_over_time(
    motion_vectors_raw: ndarray,
    movie_corrected: ndarray,
    reference_image: ndarray,
    output_path: Path,
):
    LOGGER.info("Computing motion for corrected data...")
    fft_reference = fftn(reference_image)
    motion_vectors_corrected = []
    for frame_index in range(movie_corrected.shape[0]):
        shift = estimate_image_shift(
            fft_reference,
            fftn(movie_corrected[frame_index]),
            upsample_factor=UPSAMPLE_FACTOR,
            space="fourier",
            return_error=False,
        )
        motion_vectors_corrected.append(shift)
    motion_vectors_corrected = array(motion_vectors_corrected)
    output_path = Path(output_path)
    _plot_shift_traces(motion_vectors_raw, motion_vectors_corrected, output_path)
    _plot_motion_trajectory(motion_vectors_raw, motion_vectors_corrected, output_path)


def plot_stability_maps(movie_raw: ndarray, movie_corrected: ndarray, output_path: Path):
    mean_raw = movie_raw.mean(axis=0)
    mean_corrected = movie_corrected.mean(axis=0)
    std_raw = movie_raw.std(axis=0)
    std_corrected = movie_corrected.std(axis=0)
    fig, axes = subplots(2, 2, figsize=(12, 10))
    image_0 = axes[0, 0].imshow(mean_raw, cmap="gray")
    axes[0, 0].set_title("Mean raw")
    axes[0, 0].axis("off")
    colorbar(image_0, ax=axes[0, 0], fraction=0.046)
    image_1 = axes[0, 1].imshow(mean_corrected, cmap="gray")
    axes[0, 1].set_title("Mean corrected")
    axes[0, 1].axis("off")
    colorbar(image_1, ax=axes[0, 1], fraction=0.046)
    image_2 = axes[1, 0].imshow(std_raw, cmap="magma")
    axes[1, 0].set_title("Std raw")
    axes[1, 0].axis("off")
    colorbar(image_2, ax=axes[1, 0], fraction=0.046)
    image_3 = axes[1, 1].imshow(std_corrected, cmap="magma")
    axes[1, 1].set_title("Std corrected")
    axes[1, 1].axis("off")
    colorbar(image_3, ax=axes[1, 1], fraction=0.046)
    tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path / "03_stability_maps.png", dpi=150, bbox_inches="tight")
    close(fig)
    LOGGER.info(f"Saved: {output_path / '03_stability_maps.png'}")


def create_comparison_movie(
    movie_raw: ndarray,
    movie_corrected: ndarray,
    output_file: Path,
    frames_per_second: float,
):
    n_frames, height, width = movie_raw.shape
    intensity_min = min(float(movie_raw.min()), float(movie_corrected.min()))
    intensity_max = max(float(movie_raw.max()), float(movie_corrected.max()))
    LOGGER.info(
        f"Creating comparison: {n_frames} frames, {height}x{width} each, "
        f"range=[{intensity_min:.1f}, {intensity_max:.1f}]"
    )
    fig, axes = subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("black")
    image_raw = axes[0].imshow(
        movie_raw[0],
        cmap="gray",
        vmin=intensity_min,
        vmax=intensity_max,
    )
    axes[0].set_title("Not Corrected", fontsize=16, fontweight="bold", color="white")
    axes[0].axis("off")
    image_corrected = axes[1].imshow(
        movie_corrected[0],
        cmap="gray",
        vmin=intensity_min,
        vmax=intensity_max,
    )
    axes[1].set_title("Motion Corrected", fontsize=16, fontweight="bold", color="white")
    axes[1].axis("off")
    frame_text = fig.suptitle(f"Frame 0/{n_frames}", fontsize=12, color="white")
    tight_layout()
    writer = get_writer(str(output_file), format="FFMPEG", fps=frames_per_second, codec="libx264")

    for frame_index in range(n_frames):
        image_raw.set_data(movie_raw[frame_index])
        image_corrected.set_data(movie_corrected[frame_index])
        frame_text.set_text(f"Frame {frame_index + 1}/{n_frames}")
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            facecolor=fig.get_facecolor(),
            dpi=100,
            bbox_inches="tight",
        )
        buffer.seek(0)
        frame_image = imread(buffer)
        buffer.close()

        if frame_image.shape[2] == RGBA_CHANNELS:
            frame_image = frame_image[:, :, :3]

        writer.append_data(frame_image)

    writer.close()
    close(fig)
    LOGGER.info(f"Saved: {output_file}")
