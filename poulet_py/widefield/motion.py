"""
Motion correction for widefield imaging data.

This module provides functions for motion correction of widefield imaging movies
using FFT-based cross-correlation with subpixel precision.
"""

from multiprocessing import Pool
from pathlib import Path

import imageio
import numpy as np
from scipy.fft import fftfreq, fftn, ifftn
from scipy.ndimage import fourier_shift

from poulet_py import LOGGER


def _upsampled_dft(
    frequency_domain_data: np.ndarray,
    upsampled_region_size: int | tuple,
    upsample_factor: int = 1,
    axis_offsets: tuple | None = None,
) -> np.ndarray:
    """
    Compute upsampled DFT by matrix multiplication.

    This function provides the same result as embedding the array in a larger
    array (upsample_factor times larger), taking the FFT, and extracting a region.
    It achieves this more efficiently by computing the DFT in the output array
    without zero-padding.

    Parameters
    ----------
    frequency_domain_data : ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : int or tuple of int
        The size of the region to be sampled. If one integer is provided, it
        is duplicated up to the dimensionality of the data.
    upsample_factor : int, optional
        The upsampling factor. Defaults to 1.
    axis_offsets : tuple of int, optional
        The offsets of the region to be sampled. Defaults to None (uses image center).

    Returns
    -------
    upsampled_dft : ndarray
        The upsampled DFT of the specified region.
    """
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * frequency_domain_data.ndim
    elif len(upsampled_region_size) != frequency_domain_data.ndim:
        raise ValueError(
            "shape of upsampled region sizes must be equal to input data's number of dimensions."
        )

    if axis_offsets is None:
        axis_offsets = [0] * frequency_domain_data.ndim
    elif len(axis_offsets) != frequency_domain_data.ndim:
        raise ValueError(
            "number of axis offsets must be equal to input data's number of dimensions."
        )

    imaginary_2pi = 1j * 2 * np.pi

    dimension_properties = list(
        zip(frequency_domain_data.shape, upsampled_region_size, axis_offsets, strict=False)
    )

    for n_items, ups_size, ax_offset in dimension_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(n_items, upsample_factor)
        kernel = np.exp(-imaginary_2pi * kernel)

        frequency_domain_data = np.tensordot(kernel, frequency_domain_data, axes=(1, -1))
    return frequency_domain_data


def _compute_phase_difference(cross_correlation_max: complex) -> float:
    """
    Compute global phase difference between two images.

    Should be zero if images are non-negative.

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.

    Returns
    -------
    phase_difference : float
        Phase difference in radians.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_registration_error(
    cross_correlation_max: complex, source_amplitude: float, target_amplitude: float
) -> float:
    """
    Compute RMS error metric between source and target images.

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    source_amplitude : float
        The normalized average image intensity of the source image.
    target_amplitude : float
        The normalized average image intensity of the target image.

    Returns
    -------
    error : float
        Translation invariant normalized RMS error.
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / (
        source_amplitude * target_amplitude
    )
    return np.sqrt(np.abs(error))


def estimate_image_shift(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    upsample_factor: int = 1,
    space: str = "real",
    return_error: bool = True,
) -> tuple | np.ndarray:
    """
    Estimate subpixel image translation shift by cross-correlation.

    Efficiently computes the shift required to align target_image to reference_image
    using FFT-based cross-correlation with subpixel precision. The algorithm obtains
    an initial estimate of the cross-correlation peak by FFT and then refines the
    shift estimation by upsampling the DFT in a small neighborhood.

    Parameters
    ----------
    reference_image : ndarray
        Reference image (the fixed image).
    target_image : ndarray
        Image to align (will be shifted to match reference). Must be same
        dimensionality as reference_image.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within 1/upsample_factor
        of a pixel. For example upsample_factor=20 means registration within
        1/20th of a pixel. Default is 1 (no upsampling).
    space : str, optional
        Defines how the algorithm interprets input data. "real" means data will
        be FFT'd to compute the correlation, while "fourier" data will bypass FFT
        of input data. Case insensitive. Default is "real".
    return_error : bool, optional
        If True, returns error and phase difference in addition to shifts.
        If False, only shifts are returned. Default is True.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register target_image with
        reference_image. Axis ordering is consistent with numpy (e.g. Z, Y, X).
        Shape is (n_dimensions,).
    error : float, optional
        Translation invariant normalized RMS error between images.
        Only returned if return_error=True.
    phase_difference : float, optional
        Global phase difference between the two images (should be zero if
        images are non-negative). Only returned if return_error=True.

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    """
    if reference_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for estimate_image_shift")

    if space.lower() == "fourier":
        reference_frequency = reference_image
        target_frequency = target_image
    elif space.lower() == "real":
        reference_frequency = fftn(reference_image)
        target_frequency = fftn(target_image)
    else:
        raise ValueError(
            'Error: estimate_image_shift only knows the "real" '
            'and "fourier" values for the ``space`` argument.'
        )

    image_shape = reference_frequency.shape
    image_product = reference_frequency * target_frequency.conj()
    cross_correlation = ifftn(image_product)

    maxima_indices = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in image_shape])

    shift_vector = np.array(maxima_indices, dtype=np.float64)
    shift_vector[shift_vector > midpoints] -= np.array(image_shape)[shift_vector > midpoints]

    if upsample_factor == 1:
        if return_error:
            source_amplitude = np.sum(np.abs(reference_frequency) ** 2) / reference_frequency.size
            target_amplitude = np.sum(np.abs(target_frequency) ** 2) / target_frequency.size
            cross_correlation_max = cross_correlation[maxima_indices]
    else:
        shift_vector = np.round(shift_vector * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dft_shift = np.fix(upsampled_region_size / 2.0)
        upsample_factor_float = np.array(upsample_factor, dtype=np.float64)
        normalization = reference_frequency.size * upsample_factor_float**2
        sample_region_offset = dft_shift - shift_vector * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        cross_correlation /= normalization
        maxima_indices = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        cross_correlation_max = cross_correlation[maxima_indices]

        maxima_indices = np.array(maxima_indices, dtype=np.float64) - dft_shift

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
            _compute_registration_error(cross_correlation_max, source_amplitude, target_amplitude),
            _compute_phase_difference(cross_correlation_max),
        )
    else:
        return shift_vector


def find_similar_frames(initial_frames: np.ndarray) -> tuple:
    """
    Find the indices of the 30 most similar frames.

    Uses cross-correlation to identify frames that are most similar to each other,
    which are then used to create a stable reference image.

    Parameters
    ----------
    initial_frames : ndarray
        Array of frames of shape (n_frames, height, width).

    Returns
    -------
    top_frame_indices : ndarray
        Indices of frames sorted by similarity (most similar first).
    correlation_matrix : ndarray
        Cross-correlation matrix between all frame pairs.
    """
    frames_flat = np.reshape(initial_frames, (initial_frames.shape[0], -1)).astype("float32")

    frames_flat = frames_flat - np.reshape(frames_flat.mean(axis=1), (frames_flat.shape[0], 1))

    correlation_matrix = frames_flat @ frames_flat.T

    standard_deviations = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix = correlation_matrix / np.outer(standard_deviations, standard_deviations)

    correlation_sorted = -np.sort(-correlation_matrix, axis=1)
    correlation_top_mean = np.mean(correlation_sorted[:, 1:150], axis=1)
    index_max_correlation = np.argmax(correlation_top_mean)

    top_frame_indices = np.argsort(-correlation_matrix[index_max_correlation, :])

    return top_frame_indices, correlation_matrix


def _worker_cross_correlation(
    frame_batch: np.ndarray,
    reference_frequency: np.ndarray,
) -> list:
    """
    Worker function for parallel motion estimation.

    Computes motion vectors for a batch of frames using cross-correlation
    with a reference image.

    Parameters
    ----------
    frame_batch : ndarray
        Batch of frames of shape (n_frames_batch, height, width).
    reference_frequency : ndarray
        FFT of the reference image.

    Returns
    -------
    motion_vectors : list
        List of motion vectors, one per frame in the batch.
    """
    motion_vectors = []
    for frame_idx in range(len(frame_batch)):
        motion_vectors.append(
            estimate_image_shift(
                reference_frequency,
                fftn(frame_batch[frame_idx]),
                upsample_factor=20,
                space="fourier",
                return_error=False,
            )
        )
    return motion_vectors


def estimate_motion_vectors(
    movie: np.ndarray,
    motion_region: tuple | None = None,
    n_parallel_workers: int | None = 4,
    reference_image: np.ndarray | None = None,
) -> tuple:
    """
    Estimate motion vectors for each frame in a movie.

    Calculates the shift required to align each frame to a reference image,
    but does not apply the corrections. The reference image is created from
    the 30 most similar frames if not provided.

    Parameters
    ----------
    movie : ndarray
        Input movie of shape (n_frames, height, width).
    motion_region : tuple, optional
        Region of interest for motion calculation as ((row_start, col_start), (row_end, col_end)).
        If None, uses the entire image. Default is None.
    n_parallel_workers : int or None, optional
        Number of parallel workers for processing. If None, runs sequentially.
        Default is 4.
    reference_image : ndarray, optional
        Reference image to align frames to. If None, will be calculated from
        the 30 most similar frames. Default is None.

    Returns
    -------
    motion_vectors : ndarray
        Motion vectors array of shape (n_frames, 2) with [row_shift, col_shift] per frame.
        Positive values indicate the frame needs to be shifted down/right to align.
    reference_image : ndarray
        Reference image used for motion calculation.
    """
    movie_copy = movie.copy()

    if motion_region is None:
        row_slice = slice(None)
        col_slice = slice(None)
    else:
        row_slice = slice(motion_region[0][0], motion_region[1][0])
        col_slice = slice(motion_region[0][1], motion_region[1][1])

    if reference_image is None:
        n_frames = movie_copy.shape[0]
        if n_frames > 500:
            random_generator = np.random.default_rng()
            frame_indices = random_generator.choice(n_frames, 500, replace=False)
        else:
            frame_indices = slice(None)

        initial_reference_frames = movie_copy[frame_indices, row_slice, col_slice]
        similar_frame_indices, _ = find_similar_frames(initial_reference_frames)
        reference_image = np.mean(initial_reference_frames[similar_frame_indices[:30]], axis=0)

    reference_frequency = fftn(reference_image)

    if n_parallel_workers is None:
        motion_vectors = []
        for frame_idx in range(len(movie_copy)):
            motion_vectors.append(
                estimate_image_shift(
                    reference_frequency,
                    fftn(movie_copy[frame_idx, row_slice, col_slice]),
                    upsample_factor=20,
                    space="fourier",
                    return_error=False,
                )
            )
    else:
        frame_batches = []
        for frame_idx in range(0, movie_copy.shape[0], 50):
            frame_batches.append(
                (
                    movie_copy[frame_idx : (frame_idx + 50), row_slice, col_slice],
                    reference_frequency,
                )
            )

        with Pool(processes=n_parallel_workers) as pool:
            motion_vectors = np.array(pool.starmap(_worker_cross_correlation, frame_batches))

    motion_vectors = np.vstack(motion_vectors)

    return motion_vectors, reference_image


def apply_motion_correction(
    movie: np.ndarray,
    motion_region: tuple | None = None,
    motion_vectors: np.ndarray | None = None,
    shift_method: str = "integer",
    n_parallel_workers: int | None = 4,
    reference_image: np.ndarray | None = None,
) -> tuple:
    """
    Apply motion correction to a movie.

    Aligns each frame of the movie to a reference image using the estimated
    motion vectors. If motion vectors are not provided, they will be calculated
    first.

    Parameters
    ----------
    movie : ndarray
        Input movie of shape (n_frames, height, width).
    motion_region : tuple, optional
        Region of interest for motion calculation as ((row_start, col_start), (row_end, col_end)).
        If None, uses the entire image. Only used if motion_vectors is None.
        Default is None.
    motion_vectors : ndarray, optional
        Pre-computed motion vectors of shape (n_frames, 2) with [row_shift, col_shift].
        If None, motion vectors will be calculated. Default is None.
    shift_method : str, optional
        Method for applying shifts: "integer" uses np.roll (faster, integer pixel shifts),
        "fourier" uses fourier_shift (slower, subpixel precision). Default is "integer".
    n_parallel_workers : int or None, optional
        Number of parallel workers for motion estimation. Only used if motion_vectors is None.
        If None, runs sequentially. Default is 4.
    reference_image : ndarray, optional
        Reference image to align frames to. Only used if motion_vectors is None.
        If None, will be calculated from the 30 most similar frames. Default is None.

    Returns
    -------
    corrected_movie : ndarray
        Motion-corrected movie of shape (n_frames, height, width).
    motion_vectors : ndarray
        Motion vectors array of shape (n_frames, 2) used for correction.
    reference_image : ndarray
        Reference image used for alignment.
    """
    movie_copy = movie.copy()

    if motion_vectors is None:
        if motion_region is None:
            row_slice = slice(None)
            col_slice = slice(None)
        else:
            row_slice = slice(motion_region[0][0], motion_region[1][0])
            col_slice = slice(motion_region[0][1], motion_region[1][1])

        if reference_image is None:
            n_frames = movie_copy.shape[0]
            if n_frames > 500:
                random_generator = np.random.default_rng()
                frame_indices = random_generator.choice(n_frames, 500, replace=False)
            else:
                frame_indices = slice(None)

            initial_reference_frames = movie_copy[frame_indices, row_slice, col_slice]
            similar_frame_indices, _ = find_similar_frames(initial_reference_frames)
            reference_image = np.mean(initial_reference_frames[similar_frame_indices[:30]], axis=0)

        reference_frequency = fftn(reference_image)

        if n_parallel_workers is None:
            motion_vectors = []
            for frame_idx in range(len(movie_copy)):
                motion_vectors.append(
                    estimate_image_shift(
                        reference_frequency,
                        fftn(movie_copy[frame_idx, row_slice, col_slice]),
                        upsample_factor=20,
                        space="fourier",
                        return_error=False,
                    )
                )
        else:
            frame_batches = []
            for frame_idx in range(0, movie_copy.shape[0], 50):
                frame_batches.append(
                    (
                        movie_copy[frame_idx : (frame_idx + 50), row_slice, col_slice],
                        reference_frequency,
                    )
                )

            with Pool(processes=n_parallel_workers) as pool:
                motion_vectors = np.array(pool.starmap(_worker_cross_correlation, frame_batches))

        motion_vectors = np.vstack(motion_vectors)

    if shift_method == "fourier":
        for frame_idx in range(movie_copy.shape[0]):
            shifted_frame_frequency = fourier_shift(
                fftn(movie_copy[frame_idx]), motion_vectors[frame_idx]
            )
            movie_copy[frame_idx] = ifftn(shifted_frame_frequency).real
    elif shift_method == "integer":
        integer_motion = np.round(motion_vectors).astype(np.int32)
        for frame_idx in range(movie_copy.shape[0]):
            movie_copy[frame_idx] = np.roll(
                movie_copy[frame_idx], integer_motion[frame_idx], axis=(0, 1)
            )
    else:
        LOGGER.error(f"shift_method '{shift_method}' not recognized. Use 'fourier' or 'integer'.")
        raise ValueError(f"shift_method must be 'fourier' or 'integer', got '{shift_method}'")

    return movie_copy, motion_vectors, reference_image


def load_motion_vectors(motion_vectors_path: Path) -> np.ndarray | None:
    """
    Load previously saved motion vectors.

    Parameters
    ----------
    motion_vectors_path : Path
        Path to the motion_vectors.npy file.

    Returns
    -------
    motion_vectors : ndarray or None
        Motion vectors array of shape (n_frames, 2) or None if not found.
        Values are in pixel units (stored as int16 × 100 for precision).
    """
    motion_vectors_path = Path(motion_vectors_path)

    if motion_vectors_path.exists():
        motion_vectors = np.load(motion_vectors_path) / 100
        return motion_vectors
    else:
        return None


def save_motion_vectors(motion_vectors: np.ndarray, output_path: Path) -> Path:
    """
    Save motion vectors.

    Motion vectors are saved as int16 multiplied by 100 for storage precision.

    Parameters
    ----------
    motion_vectors : ndarray
        Motion vectors array of shape (n_frames, 2) in pixel units.
    output_path : Path
        Path where to save the motion vectors (including filename).

    Returns
    -------
    output_path : Path
        Path where motion vectors were saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    motion_vectors_scaled = (100 * motion_vectors).astype(np.int16)
    np.save(output_path, motion_vectors_scaled)

    return output_path


def save_corrected_recording(corrected_movie: np.ndarray, output_path: Path) -> Path:
    """
    Save motion-corrected movie.

    Parameters
    ----------
    corrected_movie : ndarray
        Motion-corrected movie of shape (n_frames, height, width).
    output_path : Path
        Path where to save the corrected movie (including filename).

    Returns
    -------
    output_path : Path
        Path where the corrected movie was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, corrected_movie)

    return output_path


def save_reference_image(reference_image: np.ndarray, output_path: Path) -> Path:
    """
    Save the reference image used for motion correction.

    Parameters
    ----------
    reference_image : ndarray
        Reference image of shape (height, width).
    output_path : Path
        Path where to save the reference image (including filename).

    Returns
    -------
    output_path : Path
        Path where the reference image was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.imsave(output_path, reference_image.astype(np.uint16), prefer_uint8=False)

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
    """
    Save metadata about the motion correction parameters used.

    Creates a human-readable text file documenting how motion correction
    was performed, enabling reproducibility and understanding of the processing.

    Parameters
    ----------
    output_path : Path
        Path where to save the metadata file (including filename).
    shift_method : str
        Shift method used: "integer" (np.roll) or "fourier" (subpixel via fourier_shift).
    n_frames : int
        Number of frames in the movie.
    image_shape : tuple
        Shape of each frame (height, width).
    raw_trial_path : Path, optional
        Path to the raw trial folder (for documentation purposes).
    upsample_factor : int, optional
        Upsampling factor for subpixel registration. Default is 20.
    n_reference_frames : int, optional
        Number of frames used to create reference image. Default is 30.
    motion_region : tuple, optional
        Region used for motion estimation (row_start, row_end, col_start, col_end).
        None means full frame was used.

    Returns
    -------
    output_path : Path
        Path where the metadata file was saved.
    """
    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shift_method_description = {
        "integer": "Integer pixel shifts using np.roll (fast, no interpolation)",
        "fourier": "Subpixel shifts using Fourier phase shifting (slower, interpolated)",
    }

    metadata_lines = [
        "MOTION CORRECTION METADATA",
        "=" * 50,
        f"Date processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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


def save_motion_analysis(motion_vectors: np.ndarray, output_path: Path) -> Path:
    """
    Save motion analysis data for future analysis without re-running correction.

    Saves the motion vectors along with computed displacement, allowing
    post-hoc analysis of motion characteristics.

    Parameters
    ----------
    motion_vectors : ndarray
        Motion vectors of shape (n_frames, 2) with [row_shift, col_shift] per frame.
    output_path : Path
        Path where to save the motion analysis (including filename).

    Returns
    -------
    output_path : Path
        Path where the motion analysis was saved.

    Notes
    -----
    The saved .npz file contains:
    - row_shift: Row shifts in pixels (n_frames,)
    - col_shift: Column shifts in pixels (n_frames,)
    - displacement: Euclidean displacement in pixels (n_frames,)
    - motion_vectors: Original motion vectors (n_frames, 2)

    Load with: data = np.load('motion_analysis.npz')
    Access arrays: data['row_shift'], data['col_shift'], data['displacement']
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_shift = motion_vectors[:, 0]
    col_shift = motion_vectors[:, 1]
    displacement = np.sqrt(row_shift**2 + col_shift**2)

    np.savez(
        output_path,
        row_shift=row_shift,
        col_shift=col_shift,
        displacement=displacement,
        motion_vectors=motion_vectors,
    )

    return output_path


def plot_motion_over_time(
    motion_vectors_raw: np.ndarray,
    movie_corrected: np.ndarray,
    reference_image: np.ndarray,
    output_path: Path,
):
    """
    Plot estimated motion over time for both raw and corrected data.

    Parameters
    ----------
    motion_vectors_raw : ndarray
        Motion vectors calculated from raw data, shape (n_frames, 2)
    movie_corrected : ndarray
        Motion-corrected movie
    reference_image : ndarray
        Reference image
    output_path : Path
        Output directory
    """
    import matplotlib.pyplot as plt

    LOGGER.info("Computing motion for corrected data...")
    fft_reference = fftn(reference_image)
    motion_vectors_corrected = []
    for frame_index in range(movie_corrected.shape[0]):
        shift = estimate_image_shift(
            fft_reference,
            fftn(movie_corrected[frame_index]),
            upsample_factor=20,
            space="fourier",
            return_error=False,
        )
        motion_vectors_corrected.append(shift)
    motion_vectors_corrected = np.array(motion_vectors_corrected)

    row_shift_raw = motion_vectors_raw[:, 0]
    col_shift_raw = motion_vectors_raw[:, 1]
    displacement_raw = np.sqrt(row_shift_raw**2 + col_shift_raw**2)

    row_shift_corrected = motion_vectors_corrected[:, 0]
    col_shift_corrected = motion_vectors_corrected[:, 1]
    displacement_corrected = np.sqrt(row_shift_corrected**2 + col_shift_corrected**2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

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
    output_path = Path(output_path)
    fig.savefig(output_path / "motion_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path / 'motion_over_time.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    n_frames = len(col_shift_raw)
    axes[0].scatter(col_shift_raw, row_shift_raw, s=5, alpha=0.5, c=range(n_frames), cmap="viridis")
    axes[0].axhline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[0].axvline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[0].set_xlabel("Col shift (px)")
    axes[0].set_ylabel("Row shift (px)")
    axes[0].set_title("Motion trajectory - Raw")
    axes[0].set_aspect("equal")
    plt.colorbar(axes[0].collections[0], ax=axes[0], label="Frame")

    axes[1].scatter(
        col_shift_corrected, row_shift_corrected, s=5, alpha=0.5, c=range(n_frames), cmap="viridis"
    )
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[1].axvline(0, linestyle="--", color="gray", linewidth=0.5)
    axes[1].set_xlabel("Col shift (px)")
    axes[1].set_ylabel("Row shift (px)")
    axes[1].set_title("Motion trajectory - Corrected")
    axes[1].set_aspect("equal")
    plt.colorbar(axes[1].collections[0], ax=axes[1], label="Frame")

    fig.tight_layout()
    fig.savefig(output_path / "motion_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path / 'motion_trajectory.png'}")


def plot_stability_maps(movie_raw: np.ndarray, movie_corrected: np.ndarray, output_path: Path):
    """
    Plot mean and standard deviation maps before and after correction.

    Parameters
    ----------
    movie_raw : ndarray
        Raw movie array
    movie_corrected : ndarray
        Motion-corrected movie array
    output_path : Path
        Output directory
    """
    import matplotlib.pyplot as plt

    mean_raw = movie_raw.mean(axis=0)
    mean_corrected = movie_corrected.mean(axis=0)

    std_raw = movie_raw.std(axis=0)
    std_corrected = movie_corrected.std(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    image_0 = axes[0, 0].imshow(mean_raw, cmap="gray")
    axes[0, 0].set_title("Mean raw")
    axes[0, 0].axis("off")
    plt.colorbar(image_0, ax=axes[0, 0], fraction=0.046)

    image_1 = axes[0, 1].imshow(mean_corrected, cmap="gray")
    axes[0, 1].set_title("Mean corrected")
    axes[0, 1].axis("off")
    plt.colorbar(image_1, ax=axes[0, 1], fraction=0.046)

    image_2 = axes[1, 0].imshow(std_raw, cmap="magma")
    axes[1, 0].set_title("Std raw")
    axes[1, 0].axis("off")
    plt.colorbar(image_2, ax=axes[1, 0], fraction=0.046)

    image_3 = axes[1, 1].imshow(std_corrected, cmap="magma")
    axes[1, 1].set_title("Std corrected")
    axes[1, 1].axis("off")
    plt.colorbar(image_3, ax=axes[1, 1], fraction=0.046)

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path / "stability_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved: {output_path / 'stability_maps.png'}")


def create_comparison_movie(
    movie_raw: np.ndarray, movie_corrected: np.ndarray, output_file: Path, frames_per_second: float
):
    """
    Create a side-by-side comparison movie with labels.

    Parameters
    ----------
    movie_raw : ndarray
        Raw movie array
    movie_corrected : ndarray
        Motion-corrected movie array
    output_file : Path
        Output file path
    frames_per_second : float
        Frame rate
    """
    from io import BytesIO

    import matplotlib.pyplot as plt

    n_frames, height, width = movie_raw.shape

    intensity_min = min(float(movie_raw.min()), float(movie_corrected.min()))
    intensity_max = max(float(movie_raw.max()), float(movie_corrected.max()))

    LOGGER.info(
        f"Creating comparison: {n_frames} frames, {height}x{width} each, range=[{intensity_min:.1f}, {intensity_max:.1f}]"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("black")

    image_raw = axes[0].imshow(movie_raw[0], cmap="gray", vmin=intensity_min, vmax=intensity_max)
    axes[0].set_title("Not Corrected", fontsize=16, fontweight="bold", color="white")
    axes[0].axis("off")

    image_corrected = axes[1].imshow(
        movie_corrected[0], cmap="gray", vmin=intensity_min, vmax=intensity_max
    )
    axes[1].set_title("Motion Corrected", fontsize=16, fontweight="bold", color="white")
    axes[1].axis("off")

    frame_text = fig.suptitle(f"Frame 0/{n_frames}", fontsize=12, color="white")

    plt.tight_layout()

    writer = imageio.get_writer(
        str(output_file), format="FFMPEG", fps=frames_per_second, codec="libx264"
    )

    for frame_index in range(n_frames):
        image_raw.set_data(movie_raw[frame_index])
        image_corrected.set_data(movie_corrected[frame_index])
        frame_text.set_text(f"Frame {frame_index + 1}/{n_frames}")

        buffer = BytesIO()
        fig.savefig(
            buffer, format="png", facecolor=fig.get_facecolor(), dpi=100, bbox_inches="tight"
        )
        buffer.seek(0)
        frame_image = imageio.v2.imread(buffer)
        buffer.close()

        if frame_image.shape[2] == 4:
            frame_image = frame_image[:, :, :3]

        writer.append_data(frame_image)

    writer.close()
    plt.close(fig)
    LOGGER.info(f"Saved: {output_file}")
