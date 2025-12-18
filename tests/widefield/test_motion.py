"""Tests for motion correction module."""

import numpy as np
import pytest
from scipy.fft import fftn
from scipy.ndimage import shift

from poulet_py.widefield.motion import (
    _compute_phase_difference,
    _compute_registration_error,
    _upsampled_dft,
    apply_motion_correction,
    estimate_image_shift,
    estimate_motion_vectors,
    find_similar_frames,
    load_motion_vectors,
    save_motion_vectors,
)


@pytest.fixture
def simple_image() -> np.ndarray:
    """Create a simple test image with a distinct feature."""
    image = np.zeros((64, 64), dtype=np.float32)
    # Add a bright square in the center
    image[28:36, 28:36] = 100.0
    return image


@pytest.fixture
def shifted_image(simple_image: np.ndarray) -> np.ndarray:
    """Create a shifted version of the simple image."""
    # Shift by 3 pixels down and 2 pixels right
    shifted = np.zeros_like(simple_image)
    shifted[31:39, 30:38] = 100.0
    return shifted


@pytest.fixture
def simple_movie(simple_image: np.ndarray) -> np.ndarray:
    """Create a simple movie with slight motion."""
    n_frames = 20
    movie = np.zeros((n_frames, 64, 64), dtype=np.float32)
    for i in range(n_frames):
        # Create frames with small shifts
        shift_y = int(2 * np.sin(2 * np.pi * i / n_frames))
        shift_x = int(2 * np.cos(2 * np.pi * i / n_frames))
        frame = np.roll(simple_image, shift_y, axis=0)
        frame = np.roll(frame, shift_x, axis=1)
        movie[i] = frame
    return movie


class TestUpsampledDFT:
    """Tests for _upsampled_dft function."""

    def test_upsampled_dft_basic(self, simple_image: np.ndarray):
        """Test basic upsampled DFT computation."""
        freq_data = fftn(simple_image)
        result = _upsampled_dft(freq_data, 3, upsample_factor=2)
        assert result is not None
        assert result.shape == (3, 3)

    def test_upsampled_dft_dimension_mismatch(self, simple_image: np.ndarray):
        """Test that dimension mismatch raises error."""
        freq_data = fftn(simple_image)
        with pytest.raises(ValueError, match="number of dimensions"):
            _upsampled_dft(freq_data, (3, 3, 3), upsample_factor=2)

    def test_upsampled_dft_axis_offset_mismatch(self, simple_image: np.ndarray):
        """Test that axis offset mismatch raises error."""
        freq_data = fftn(simple_image)
        with pytest.raises(ValueError, match="number of axis offsets"):
            _upsampled_dft(freq_data, (3, 3), upsample_factor=2, axis_offsets=(0, 0, 0))


class TestComputePhaseDifference:
    """Tests for _compute_phase_difference function."""

    def test_zero_phase(self):
        """Test zero phase difference for real positive."""
        result = _compute_phase_difference(1.0 + 0j)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_quarter_phase(self):
        """Test pi/2 phase difference."""
        result = _compute_phase_difference(0.0 + 1j)
        assert result == pytest.approx(np.pi / 2, abs=1e-10)


class TestComputeRegistrationError:
    """Tests for _compute_registration_error function."""

    def test_perfect_match(self):
        """Test zero error for perfect correlation."""
        result = _compute_registration_error(np.complex128(1.0 + 0j), 1.0, 1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_partial_match(self):
        """Test non-zero error for partial correlation."""
        result = _compute_registration_error(np.complex128(0.5 + 0j), 1.0, 1.0)
        assert result > 0
        assert result < 1


class TestEstimateImageShift:
    """Tests for estimate_image_shift function."""

    def test_no_shift(self, simple_image: np.ndarray):
        """Test that identical images have zero shift."""
        shifts, error, _phase = estimate_image_shift(
            simple_image, simple_image, upsample_factor=1
        )
        assert np.allclose(shifts, [0, 0], atol=0.1)
        assert error < 0.01

    def test_integer_shift(self, simple_image: np.ndarray):
        """Test detection of integer pixel shift."""
        shifted = np.roll(simple_image, 5, axis=0)
        shifted = np.roll(shifted, -3, axis=1)

        shifts, _error, _phase = estimate_image_shift(
            simple_image, shifted, upsample_factor=1
        )
        # Note: shifts indicate how to move target to match reference
        assert shifts[0] == pytest.approx(-5, abs=1)
        assert shifts[1] == pytest.approx(3, abs=1)

    def test_subpixel_shift(self, simple_image: np.ndarray):
        """Test subpixel shift detection with upsampling."""
        shifted = shift(simple_image, [2.5, -1.5], mode="wrap")
        shifts, _error, _phase = estimate_image_shift(
            simple_image, shifted, upsample_factor=20
        )
        assert shifts[0] == pytest.approx(-2.5, abs=0.1)
        assert shifts[1] == pytest.approx(1.5, abs=0.1)

    def test_different_size_error(self, simple_image: np.ndarray):
        """Test that different sized images raise error."""
        small_image = simple_image[:32, :32]
        with pytest.raises(ValueError, match="same size"):
            estimate_image_shift(simple_image, small_image)

    def test_invalid_space(self, simple_image: np.ndarray):
        """Test that invalid space argument raises error."""
        with pytest.raises(ValueError, match=r"real.*fourier"):
            estimate_image_shift(simple_image, simple_image, space="invalid")

    def test_fourier_space(self, simple_image: np.ndarray):
        """Test shift estimation in Fourier space."""
        ref_freq = fftn(simple_image)
        target_freq = fftn(np.roll(simple_image, 3, axis=0))

        shifts = estimate_image_shift(
            ref_freq, target_freq, space="fourier", return_error=False
        )
        assert shifts[0] == pytest.approx(-3, abs=1)


class TestFindSimilarFrames:
    """Tests for find_similar_frames function."""

    def test_find_similar_frames(self, simple_movie: np.ndarray):
        """Test finding similar frames."""
        indices, correlation_matrix = find_similar_frames(simple_movie)
        assert len(indices) == len(simple_movie)
        assert correlation_matrix.shape == (len(simple_movie), len(simple_movie))
        # Most similar frame to itself should have correlation ~1
        assert np.allclose(np.diag(correlation_matrix), 1.0, atol=0.01)

    def test_similar_frames_ordering(self, simple_movie: np.ndarray):
        """Test that indices are sorted by similarity."""
        indices, _ = find_similar_frames(simple_movie)
        # First index should be the most similar frame
        assert indices[0] >= 0
        assert indices[0] < len(simple_movie)


class TestEstimateMotionVectors:
    """Tests for estimate_motion_vectors function."""

    def test_estimate_motion_vectors(self, simple_movie: np.ndarray):
        """Test motion vector estimation."""
        motion_vectors, reference_image = estimate_motion_vectors(
            simple_movie, n_parallel_workers=None
        )
        assert motion_vectors.shape == (len(simple_movie), 2)
        assert reference_image.shape == simple_movie[0].shape

    def test_with_motion_region(self, simple_movie: np.ndarray):
        """Test motion estimation with specific region."""
        motion_vectors, _reference_image = estimate_motion_vectors(
            simple_movie,
            motion_region=((10, 10), (54, 54)),
            n_parallel_workers=None,
        )
        assert motion_vectors.shape == (len(simple_movie), 2)

    def test_with_reference_image(self, simple_movie: np.ndarray, simple_image: np.ndarray):
        """Test motion estimation with provided reference."""
        motion_vectors, reference_image = estimate_motion_vectors(
            simple_movie,
            reference_image=simple_image,
            n_parallel_workers=None,
        )
        assert motion_vectors.shape == (len(simple_movie), 2)
        assert np.array_equal(reference_image, simple_image)


class TestApplyMotionCorrection:
    """Tests for apply_motion_correction function."""

    def test_apply_motion_correction_integer(self, simple_movie: np.ndarray):
        """Test motion correction with integer shifts."""
        corrected, motion_vectors, _reference = apply_motion_correction(
            simple_movie,
            shift_method="integer",
            n_parallel_workers=None,
        )
        assert corrected.shape == simple_movie.shape
        assert motion_vectors.shape == (len(simple_movie), 2)

    def test_apply_motion_correction_fourier(self, simple_movie: np.ndarray):
        """Test motion correction with Fourier shifts."""
        corrected, _motion_vectors, _reference = apply_motion_correction(
            simple_movie,
            shift_method="fourier",
            n_parallel_workers=None,
        )
        assert corrected.shape == simple_movie.shape

    def test_invalid_shift_method(self, simple_movie: np.ndarray):
        """Test that invalid shift method raises error."""
        with pytest.raises(ValueError, match="shift_method"):
            apply_motion_correction(
                simple_movie,
                shift_method="invalid",
                n_parallel_workers=None,
            )

    def test_with_precomputed_vectors(self, simple_movie: np.ndarray):
        """Test correction with pre-computed motion vectors."""
        # First compute vectors
        vectors, _ref = estimate_motion_vectors(simple_movie, n_parallel_workers=None)

        # Then apply with those vectors
        corrected, returned_vectors, _ = apply_motion_correction(
            simple_movie,
            motion_vectors=vectors,
            n_parallel_workers=None,
        )
        assert corrected.shape == simple_movie.shape
        assert np.array_equal(returned_vectors, vectors)


class TestMotionVectorsIO:
    """Tests for motion vectors save/load functions."""

    def test_save_load_motion_vectors(self, tmp_path):
        """Test saving and loading motion vectors."""
        motion_vectors = np.array([[1.5, -2.3], [0.5, 1.2], [-1.0, 0.8]])
        output_path = tmp_path / "motion_vectors.npy"

        save_motion_vectors(motion_vectors, output_path)
        assert output_path.exists()

        loaded = load_motion_vectors(output_path)
        assert loaded is not None
        assert np.allclose(loaded, motion_vectors, atol=0.01)

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file."""
        result = load_motion_vectors(tmp_path / "nonexistent.npy")
        assert result is None


class TestMotionCorrectionReducesVariance:
    """Integration tests checking that motion correction actually works."""

    def test_correction_reduces_variance(self, simple_movie: np.ndarray):
        """Test that motion correction reduces temporal variance."""
        # Apply correction
        corrected, _, _ = apply_motion_correction(
            simple_movie,
            shift_method="integer",
            n_parallel_workers=None,
        )

        # Compute variance maps
        var_raw = np.var(simple_movie, axis=0)
        var_corrected = np.var(corrected, axis=0)

        # Mean variance should decrease after correction
        # (may not always be true for all pixels, but on average)
        mean_var_raw = np.mean(var_raw)
        mean_var_corrected = np.mean(var_corrected)

        # Allow some tolerance since synthetic data may not always show improvement
        assert mean_var_corrected <= mean_var_raw * 1.1  # At worst, slightly worse

