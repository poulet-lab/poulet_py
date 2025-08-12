from pickle import load
from unittest.mock import patch

import pytest
from h5py import File
from numpy import float32, ndarray, random, uint16

from poulet_py import Seq


@pytest.fixture
def sample_seq():
    """Fixture that mocks ONLY Seq.__iter__ with dummy frames."""
    # Generate dummy frames (10 frames of 480x640 uint16)
    dummy_frames = [random.randint(0, 256, (480, 640), dtype=uint16) for _ in range(10)]

    # Create a real Seq instance
    seq = Seq("dummy.seq")

    with patch.object(seq, "__iter__", return_value=iter(dummy_frames)):
        yield seq


def test_to_list(sample_seq: Seq):
    """Test `to_list()` returns a list of frames."""
    frames = sample_seq.to_list()
    assert isinstance(frames, list)
    assert all(isinstance(f, ndarray) for f in frames)


def test_to_numpy(sample_seq: Seq):
    """Test `to_numpy()` returns a 3D array."""
    arr = sample_seq.to_numpy()
    assert isinstance(arr, ndarray)
    assert arr.ndim == 3


def test_to_numpy_dtype(sample_seq: Seq):
    """Test dtype conversion in `to_numpy()`."""
    arr = sample_seq.to_numpy(dtype=float32)
    assert arr.dtype == float32


def test_to_hdf(sample_seq: Seq, tmp_path):
    """Test HDF5 export with metadata."""
    hdf_path = tmp_path / "test.h5"
    sample_seq.to_hdf(
        hdf_path,
        key="data",
        meta={"source": "FLIR", "fps": 30},
    )

    with File(hdf_path, "r") as f:
        assert "data" in f
        assert f["data"].attrs["source"] == "FLIR"


def test_to_pickle(sample_seq: Seq, tmp_path):
    """Test pickle export with compression."""
    pickle_path = tmp_path / "test.pkl.gz"
    sample_seq.to_pickle(pickle_path, compression="gzip")

    with open(pickle_path, "rb") as f:
        from gzip import decompress

        loaded = decompress(load(f))

    assert isinstance(loaded, ndarray)
