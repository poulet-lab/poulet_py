try:
    from pathlib import Path
    from pickle import HIGHEST_PROTOCOL
    from typing import Any

    from flirpy.io.fff import Fff
    from flirpy.io.seq import Seq as SeqBase
    from h5py import File
    from numpy import array, ndarray
    from pandas._typing import CompressionOptions
    from pandas.io.pickle import to_pickle as pd_to_pickle
except ImportError as e:
    msg = """
Missing 'seq' module. Install options:
- Dedicated:    pip install poulet_py[seq]
- Module:       pip install poulet_py[converters]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class Seq(SeqBase):
    """A sequence handler for FLIR thermal data with export capabilities.

    Extends `flirpy.io.seq.Seq` to add NumPy, HDF5, and pickle serialization.

    Parameters
    ----------
    input_file : str | Path
        Path to the input SEQ file.
    height : int, optional
        Force a specific image height. If None, uses the file's metadata.
    width : int, optional
        Force a specific image width. If None, uses the file's metadata.
    raw : bool, default=False
        If True, returns the raw bytes not Fff class.
    """

    def __init__(
        self,
        input_file: str | Path,
        *,
        height: int | None = None,
        width: int | None = None,
        raw: bool = False,
    ):
        super().__init__(input_file, height=height, width=width, raw=raw)

    def to_list(self) -> list:
        """Convert the sequence to a list of frames.

        Returns
        -------
        list
            List of frames (as ndarray or bytes if unprocessed).
        """
        return [frame.get_image() if isinstance(frame, Fff) else frame for frame in self]

    def to_numpy(self, *, dtype: type | None = None) -> ndarray:
        """Convert the sequence to a NumPy array.

        Parameters
        ----------
        dtype : type, optional
            Desired data type for the output array (e.g., `np.float32`).

        Returns
        -------
        ndarray
            Stacked frames as a 3D array (n_frames, height, width).
        """
        return array(self.to_list(), dtype=dtype)

    def to_hdf(
        self,
        path: str | Path,
        *,
        key,
        mode="w",
        dtype: type | None = None,
        complib=None,
        complevel=None,
        meta: dict[str, Any] | None = None,
    ):
        """Save the sequence to an HDF5 file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        key : str
            HDF5 dataset name.
        mode : str, default="w"
            File mode ('w', 'a', 'r+', etc.).
        dtype : type, optional
            Data type for the stored array.
        complib : str, optional
            Compression library (e.g., 'gzip', 'lzf').
        complevel : int, optional
            Compression level (1-9 for gzip).
        meta : dict, optional
            Metadata to store as HDF5 attributes.
        """
        data = self.to_numpy(dtype=dtype)
        with File(path, mode) as f:
            ds = f.create_dataset(key, data=data, compression=complib, compression_opts=complevel)

            if meta:
                for k, v in meta.items():
                    ds.attrs[k] = v

    def to_pickle(
        self,
        path: str | Path,
        *,
        dtype: type | None = None,
        compression: CompressionOptions = "infer",
        protocol: int = HIGHEST_PROTOCOL,
    ):
        """Save the sequence to a pickle file using pandas' I/O handler.

        Parameters
        ----------
        path : str | Path
            Output file path.
        dtype : type, optional
            Data type for the array.
        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data.
            If 'infer' and 'path' is path-like,
            then detect compression from the following extensions:
            '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to None for no compression.
            Can also be a dict with key 'method'
            set to one of {'zip', 'gzip', 'bz2', 'zstd', 'xz', 'tar'}
            and other key-value pairs are forwarded to zipfile.ZipFile,
            gzip.GzipFile, bz2.BZ2File, zstandard.ZstdCompressor,
            lzma.LZMAFile or tarfile.TarFile, respectively.
            As an example, the following could be passed for faster compression
            and to create a reproducible gzip archive:
            compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}
        protocol : int, default=HIGHEST_PROTOCOL
            Pickle protocol version.
        """
        data = self.to_numpy(dtype=dtype)
        pd_to_pickle(data, path, compression=compression, protocol=protocol)
