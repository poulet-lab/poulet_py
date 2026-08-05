from numpy import ndarray, zeros
from pydantic import PrivateAttr

from poulet_py import DCAM, BaseSource


class DCAMSource(BaseSource, DCAM):
    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self) -> None:
        self._source_buffer_dtype = self._dcam_buffer.dtype

    def _open(self) -> None:
        DCAM.open(self)
        self._temp_buffer = zeros(self._dcam_buffer.size, dtype=self._dcam_buffer.dtype)

    def _close(self) -> None:
        DCAM.close(self)

    def _acquire(self) -> None:
        samples = self.read_many_sample(data=self._temp_buffer)
        self._write_samples(self._temp_buffer[:samples])
