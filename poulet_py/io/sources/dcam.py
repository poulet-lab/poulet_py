try:
    from time import monotonic_ns

    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import DCAM, LOGGER, AcquisitionType, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DCAMSource(BaseSource, DCAM):
    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = self._dcam_buffer.dtype

    def _open(self):
        DCAM.open(self)
        self._temp_buffer = zeros(self.buffer_size, dtype=self._dcam_buffer.dtype)

    def _close(self):
        DCAM.close(self)

    def _fire(self) -> bool:
        if self.acquisition_type == AcquisitionType.FINITE:
            deadline = monotonic_ns() + self._max_stimulus_duration_ms * 1000000

            while monotonic_ns() < deadline:
                sample = self.read_sample()

                if sample is None:
                    LOGGER.error("DcamSource error in reading sample, drop frame")
                    continue

                self._write_samples(sample)
        else:
            precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            samples = self.read_many_sample(data=self._temp_buffer, n=-1, timeout=-1)
            if samples > 0:
                self._write_samples(self._temp_buffer[:samples])

        return True
