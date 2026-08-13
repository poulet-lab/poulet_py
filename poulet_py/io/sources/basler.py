try:
    from time import monotonic_ns

    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import LOGGER, AcquisitionType, BaseSource, Basler

except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class BaslerSource(BaseSource, Basler):
    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self) -> None:
        self._source_buffer_dtype = self._basler_buffer.dtype

    def _open(self) -> None:
        Basler.open(self)
        self._temp_buffer = zeros(self._basler_buffer.size, dtype=self._basler_buffer.dtype)

    def _close(self) -> None:
        Basler.close(self)

    def _fire(self) -> bool:
        if self.acquisition_type == AcquisitionType.FINITE:
            deadline = monotonic_ns() + self._max_stimulus_duration_ms * 1000000

            while monotonic_ns() < deadline:
                sample = self.read_sample()

                if sample is None:
                    LOGGER.error("DcamSource error in reading sample, drop frame")
                    continue

                self._write_samples(sample)

        return True

    def _acquire(self) -> None:
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            samples = self.read_many_sample(data=self._temp_buffer)
            self._write_samples(self._temp_buffer[:samples])
