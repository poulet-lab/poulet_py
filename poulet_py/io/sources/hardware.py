try:
    from time import monotonic_ns

    from numpy import ndarray, zeros
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, AcquisitionType, BaseSource, precise_sleep
    from poulet_py.hardware import BaseHardware
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class HardwareSource(BaseSource):
    device: BaseHardware = Field(...)

    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self) -> None:
        self._source_buffer_dtype = self.device._buffer.dtype

    def _open(self) -> None:
        self.device.open()
        self._temp_buffer = zeros(self.buffer_size, dtype=self.device._buffer.dtype)

    def _close(self) -> None:
        self.device.close()

    def _fire(self) -> bool:
        if self.device.acquisition_type == AcquisitionType.FINITE:
            deadline = monotonic_ns() + self._max_stimulus_duration_ms * 1000000

            while monotonic_ns() < deadline:
                sample = self.device.read_sample(timeout=1)

                if sample is None:
                    LOGGER.error(f"{type(self).__name__} error in reading sample, drop sample")
                    continue

                self._write_samples(sample)
        else:
            precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        if self.device.acquisition_type == AcquisitionType.CONTINUOUS:
            samples = self.device.read_many_sample(data=self._temp_buffer, n=-1, timeout=-1)
            if samples > 0:
                self._write_samples(self._temp_buffer[:samples])

        return True
