try:
    from time import monotonic_ns

    from poulet_py import DCAM, LOGGER, AcquisitionType, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DCAMSource(DCAM, BaseSource):
    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
            (
                "dcam",
                self.DTYPE_MAP.get(self.pixel_type, "float32"),
                (self.__dcam_internal_buffer.height, self.__dcam_internal_buffer.width),
            ),
        ]

    def _open(self):
        DCAM.open(self)
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            # pointers
            self._source_buffer = self._dcam_buffer
            self._source_buffer_idx = self._dcam_buffer_idx
            self._source_buffer_needle = self._dcam_buffer_needle

    def _close(self):
        DCAM.close(self)

    def _trigger(self) -> bool:
        if self.acquisition_type == AcquisitionType.FINITE:
            deadline = monotonic_ns() + self._max_stimulus_duration_ms * 1000000
            while monotonic_ns() < deadline:
                sample = self.read_sample()
                if sample is None:
                    LOGGER.error("DcamSource error in reading sample")
                self._source_buffer[self._source_buffer_idx % self.buffer_size] = sample
        else:
            precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        return True
