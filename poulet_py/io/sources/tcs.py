try:
    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import LOGGER, TCS, AcquisitionType, BaseSource, TCSStimulus
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class TCSSource(BaseSource, TCS):
    _temp_buffer: ndarray = PrivateAttr()

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = self._tcs_buffer.dtype

    def _open(self):
        TCS.open(self)
        self._temp_buffer = zeros(self.buffer_size, dtype=self._tcs_buffer.dtype)

    def _close(self):
        TCS.close(self)

    def _fire(self) -> bool:
        # TODO precise_sleep if not stimulus?
        for st in self._stimuli:
            if isinstance(st, TCSStimulus):
                self.trigger(st)

            if self.acquisition_type == AcquisitionType.FINITE:
                while self.stimulus_running:
                    sample = self.read_sample()

                    if sample is None:
                        LOGGER.error(f"{type(self).__name__} error in reading sample")
                        continue

                    self._write_samples(sample)

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            samples = self.read_many_sample(data=self._temp_buffer, n=-1, timeout=-1)
            if samples > 0:
                self._write_samples(self._temp_buffer[:samples])

        return True
