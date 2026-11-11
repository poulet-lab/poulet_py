try:
    from numpy import ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import TCS, BaseSource, TCSStimulus

except ImportError as e:
    raise ImportError(
        """
Missing 'sources' module.

Install options:
- Dedicated: pip install poulet_py[sources]
- Module:    pip install poulet_py[io]
- Full:      pip install poulet_py[all]
"""
    ) from e


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
        for st in self._stimuli:
            if not isinstance(st, TCSStimulus):
                continue

            self.trigger(st, wait=True)

        return True

    def _acquire(self) -> None:
        samples = self.read_many_sample(data=self._temp_buffer)
        self._write_samples(self._temp_buffer[:samples])
