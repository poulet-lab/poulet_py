try:
    from numpy import ndarray
    from pydantic import PrivateAttr

    from poulet_py import TCS, BaseSource, TCSStimulus, precise_sleep
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

    def _close(self):
        TCS.close(self)

    def _fire(self) -> bool:
        for st in self._stimuli:
            if isinstance(st, TCSStimulus):
                precise_sleep(st.pre_delay / 1000.0)

                self.trigger(st)

                while self.stimulus_running:
                    precise_sleep(0.001)

                precise_sleep(st.post_delay / 1000.0)

                with self._sampling_cond:
                    count = self._tcs_buffer_idx - self._tcs_buffer_needle
                    if count <= 0:
                        return True

                    size = self.buffer_size
                    buffer = self._tcs_buffer
                    needle = self._tcs_buffer_needle

                    if count > size:
                        needle = self._tcs_buffer_idx - size
                        count = size

                    start = needle % size
                    end = start + count

                    if end <= size:
                        self._write_samples(buffer[start:end])
                    else:
                        first = size - start
                        self._write_samples(buffer[start:])
                        self._write_samples(buffer[: count - first])

                    self._tcs_buffer_needle = needle + count

        return True
