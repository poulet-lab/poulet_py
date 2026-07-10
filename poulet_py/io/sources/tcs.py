
try:
    from numpy import concatenate, ndarray, zeros
    from pydantic import PrivateAttr

    from poulet_py import LOGGER, TCS, AcquisitionType, BaseSource, TCSStimulus, precise_sleep

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

            self._trigger_and_wait(st)
            self._drain_tcs_buffer()

        return True

    def _trigger_and_wait(self, stimulus: TCSStimulus) -> None:
        """
        Supports both TCS implementations:

        - new vk_tcs_fix-style TCS.trigger(stimulus, wait=True)
        - old speed_up_dev-style TCS.trigger(stimulus) plus source-side waiting
        """
        try:
            self.trigger(stimulus, wait=True)
            return

        except TypeError:
            # Backward-compatible path for the old trigger(stimulus) API.
            precise_sleep(stimulus.pre_delay / 1000.0)
            self.trigger(stimulus)

            while self.stimulus_running:
                precise_sleep(0.001)

            precise_sleep(stimulus.post_delay / 1000.0)

    def _drain_tcs_buffer(self) -> None:
        """
        Copy all newly acquired TCS samples from the TCS hardware ring buffer
        into the BaseSource buffer.

        This avoids read_sample(timeout=None), which can block forever if the
        stimulus ends while no new sample arrives.
        """
        cond = getattr(self, "_acquisition_cond", None)
        if cond is None:
            cond = getattr(self, "_sampling_cond")

        chunks = []

        with cond:
            count = self._tcs_buffer_idx - self._tcs_buffer_needle

            if count <= 0:
                LOGGER.warning("TCSSource %s: no TCS samples collected for stimulus", self.name)
                return

            size = self.buffer_size
            buffer = self._tcs_buffer
            needle = self._tcs_buffer_needle

            if count > size:
                LOGGER.warning(
                    "TCSSource %s: dropped %d TCS samples before drain",
                    self.name,
                    count - size,
                )
                needle = self._tcs_buffer_idx - size
                count = size

            start = needle % size
            end = start + count

            if end <= size:
                chunks.append(buffer[start:end].copy())
            else:
                first = size - start
                chunks.append(buffer[start:].copy())
                chunks.append(buffer[: count - first].copy())

            self._tcs_buffer_needle = needle + count

        if len(chunks) == 1:
            self._write_samples(chunks[0])
        else:
            self._write_samples(concatenate(chunks))
