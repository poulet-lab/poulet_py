try:
    from collections.abc import Sequence

    from pydantic import Field, PrivateAttr

    from poulet_py import TCS, AcquisitionType, BaseSource, BaseStimulus, SinkEvent, TCSStimulus, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TCSSource(BaseSource, TCS):
    name: str = Field("trial", description="Name of the trial source")

    _last_sent_idx: int = PrivateAttr(0)
    _last_timestamp: int = PrivateAttr(0)

    def _init(self):
        TCS.open(self)
        self._last_timestamp = 0

    def _close(self):
        TCS.close(self)

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        return [st for st in stimuli if isinstance(st, TCSStimulus)]

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer is not None and self.acquisition_type == AcquisitionType.FINITE:
            self._last_timestamp = self._buffer["timestamp"][
                (self._buffer_idx % self.buffer_size) - 1
            ]

        for st in stimuli:
            if isinstance(st, TCSStimulus):
                precise_sleep(st.pre_delay / 1000.0)
                self.trigger(st)
                self._stimulus_done.wait()

                precise_sleep((st._isi + st.post_delay) / 1000.0)

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer is None:
            return False

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            return self._publish_continuous(stimuli)
        elif self.acquisition_type == AcquisitionType.FINITE:
            return self._publish_finite(stimuli)

        return False

    def _publish_continuous(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer is None:
            return False

        start = self._last_timestamp
        end = self._buffer["timestamp"][(self._buffer_idx % self.buffer_size) - 1]

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] < end)
        chunk = self._buffer[mask]
        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={"tcs": chunk}, meta={"acquisition": "continuous"})
        )

        self._last_timestamp = end

        return True

    def _publish_finite(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer is None or not stimuli:
            return False

        total_ms = 0
        for st in stimuli:
            if isinstance(st, TCSStimulus):
                total_ms += st.pre_delay + st.duration + st.post_delay + st._isi

        start = self._last_timestamp
        end = self._last_timestamp + total_ms * 1_000_000

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] <= end)
        chunk = self._buffer[mask]

        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={"tcs": chunk}, meta={"acquisition": "finite"})
        )

        self._last_timestamp = end

        return True
