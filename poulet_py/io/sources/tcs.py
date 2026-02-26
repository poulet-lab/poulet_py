try:
    from numpy import empty

    from numpy import ndarray
    from pydantic import PrivateAttr

    from pydantic import Field

    from poulet_py import TCS, BaseSource, BaseStimulus, TCSStimulus, AcquisitionType
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
    pre_ms: int | None = Field(default=None)
    post_ms: int | None = Field(default=None)

    _last_sent_idx: int = PrivateAttr(0)

    def _init(self):
        if self.data is None:
            self.data = empty(
                self.buffer_size,
                dtype=[
                    ("timestamp", "uint64"),
                    ("temperature", "float32", (5,)),
                ],
            )
        self.open()

    def _close(self):
        self.close()

    def _next(self, stimulus: list[BaseStimulus]) -> ndarray:
        for st in stimulus:
            if isinstance(st, TCSStimulus):
                self.trigger(st)
                while True:
                    if not self.is_stimulus_running:
                        break

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            available = self._sampling_idx - self._last_sent_idx
            if available <= 0:
                return None

            start = self._last_sent_idx % self.buffer_size
            end = self._sampling_idx % self.buffer_size
            if start < end:
                chunk = self._samples_buffer[start:end]
            else:
                chunk = np.concatenate(
                    (
                        self._samples_buffer[start:],
                        self._samples_buffer[:end],
                    )
                )
            self._last_sent_idx = self._sampling_idx
            
            if self._samples_buffer is not None and self._sampling_idx > self._last_sent_idx:
                self.publish(
                    payload={"tcs": self._samples_buffer[self._last_sent_idx : self._sampling_idx]}
                )
                return self._samples_buffer[self._last_sent_idx : self._sampling_idx]

        elif self.acquisition_type == AcquisitionType.FINITE:
            pass
            # TODO
