try:
    from collections.abc import Sequence

    from poulet_py import (
        AcquisitionType,
        BaseSource,
        BaseStimulus,
        NIAnalogBaseStimulus,
        NIDaQ,
        NIDigitalBaseStimulus,
        SinkEvent,
    )
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class NIDaQSource(BaseSource, NIDaQ):
    def _init(self):
        NIDaQ.open(self)
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.start()

    def _close(self):
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.stop()
        NIDaQ.close(self)

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        return [
            st for st in stimuli if isinstance(st, (NIAnalogBaseStimulus, NIDigitalBaseStimulus))
        ]

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        duration = 0

        for st in stimuli:
            if isinstance(st, (NIAnalogBaseStimulus, NIDigitalBaseStimulus)):
                self.write(st)
                current_duration = (st.duration + st.pre_delay + st.post_delay) / 1000
                duration = max(duration, current_duration)

        if self.acquisition_type == AcquisitionType.FINITE:
            self.start()

        self.wait(timeout=duration + 2)

        if self.acquisition_type == AcquisitionType.FINITE:
            self.stop()

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:

        data = self.read(-1, 0.01)
        if data:
            self.publish(
                SinkEvent(
                    name=self.name, payload=data, meta={"acquisition": self.acquisition_type.value}
                )
            )

        return True
