try:
    from collections.abc import Sequence

    from poulet_py import (
        AcquisitionType,
        BaseSource,
        BaseStimulus,
        NIAnalogBaseStimulus,
        NIAnalogCompositeStimulus,
        NIDaQ,
        NIDigitalBaseStimulus,
        NIDigitalCompositeStimulus,
        SinkEvent,
        precise_sleep,
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
            st
            for st in stimuli
            if isinstance(
                st,
                (
                    NIAnalogCompositeStimulus,
                    NIDigitalCompositeStimulus,
                    NIAnalogBaseStimulus,
                    NIDigitalBaseStimulus,
                ),
            )
        ]

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if not stimuli:
            return False

        for st in stimuli:
            if isinstance(
                st,
                (
                    NIAnalogCompositeStimulus,
                    NIDigitalCompositeStimulus,
                    NIAnalogBaseStimulus,
                    NIDigitalBaseStimulus,
                ),
            ):
                self.write(st)

        if self.acquisition_type == AcquisitionType.FINITE:
            self.start()

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if not stimuli and self.acquisition_type == AcquisitionType.FINITE:
            return False
        data = self.read(-1, -1)
        if data:
            self.publish(
                SinkEvent(
                    name=self.name, payload=data, meta={"acquisition": self.acquisition_type.value}
                )
            )

        if self.acquisition_type == AcquisitionType.FINITE:
            self.stop()

        isi = 0
        for st in stimuli:
            if isinstance(
                st,
                (
                    NIAnalogCompositeStimulus,
                    NIDigitalCompositeStimulus,
                    NIAnalogBaseStimulus,
                    NIDigitalBaseStimulus,
                ),
            ):
                isi = max(isi, st._isi)

        precise_sleep(isi / 1000.0)

        return True
