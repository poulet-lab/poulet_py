try:
    from poulet_py import (
        AcquisitionType,
        BaseSource,
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
    def _set_buffer_dtype(self):
        self._buffer_dtype = []  # TODO

    def _open(self):
        NIDaQ.open(self)
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.start()

    def _close(self):
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self.stop()

        NIDaQ.close(self)

    def _fire(self) -> bool:
        for st in self._stimuli:
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

        data = self.read(-1, -1)

        if data:
            # TODO to buffer
            self.publish(
                SinkEvent(
                    name=self.name, payload=data, meta={"acquisition": self.acquisition_type.value}
                )
            )

        if self.acquisition_type == AcquisitionType.FINITE:
            self.stop()

        isi = 0
        for st in self._stimuli:
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
