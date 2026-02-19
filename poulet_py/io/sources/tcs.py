from numpy import ndarray
from pydantic import PrivateAttr


try:
    from typing import Literal
    from collections.abc import Generator
    from secrets import choice
    from time import perf_counter_ns, time

    from numpy import array
    from pydantic import Field

    from poulet_py import LOGGER, TCS, BaseDataPacket, BaseSource, TCSStimulus
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
    sample_mode: 


    def _init(self):
        self.open()

    def _close(self):
        self.close()

    def next(self, stimulus: TCSStimulus):
        self.trigger(stimulus)

        start = perf_counter_ns()

        readings = []
        while self._should_continue_trial(start_time, stimulus.duration + interstimulus_period):
            reading = self.get_readings()
            if reading:
                reading["trial"] = idx
                readings.append(reading.copy())

        if self._sink is None:
            LOGGER.warning("CounterSource is not attached to a DataSink")
        else:
            timestamp = array([perf_counter_ns()], dtype="uint64")
            counter = array([self._counter], dtype="uint64")
            packet = BaseDataPacket(
                name=self.name,
                data={"counter": counter, "timestamp": timestamp},
                meta={"description": "Counter data"},
            )

            self._sink.push(packet)

        reading

    def _daq_thread(self):
