try:
    from collections.abc import Sequence
    from time import monotonic_ns

    from numpy import array
    from pydantic import Field, PrivateAttr

    from poulet_py import BaseSource, BaseStimulus, SinkEvent
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class CounterSource(BaseSource):
    name: str = Field("trial", description="Name of the trial source")
    _counter: int = PrivateAttr(default=0)

    def _init(self):
        self._counter = 0

    def _close(self):
        pass

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        return stimuli

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if not stimuli:
            return False

        self._counter += 1

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if not stimuli:
            return False

        counter = array(
            [(monotonic_ns(), self._counter)],
            dtype=[("timestamp", "uint64"), ("counter", "uint64")],
        )

        self.publish(
            SinkEvent(
                name=self.name, payload={"counter": counter}, meta={"description": "Counter data"}
            )
        )

        return True
