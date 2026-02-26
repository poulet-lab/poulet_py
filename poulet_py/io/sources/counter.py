try:
    from time import perf_counter_ns

    from numpy import array
    from pydantic import Field, PrivateAttr

    from poulet_py import BaseSource, BaseStimulus
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
        self._counter = 0

    def _next(self, stimulus: list[BaseStimulus]) -> int:
        self._counter += 1

        timestamp = array([perf_counter_ns()], dtype="uint64")
        counter = array([self._counter], dtype="uint64")

        self.publish(
            payload={"counter": counter, "timestamp": timestamp},
            meta={"description": "Counter data"},
        )

        return self._counter
