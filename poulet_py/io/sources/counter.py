try:
    from time import monotonic_ns

    from pydantic import PrivateAttr

    from poulet_py import BaseSource, precise_sleep
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class CounterSource(BaseSource):
    _counter: int = PrivateAttr(default=0)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [("timestamp", "uint64"), ("counter", "uint64")]

    def _open(self):
        self._counter = 0

    def _close(self):
        pass

    def _fire(self) -> bool:
        self._counter += 1
        timestamp = monotonic_ns()

        self._write_sample((timestamp, self._counter))
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        return True
