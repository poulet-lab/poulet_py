try:
    from time import monotonic_ns

    from pydantic import PrivateAttr

    from poulet_py import BaseSource
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class CounterSource(BaseSource):
    _counter: int = PrivateAttr(default=0)

    def _set_buffer_dtype(self):
        self._buffer_dtype = [("timestamp", "uint64"), ("counter", "uint64")]

    def _open(self):
        self._counter = 0

    def _close(self):
        pass

    def _trigger(self) -> bool:
        timestamp = monotonic_ns()
        self._counter += 1

        self._write_sample((timestamp, self._counter))

        return True
