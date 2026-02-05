try:
    from time import monotonic_ns

    from poulet_py import DataPacket, DataSink, DataSource
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TrialSource(DataSource):
    def __init__(self, name: str = "trial"):
        self.name = name
        self._trial = 0
        self._sink = None

    def attach(self, sink: DataSink):
        self._sink = sink

    def next_trial(self, **metadata) -> int:
        self._trial += 1

        packet = DataPacket(
            source=self.name,
            kind="trial",
            t0_ns=monotonic_ns(),
            data={
                "trial": self._trial,
                **metadata,
            },
        )

        self._sink.push(packet)
        return self._trial
