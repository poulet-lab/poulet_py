try:
    from time import monotonic_ns
    from numpy import zeros
    from poulet_py import DataSink, DataPacket, DataSource
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class NIAnalogInputDP(DataPacket):
    source: str
    t0_ns: int
    dt_ns: int
    data: np.ndarray
    channels: list[str]
    sample_index: int


class NIAnalogInputSource(DataSource):
    def __init__(self, task, reader, channels, samples_per_callback: int):
        self.task = task
        self.reader = reader
        self.channels = channels
        self.samples = samples_per_callback

        self.buffer = zeros((len(channels), samples_per_callback))

    def attach(self, sink: DataSink):
        dt_ns = int(1e9 / sample_rate)
        sample_index = 0

        def callback(task_handle, event_type, n_samples, cb_data):
            nonlocal sample_index
            reader.read_many_sample(self.buffer, n_samples)

            t0 = monotonic_ns()

            packet = NIAnalogInputDP(
                source="ai",
                t0_ns=t0,
                dt_ns=dt_ns,
                data=self.buffer.copy(),
                channels=self.channels,
                sample_index0=sample_index,
            )

            sample_index += n_samples
            sink.push(packet)

            return 0

        self.task.register_every_n_samples_acquired_into_buffer_event(self.samples, callback)
