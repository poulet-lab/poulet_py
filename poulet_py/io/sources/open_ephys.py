try:
    from threading import Thread
    from time import monotonic_ns

    from numpy import float64, ndarray, uint64, zeros
    from open_ephys.control import OpenEphysHTTPServer
    from open_ephys.streaming import EventListener
    from pydantic import Field, IPvAnyAddress, PrivateAttr

    from poulet_py import AcquisitionType, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class OpenEphysSource(BaseSource):
    address: IPvAnyAddress = Field(default="localhost")
    port: int = Field(default=5557)
    num_channels: int = Field(default=16, description="Number of recording channels")

    _control: OpenEphysHTTPServer = PrivateAttr()
    _listener: EventListener = PrivateAttr()
    _thread: Thread = PrivateAttr()

    def _set_buffer_dtype(self):
        self._buffer_dtype = [
            ("timestamp", uint64),
            ("channel_data", float64, (self.num_channels,)),
        ]

    def _open(self):
        self._control = OpenEphysHTTPServer(str(self.address))

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._control.acquire()

        self._listener = EventListener(str(self.address), self.port)
        self._thread = Thread(
            target=self._listener.start, args=(self.ttl_callback, self.spike_callback)
        )

        self._thread.start()

    def _close(self):
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._control.idle()

        self._listener.stop()
        self._thread.join()

    def ttl_callback(self, info: dict):
        print(info)

    def spike_callback(self, info: dict):
        """
        Handle spike events from Open Ephys

        Args:
            info: Dictionary containing spike event information
        """
        print(info)

        # Store spike event in buffer if needed
        # Spike data includes amplitude values for up to 4 channels
        spike_data = zeros(self.num_channels)
        for i in range(self.num_channels):
            amp_key = f"amp{i + 1}"
            if amp_key in info:
                spike_data[i] = info[amp_key]

        self._store_sample(
            timestamp=monotonic_ns(),
            channel_data=spike_data,
        )

    def _store_sample(self, timestamp: int, channel_data: ndarray):
        self._buffer[self._buffer_idx]["timestamp"] = timestamp
        self._buffer[self._buffer_idx]["channel_data"] = channel_data

        self._buffer_idx = (self._buffer_idx + 1) % self.buffer_size

    def _fire(self) -> bool:
        if self.acquisition_type == AcquisitionType.FINITE:
            self._control.acquire()
            precise_sleep(self._max_stimulus_duration_ms / 1000.0)
            self._control.idle()

        return True
