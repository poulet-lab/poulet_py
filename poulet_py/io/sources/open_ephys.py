try:
    from collections.abc import Sequence
    from threading import Thread
    from time import perf_counter_ns

    from numpy import dtype, float64, ndarray, uint64, zeros
    from open_ephys.control import OpenEphysHTTPServer
    from open_ephys.streaming import EventListener
    from pydantic import Field, IPvAnyAddress, PrivateAttr

    from poulet_py import (
        AcquisitionType,
        BaseSource,
        BaseStimulus,
        EmptyStimulus,
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


class OpenEphysSource(BaseSource):
    address: IPvAnyAddress = Field(default="127.0.0.1")
    port: int = Field(default=5557)
    buffer_size: int = Field(1000)
    num_channels: int = Field(default=16, description="Number of recording channels")

    _control: OpenEphysHTTPServer = PrivateAttr()
    _listener: EventListener = PrivateAttr()
    _thread: Thread = PrivateAttr()
    _sample_rate: int = PrivateAttr()
    _buffer: ndarray = PrivateAttr()
    _buffer_idx: int = PrivateAttr(default=0)
    _last_timestamp: int = PrivateAttr(default=0)

    def _init(self):
        self._buffer = zeros(
            self.buffer_size,
            dtype=dtype(
                [
                    ("timestamp", uint64),
                    ("channel_data", float64, (self.num_channels,)),
                ]
            ),
        )
        self._buffer_idx = 0
        self._last_timestamp = 0

        self._control = OpenEphysHTTPServer(str(self.address))
        self._sample_rate = int(self._control.get_audio_settings("sample_rate"))

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
            timestamp=perf_counter_ns(),
            channel_data=spike_data,
        )

    def _store_sample(self, timestamp: int, channel_data: ndarray):
        """
        Store a sample in the circular buffer

        Args:
            timestamp: Sample timestamp
            channel_data: Array of channel data
        """
        # Store data at current buffer position
        self._buffer[self._buffer_idx]["timestamp"] = timestamp
        self._buffer[self._buffer_idx]["channel_data"] = channel_data

        # Update buffer index (circular)
        self._buffer_idx = (self._buffer_idx + 1) % self.buffer_size

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        return [st for st in stimuli if isinstance(st, EmptyStimulus)]

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if not stimuli:
            return False

        for st in stimuli:
            if isinstance(st, EmptyStimulus) and self.acquisition_type == AcquisitionType.FINITE:
                self._control.acquire()
                precise_sleep((st.pre_delay + st.duration + st.post_delay + st._isi) / 1000.0)
                self._control.idle()

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            return self._publish_continuous(stimuli)
        elif self.acquisition_type == AcquisitionType.FINITE:
            return self._publish_finite(stimuli)

        return False

    def _publish_continuous(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer_idx == 0:
            return False

        start = self._last_timestamp
        end = self._buffer["timestamp"][self._buffer_idx]

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] < end)
        chunk = self._buffer[mask]

        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(
                name=self.name, payload={"open_ephys": chunk}, meta={"acquisition": "continuous"}
            )
        )

        self._last_timestamp = end

        return True

    def _publish_finite(self, stimuli: Sequence[BaseStimulus]) -> bool:
        if self._buffer_idx == 0:
            return False
        pre_ms = 0
        total_ms = 0
        for st in stimuli:
            if isinstance(st, EmptyStimulus):
                pre_ms += st.pre_delay
                total_ms += st.duration + st.post_delay + st._isi

        start = self._last_timestamp - pre_ms * 1_000_000
        end = self._last_timestamp + total_ms * 1_000_000

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] <= end)
        chunk = self._buffer[mask]

        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={"open_ephys": chunk}, meta={"acquisition": "finite"})
        )

        self._last_timestamp = end

        return True
