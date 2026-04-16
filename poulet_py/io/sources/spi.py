try:
    from collections.abc import Sequence
    from threading import Lock, Thread
    from time import perf_counter_ns

    import spidev
    from numpy import empty, ndarray
    from pydantic import Field, PrivateAttr

    from poulet_py import (
        LOGGER,
        AcquisitionType,
        BaseSource,
        BaseStimulus,
        SinkEvent,
        SPIStimulus,
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


class SPISource(BaseSource):
    name: str = Field(..., description="Name of the SPI source")
    spi_bus: int = Field(default=0, description="spi bus")
    spi_cs: int = Field(default=0, description="spi cs")
    bits_per_word: int = Field(default=8, description="Bits per word for SPI communication")
    max_speed_hz: int = Field(default=12_500_000, description="Maximum SPI clock speed in Hz")
    mode: int = Field(default=0, description="SPI mode (0-3)", ge=0, le=3)
    cs_high: bool = Field(default=False, description="Chip select active high")
    loop: bool = Field(default=False, description="Loopback mode enabled")
    no_cs: bool = Field(default=False, description="No chip select")
    lsb_first: bool = Field(default=False, description="LSB first bit order")
    three_wire: bool = Field(default=False, description="Three-wire mode enabled")
    read0: bool = Field(default=False, description="Read 0 after transfer")
    mosi_idle_low: bool = Field(default=False, description="MOSI idle low")
    buffer_size: int = Field(default=1000, description="Size of the circular buffer")
    read_size: int = Field(default=100, description="Number of bytes to read per acquisition")

    _spi: spidev.SpiDev = PrivateAttr()
    _buffer: ndarray = PrivateAttr()
    _buffer_idx: int = PrivateAttr(default=0)
    _last_timestamp: int = PrivateAttr(default=0)
    _is_open: bool = PrivateAttr(default=False)
    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition: bool = PrivateAttr(default=False)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def _init(self):
        """Initialize the SPI device and start acquisition if in CONTINUOUS mode."""
        if self._is_open:
            return

        try:
            self._spi = spidev.SpiDev(self.spi_bus, self.spi_cs)
            self._spi.bits_per_word = self.bits_per_word
            self._spi.cshigh = self.cs_high
            self._spi.loop = self.loop
            self._spi.no_cs = self.no_cs
            self._spi.lsbfirst = self.lsb_first
            self._spi.max_speed_hz = self.max_speed_hz
            self._spi.mode = self.mode
            self._spi.threewire = self.three_wire
            self._spi.read0 = self.read0
            self._spi.mosi_idle_low = self.mosi_idle_low
        except Exception as e:
            msg = f"Failed to initialize SPI device {self.spi_bus}:{self.spi_cs}: {e}"
            raise RuntimeError(msg) from e

        self._buffer = empty(
            self.buffer_size, dtype=[("timestamp", "uint64"), ("data", "uint8", self.read_size)]
        )
        self._buffer_idx = 0
        self._last_timestamp = 0
        self._is_open = True
        self._stop_acquisition = False

        self._start_acquisition_thread()

    def _close(self):
        """Close the SPI device and stop acquisition thread."""
        self._stop_acquisition = True

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=1.0)

        if self._is_open:
            try:
                self._spi.close()
            except Exception as e:
                LOGGER.warning(f"Failed to close SPISource: {e}")

        self._is_open = False
        self._acquisition_thread = None

    def _start_acquisition_thread(self):
        """Start the background acquisition thread."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            return

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func, daemon=True, name=f"SPI-Acquisition-{self.name}"
        )
        self._acquisition_thread.start()

    def _acquisition_thread_func(self):
        """Background thread for continuous SPI data acquisition."""
        while not self._stop_acquisition and self._is_open:
            try:
                # Read bytes from SPI (returns list of ints)
                start = perf_counter_ns()
                data = self._spi.readbytes(self.read_size)  # Returns List[int]
                end = perf_counter_ns()

                # Single timestamp for the entire read operation
                timestamp = (start + end) // 2  # Midpoint timestamp

                with self._lock:
                    idx = self._buffer_idx % self.buffer_size
                    self._buffer[idx]["timestamp"] = timestamp
                    self._buffer[idx]["data"] = data  # Store the list of ints
                    self._buffer_idx += 1

            except Exception as e:
                LOGGER.error(f"SPI acquisition error: {e}")
                break

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        if not self._is_open:
            msg = "SPISource need to me opened first"
            raise RuntimeError(msg)
        return [st for st in stimuli if isinstance(st, SPIStimulus)]

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        """Execute SPI stimuli with precise timing."""
        if not self._is_open:
            msg = "SPISource needs to be opened first"
            raise RuntimeError(msg)

        if self.acquisition_type == AcquisitionType.FINITE:
            with self._lock:
                self._last_timestamp = self._buffer["timestamp"][
                    (self._buffer_idx % self.buffer_size) - 1
                ]

        for st in stimuli:
            if isinstance(st, SPIStimulus):
                if st.pre_delay > 0:
                    precise_sleep(st.pre_delay / 1000.0)

                # Write SPI data
                self._spi.writebytes2(st.build())

                if st._isi + st.post_delay > 0:
                    precise_sleep((st._isi + st.post_delay) / 1000.0)

                    # Read response if in finite mode (store in buffer)
                    data = self._spi.readbytes(self.read_size)
                    timestamp = perf_counter_ns()

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        """Publish acquired SPI data based on acquisition type."""
        if not self._is_open:
            raise RuntimeError("SPISource needs to be opened first")

        with self._lock:
            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                return self._publish_continuous(stimuli)
            elif self.acquisition_type == AcquisitionType.FINITE:
                return self._publish_finite(stimuli)

        return False

    def _publish_continuous(self, stimuli: Sequence[BaseStimulus]) -> bool:
        with self._lock:
            start = self._last_timestamp
            end = self._buffer["timestamp"][(self._buffer_idx % self.buffer_size) - 1]

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] < end)
        chunk = self._buffer[mask]
        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={"spi": chunk}, meta={"acquisition": "continuous"})
        )

        with self._lock:
            self._last_timestamp = end

        return True

    def _publish_finite(self, stimuli: Sequence[BaseStimulus]) -> bool:
        total_ms = 0
        for st in stimuli:
            if isinstance(st, SPIStimulus):
                total_ms += st.pre_delay + st.duration + st.post_delay + st._isi

        with self._lock:
            start = self._last_timestamp

        end = self._last_timestamp + total_ms * 1_000_000

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] <= end)
        chunk = self._buffer[mask]

        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={"spi": chunk}, meta={"acquisition": "finite"})
        )

        with self._lock:
            self._last_timestamp = end

        return True
