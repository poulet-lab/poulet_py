try:
    from collections.abc import Sequence
    from threading import Lock, Thread
    from time import time_ns

    from adafruit_max31856 import MAX31856, ThermocoupleType
    from busio import SPI
    from digitalio import DigitalInOut, Direction
    from numpy import empty, ndarray
    from pydantic import Field, PrivateAttr

    from poulet_py import (
        LOGGER,
        AcquisitionType,
        BaseSource,
        BaseStimulus,
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


class Max31856Source(BaseSource):
    name: str = Field(..., description="Name of the SPI source")
    cs_pin: int | None = Field(default=None, description="SPI chip select pin")
    thermocouple_type: ThermocoupleType = Field(
        default=ThermocoupleType.K, description="Type of thermocouple"
    )
    baudrate: int = Field(default=12_500_000, description="SPI clock speed in Hz")
    buffer_size: int = Field(default=1000, description="Size of the circular buffer")
    read_size: int = Field(default=4, description="Number of bytes to read per acquisition")

    _spi: SPI | None = PrivateAttr(None)
    _cs: DigitalInOut = PrivateAttr()
    _max31856: MAX31856 = PrivateAttr()
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
            self._spi = SPI()

            if self.cs_pin:
                self._cs = DigitalInOut(self.cs_pin)
                self._cs.direction = Direction.OUTPUT

            self._max31856 = MAX31856(
                self._spi,
                self._cs,
                thermocouple_type=self.thermocouple_type,
                baudrate=self.baudrate,
            )

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

        self._is_open = False
        self._acquisition_thread = None

    def _start_acquisition_thread(self):
        """Start the background acquisition thread."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            return

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func,
            daemon=True,
            name=f"MAX31856-Acquisition-{self.name}",
        )
        self._acquisition_thread.start()

    def _acquisition_thread_func(self):
        """Background thread for continuous SPI data acquisition."""
        while not self._stop_acquisition and self._is_open:
            try:
                timestamp = time_ns()

                with self._lock:
                    idx = self._buffer_idx % self.buffer_size
                    self._buffer[idx]["timestamp"] = timestamp
                    self._buffer[idx]["data"] = data  # Store the list of ints
                    self._buffer_idx += 1

            except Exception as e:
                LOGGER.error(f"SPI acquisition error: {e}")
                break

    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]:
        return []

    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool:

        return True

    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool:
        """Publish acquired SPI data based on acquisition type."""
        if not self._is_open:
            msg = "MAX31856Source needs to be opened first"
            raise RuntimeError(msg)

        with self._lock:
            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                return self._publish_continuous(stimuli)
            elif self.acquisition_type == AcquisitionType.FINITE:
                return self._publish_finite(stimuli)

        return False

    def _publish_continuous(self, stimuli: Sequence[BaseStimulus]) -> bool:

        return True

    def _publish_finite(self, stimuli: Sequence[BaseStimulus]) -> bool:

        return True
