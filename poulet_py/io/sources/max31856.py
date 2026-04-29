try:
    from enum import Enum
    from threading import Event, Thread
    from time import time_ns
    from typing import Literal

    from adafruit_blinka.microcontroller.generic_linux.rpi_gpio_pin import Pin
    from adafruit_max31856 import (
        _MAX31856_FAULT_CJHIGH,
        _MAX31856_FAULT_CJLOW,
        _MAX31856_FAULT_CJRANGE,
        _MAX31856_FAULT_OPEN,
        _MAX31856_FAULT_OVUV,
        _MAX31856_FAULT_TCHIGH,
        _MAX31856_FAULT_TCLOW,
        _MAX31856_FAULT_TCRANGE,
        _MAX31856_SR_REG,
        MAX31856,
    )
    from adafruit_max31856 import ThermocoupleType as ThType
    from board import MISO, MOSI, SCLK
    from busio import SPI
    from digitalio import DigitalInOut, Direction
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class ThermocoupleType(int, Enum):
    B = ThType.B
    E = ThType.E
    J = ThType.J
    K = ThType.K
    N = ThType.N
    R = ThType.R
    S = ThType.S
    T = ThType.T
    G8 = ThType.G8
    G32 = ThType.G32


class Max31856Source(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thermocouple_type: ThermocoupleType = Field(
        default=ThermocoupleType.K, description="Type of thermocouple"
    )
    baud_rate: int = Field(default=500_000, description="SPI clock speed in Hz")
    cs_pin: Pin | None = Field(default=None, description="SPI chip select pin")
    sclk: Pin | None = Field(default=SCLK, description="SPI SCLK pin")
    miso: Pin | None = Field(default=MISO, description="SPI MISO pin")
    mosi: Pin | None = Field(default=MOSI, description="SPI MOSI pin")
    averaging: Literal[1, 2, 4, 8, 16] = Field(
        default=1,
        description="Number of samples averaged together in each result. No averaging by default",
    )
    temperature_thresholds: tuple[float, float] = Field(
        default=(-2.0, 40.0), description="Thermocouple low/high fault threshold."
    )
    reference_temperature_thresholds: tuple[float, float] = Field(
        default=(-2.0, 40.0), description="Cold junction low/high fault threshold."
    )

    _spi: SPI | None = PrivateAttr(None)
    _cs: DigitalInOut = PrivateAttr()
    _max31856: MAX31856 = PrivateAttr()

    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._buffer_dtype = [
            ("timestamp", "uint64"),
            ("temperature", "float32"),
            ("reference", "float32"),
            ("faults", "uint8"),
        ]

    def _open(self):
        try:
            self._spi = SPI(self.sclk, self.mosi, self.miso)

            if self.cs_pin:
                self._cs = DigitalInOut(self.cs_pin)
                self._cs.direction = Direction.OUTPUT

            self._max31856 = MAX31856(
                self._spi,
                self._cs,
                thermocouple_type=self.thermocouple_type,
                baudrate=self.baud_rate,
            )
            self._max31856.temperature_thresholds = self.temperature_thresholds
            self._max31856.reference_temperature_thresholds = self.reference_temperature_thresholds

        except Exception as e:
            msg = f"Failed to initialize MAX31856: {e}"
            raise RuntimeError(msg) from e

        self._stop_acquisition = False
        self._start_acquisition_thread()

    def _close(self):
        """Close the SPI device and stop acquisition thread."""
        self._stop_acquisition = True

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=1.0)

        if self._spi:
            self._spi.deinit()
        if hasattr(self, "_cs"):
            self._cs.deinit()

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
                faults = self._max31856._read_register(_MAX31856_SR_REG, 1)[0]
                self._max31856._perform_one_shot_measurement()
                timestamp = time_ns()
                temperature = self._max31856.read_high_res_temp()
                reference = self._max31856.unpack_reference_temperature()

                if faults:
                    msg = "Faults found in the following: "
                    if faults & _MAX31856_FAULT_CJRANGE:
                        msg += "cj_range"
                    if faults & _MAX31856_FAULT_TCRANGE:
                        msg += "tc_range"
                    if faults & _MAX31856_FAULT_CJHIGH:
                        msg += "cj_high"
                    if faults & _MAX31856_FAULT_CJLOW:
                        msg += "cj_low"
                    if faults & _MAX31856_FAULT_TCHIGH:
                        msg += "tc_high"
                    if faults & _MAX31856_FAULT_TCLOW:
                        msg += "tc_low"
                    if faults & _MAX31856_FAULT_OVUV:
                        msg += "voltage"
                    if faults & _MAX31856_FAULT_OPEN:
                        msg += "open_tc"
                    LOGGER.warning(msg)

                with self._lock:
                    idx = self._buffer_idx % self.buffer_size
                    print(
                        f"Writing MAX31856 data to buffer at index {idx}, timestamp {timestamp}, temperature {temperature:.2f}°C, reference {reference:.2f}°C, faults {faults:08b}"
                    )
                    self._buffer[idx]["timestamp"] = timestamp
                    self._buffer[idx]["temperature"] = temperature
                    self._buffer[idx]["reference"] = reference
                    self._buffer[idx]["faults"] = faults
                    self._buffer_idx += 1

            except Exception as e:
                LOGGER.error(f"MAX31856 acquisition error: {e}")
                break

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        return True
