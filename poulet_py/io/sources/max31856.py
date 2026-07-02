try:
    from enum import Enum
    from threading import Event, Thread
    from time import monotonic_ns
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
    _cs: DigitalInOut | None = PrivateAttr(default=None)
    _max31856: MAX31856 = PrivateAttr()

    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
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

        self._start_acquisition_thread()

    def _start_acquisition_thread(self):
        """Start the background acquisition thread."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            return

        self._stop_acquisition_event.clear()

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func,
            daemon=True,
            name=f"MAX31856-Acquisition-{self.name}",
        )
        self._acquisition_thread.start()

    def _close(self):
        """Close the SPI device and stop acquisition thread."""
        self._stop_acquisition_event.set()

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)
            if self._acquisition_thread.is_alive():
                LOGGER.warning("MAX31856: Acquisition thread is still alive after closure")

        if self._spi:
            self._spi.deinit()

        if self._cs:
            self._cs.deinit()

        self._acquisition_thread = None

    def _acquisition_thread_func(self):
        """Background thread for continuous SPI data acquisition."""
        while not self._stop_acquisition_event.is_set():
            try:
                faults = self._max31856._read_register(_MAX31856_SR_REG, 1)[0]
                self._max31856._perform_one_shot_measurement()
                timestamp = monotonic_ns()
                temperature = self._max31856.read_high_res_temp()
                reference = self._max31856.unpack_reference_temperature()

                self._log_faults(faults)
                self._write_sample((timestamp, temperature, reference, faults))

                precise_sleep(0.1)

            except Exception as e:
                LOGGER.error(f"MAX31856 acquisition error: {e}")
                break

    def _trigger(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)

        return True

    @staticmethod
    def _log_faults(faults: int):
        if faults:
            fault_msgs = []
            if faults & _MAX31856_FAULT_CJRANGE:
                fault_msgs.append("cj_range")
            if faults & _MAX31856_FAULT_TCRANGE:
                fault_msgs.append("tc_range")
            if faults & _MAX31856_FAULT_CJHIGH:
                fault_msgs.append("cj_high")
            if faults & _MAX31856_FAULT_CJLOW:
                fault_msgs.append("cj_low")
            if faults & _MAX31856_FAULT_TCHIGH:
                fault_msgs.append("tc_high")
            if faults & _MAX31856_FAULT_TCLOW:
                fault_msgs.append("tc_low")
            if faults & _MAX31856_FAULT_OVUV:
                fault_msgs.append("voltage")
            if faults & _MAX31856_FAULT_OPEN:
                fault_msgs.append("open_tc")

            LOGGER.warning(
                "MAX31856 faults: %s",
                ", ".join(fault_msgs),
            )
