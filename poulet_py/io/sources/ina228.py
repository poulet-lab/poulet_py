try:
    from threading import Event, Thread
    from time import monotonic_ns
    from typing import Literal
    from adafruit_ina228 import INA228
    from board import SCL, SDA
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource, precise_sleep

except ImportError as e:
    raise ImportError(
        """
Missing 'sources' module.

Install options:
- Dedicated: pip install poulet_py[sources]
- Module:    pip install poulet_py[io]
- Full:      pip install poulet_py[all]
"""
    ) from e


class INA228Source(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x40, description="INA228 I2C address")

    bus_frequency: int = Field(
        default=400_000,
        description="I2C clock speed in Hz. With FT232H/Blinka this may be backend-limited.",
        gt=1,
    )

    ftdi_latency_ms: int | None = Field(
        default=1,
        description="FTDI USB latency timer in ms. Use None to leave default.",
        ge=1,
        le=255,
    )

    sample_interval_s: float = Field(
        default=0.001,
        gt=0,
        description=(
            "Delay between INA228 reads in seconds. "
            "Use 0.005-0.02 s for initial FT232H testing."
        ),
    )
    bus_voltage_conv_time: Literal[50, 80, 150, 280, 540, 1052, 2074, 4120] = Field(

        default=50, description="ADC conversion time for bus voltage measurement in microseconds"

    )

    shunt_voltage_conv_time: Literal[50, 80, 150, 280, 540, 1052, 2074, 4120] = Field(

        default=50, description="ADC conversion time for shunt voltage measurement in microseconds, higher conversion times can improve the accurcay of a signal but also increase the time it takes to acquire a signal"

    )

    averaging_count: Literal[1, 4, 16, 64, 128, 256, 512, 1024] = Field(

        default=1, description="Number of samples to average for each reading, higher values can improve the accurcay of a signal but also increase the time it takes to acquire a signal"

    )
    i2c: I2C | None = Field(
        default=None,
        description=(
            "Optional externally supplied I2C bus. "
            "For multiple INA228 devices on one FT232H, pass the same i2c object to all sources."
        ),
    )

    _ina228: INA228 | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)
    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
            ("bus_voltage", "float32"),
        ]

    def open(self) -> None:
        """
        Open BaseSource first so _source_buffer exists before the acquisition
        thread starts writing samples.
        """
        if self._is_open:
            return

        super().open()
        self._start_acquisition_thread()

    def close(self) -> None:
        """
        Stop the acquisition thread before BaseSource deletes _source_buffer.
        """
        self._stop_acquisition_thread()
        super().close()

    def _open(self):
        try:
            if self.i2c is None:
                self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
                self._internal_i2c = True
                self._set_ftdi_latency_timer()

            self._ina228 = INA228(self.i2c, address=self.address)

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize INA228 at address 0x{self.address:02X}"
            ) from e

        # IMPORTANT:
        # Do not start the acquisition thread here.
        # BaseSource has not created _source_buffer yet.

    def _close(self):
        self._ina228 = None

        if self._internal_i2c and self.i2c is not None:
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True

    def _start_acquisition_thread(self):
        if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
            return

        self._stop_acquisition_event.clear()

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func,
            daemon=True,
            name=f"INA228-Acquisition-{self.name}",
        )
        self._acquisition_thread.start()

    def _stop_acquisition_thread(self):
        self._stop_acquisition_event.set()

        if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)

            if self._acquisition_thread.is_alive():
                LOGGER.warning(
                    f"INA228 {self.name}: acquisition thread is still alive after closure"
                )

        self._acquisition_thread = None

    def _acquisition_thread_func(self):
        while not self._stop_acquisition_event.is_set():
            try:
                if self._ina228 is None:
                    LOGGER.warning(
                        f"INA228 {self.name}: acquisition stopped because device is not initialized"
                    )
                    break

                self._write_sample(
                    (
                        monotonic_ns(),
                        float(self._ina228.bus_voltage),
                    )
                )

                precise_sleep(self.sample_interval_s)

            except Exception as e:
                LOGGER.error(
                    f"INA228 {self.name} acquisition error at address "
                    f"0x{self.address:02X}: {e}"
                )
                break

    def _set_ftdi_latency_timer(self):
        if self.ftdi_latency_ms is None:
            return

        try:
            pyftdi_i2c_controller = self.i2c._i2c._i2c
            ftdi = pyftdi_i2c_controller.ftdi
            ftdi.set_latency_timer(self.ftdi_latency_ms)
            LOGGER.info(
                f"INA228 {self.name}: set FTDI latency timer to {self.ftdi_latency_ms} ms"
            )

        except AttributeError as e:
            raise RuntimeError("Could not access pyftdi controller through Blinka.") from e
