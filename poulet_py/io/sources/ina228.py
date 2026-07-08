try:
    from threading import Event, Thread
    from time import monotonic_ns

    from adafruit_ina228 import INA228
    from board import SCL, SDA
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource, precise_sleep
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class INA228Source(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x40, description="INA228 I2C address")
    bus_frequency: int = Field(
        default=400_000, description="I2C clock speed in Hz, has no effect in Linux", gt=1
    )
    ftdi_latency_ms: int | None = Field(
        default=1,
        description="FTDI USB latency timer in ms. Use None to leave default.",
        gt=1,
        lt=255,
    )

    i2c: I2C | None = Field(default=None)

    _ina228: INA228 = PrivateAttr()
    _internal_i2c: bool = PrivateAttr(default=False)

    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._buffer_dtype = [("timestamp", "uint64"), ("bus_voltage", "float32")]

    def _open(self):
        try:
            if self.i2c is None:
                self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
                self._internal_i2c = True

            self._set_ftdi_latency_timer()
            self._ina228 = INA228(self.i2c, address=self.address)
        except Exception as e:
            raise RuntimeError("Failed to initialize INA228") from e

        self._start_acquisition_thread()

    def _close(self):
        self._stop_acquisition_event.set()

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)
            if self._acquisition_thread.is_alive():
                LOGGER.warning("INA228: Acquisition thread is still alive after closure")

        del self._acquisition_thread
        del self._ina228

        if self._internal_i2c and self.i2c is not None:
            self.i2c.deinit()
        self.i2c = None

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True

    def _start_acquisition_thread(self):
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            return

        self._stop_acquisition_event.clear()

        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func,
            daemon=True,
            name=f"INA228-Acquisition-{self.name}",
        )
        self._acquisition_thread.start()

    def _acquisition_thread_func(self):
        while not self._stop_acquisition_event.is_set():
            try:
                bus_voltage = self._ina228.bus_voltage
                timestamp = monotonic_ns()
                self._write_sample((timestamp, bus_voltage))
                precise_sleep(0.0005)
            except Exception as e:
                LOGGER.error(f"INA228 acquisition error: {e}")
                break

    def _set_ftdi_latency_timer(self):
        if self.ftdi_latency_ms is None:
            return

        try:
            pyftdi_i2c_controller = self.i2c._i2c._i2c
            ftdi = pyftdi_i2c_controller.ftdi
            ftdi.set_latency_timer(self.ftdi_latency_ms)
        except AttributeError as e:
            raise RuntimeError("Could not access pyftdi controller through Blinka. ") from e
