try:
    from threading import Event, Thread
    from time import time_ns

    import board
    from adafruit_ina228 import INA228
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource, precise_sleep

except ImportError as e:
    msg = """
Missing INA228 source dependencies. Install options:
- Dedicated:    pip install poulet_py[sources] adafruit-circuitpython-ina228 adafruit-blinka
- Module:       pip install poulet_py[io] adafruit-circuitpython-ina228 adafruit-blinka
- Full:         pip install poulet_py[all] adafruit-circuitpython-ina228 adafruit-blinka
"""
    raise ImportError(msg) from e


class INA228Source_minimal(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x40, description="INA228 I2C address")
    sample_interval_s: float = Field(default=1, description="Sample interval in seconds")

    _i2c: I2C | None = PrivateAttr(default=None)
    _ina228: INA228 | None = PrivateAttr(default=None)

    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._buffer_dtype = [
            ("timestamp", "uint64"),
            ("current", "float32"),        # A
            ("bus_voltage", "float32"),    # V
            ("shunt_voltage", "float32"),  # V
            ("power", "float32"),          # W
            ("energy", "float32"),         # J
            ("temperature", "float32"),    # °C
        ]

    def _open(self):
        try:
            self._i2c = board.I2C()
            self._ina228 = INA228(self._i2c, address=self.address)

        except Exception as e:
            msg = f"Failed to initialize INA228: {e}"
            raise RuntimeError(msg) from e

        self._start_acquisition_thread()

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

    def _close(self):
        self._stop_acquisition_event.set()

        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)
            if self._acquisition_thread.is_alive():
                LOGGER.warning("INA228: Acquisition thread is still alive after closure")

        if self._i2c:
            self._i2c.deinit()

        self._ina228 = None
        self._i2c = None
        self._acquisition_thread = None

    def _acquisition_thread_func(self):
        while not self._stop_acquisition_event.is_set():
            try:
                if self._ina228 is None:
                    raise RuntimeError("INA228 is not initialized")

                timestamp = time_ns()
                #current = self._ina228.current
                bus_voltage = self._ina228.bus_voltage
                #shunt_voltage = self._ina228.shunt_voltage
                #power = self._ina228.power
                #energy = self._ina228.energy
                #temperature = self._ina228.die_temperature

                self._write_sample(
                    (
                        timestamp,
                        #current,
                        bus_voltage,
                        #shunt_voltage,
                        #power,
                        #energy,
                        #temperature,
                    )
                )

                precise_sleep(self.sample_interval_s)

            except Exception as e:
                LOGGER.error(f"INA228 acquisition error: {e}")
                break

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True