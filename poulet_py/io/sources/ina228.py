try:
    from threading import Event, Thread
    from time import monotonic_ns, sleep

    from adafruit_ina228 import INA228, AveragingCount, ConversionTime, Mode
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
    bus_frequency: int = Field(default=400_000, gt=1)

    ftdi_latency_ms: int | None = Field(
        default=1,
        ge=1,
        le=255,
        description="FTDI USB latency timer in ms. Use None to leave unchanged.",
    )

    sample_interval_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Delay after each INA228 read. Use 0.0 for maximum throughput. "
            "Use ~0.001 for approximately 1 kHz target pacing."
        ),
    )

    mode: int = Field(
        default=Mode.CONT_BUS,
        description="INA228 ADC mode. Default is continuous bus-voltage-only.",
    )

    bus_voltage_conv_time: int = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 bus voltage conversion time setting.",
    )

    shunt_voltage_conv_time: int = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 shunt voltage conversion time setting.",
    )

    temp_conv_time: int = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 temperature conversion time setting.",
    )

    averaging_count: int = Field(
        default=AveragingCount.COUNT_1,
        description="INA228 averaging count.",
    )

    skip_reset: bool = Field(
        default=False,
        description="Pass skip_reset to the Adafruit INA228 driver.",
    )

    i2c: I2C | None = Field(default=None)

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
        if self._is_open:
            return

        # BaseSource.open() creates _source_buffer only after _open().
        # Therefore the acquisition thread must start after super().open().
        super().open()
        self._start_acquisition_thread()

    def close(self) -> None:
        if not self._is_open:
            return

        # Stop writing before BaseSource deletes _source_buffer.
        self._stop_acquisition_thread()
        super().close()

    def _open(self):
        try:
            if self.i2c is None:
                self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
                self._internal_i2c = True

            self._set_ftdi_latency_timer()

            self._ina228 = INA228(
                self.i2c,
                address=self.address,
                skip_reset=self.skip_reset,
            )

            self._configure_ina228()

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize INA228 at address 0x{self.address:02X}"
            ) from e

    def _close(self):
        self._ina228 = None

        if self._internal_i2c and self.i2c is not None:
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True

    def _configure_ina228(self):
        if self._ina228 is None:
            raise RuntimeError("INA228 is not initialized.")

        self._ina228.averaging_count = self.averaging_count
        self._ina228.bus_voltage_conv_time = self.bus_voltage_conv_time
        self._ina228.shunt_voltage_conv_time = self.shunt_voltage_conv_time
        self._ina228.temp_conv_time = self.temp_conv_time
        self._ina228.mode = self.mode

        LOGGER.info(
            "INA228 %s at 0x%02X configured: mode=%s, avg=%s, vbus_ct=%s, vshunt_ct=%s, temp_ct=%s",
            self.name,
            self.address,
            self.mode,
            self.averaging_count,
            self.bus_voltage_conv_time,
            self.shunt_voltage_conv_time,
            self.temp_conv_time,
        )

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
                    "INA228 %s: acquisition thread still alive after closure",
                    self.name,
                )

        self._acquisition_thread = None

    def _acquisition_thread_func(self):
        n = 0
        t0 = monotonic_ns()
        consecutive_errors = 0
        while not self._stop_acquisition_event.is_set():
            try:
                if self._ina228 is None:
                    LOGGER.warning("INA228 %s stopped: device not initialized", self.name)
                    break

                self._write_sample(
                    (
                        monotonic_ns(),
                        float(self._ina228.bus_voltage),
                    )
                )

                n += 1

                if n % 1000 == 0:
                    now = monotonic_ns()
                    hz = 1000 / ((now - t0) * 1e-9)
                    LOGGER.info("INA228 %s acquisition rate: %.1f Hz", self.name, hz)
                    LOGGER.info(
                        "INA228 I2C: %.0f Hz (configured=%s)",
                        self.i2c._i2c._i2c.frequency,
                        self.i2c._i2c._i2c.configured,
                    )
                    t0 = now
                    if 1.7 >= self._ina228.bus_voltage >= 1.6:
                        LOGGER.warning(
                            "INA228 %s caution approaching unsafe temperature", self.name
                        )
                    elif self._ina228.bus_voltage >= 1.7:
                        LOGGER.error(
                            "INA228 %s temperature too high, terminate experiment!", self.name
                        )

                if self.sample_interval_s > 0:
                    precise_sleep(self.sample_interval_s)
                else:
                    # Yield very lightly; the I2C call already dominates timing,
                    # but this helps avoid starving other Python threads.
                    sleep(0)

            except Exception as e:
                consecutive_errors += 1
                LOGGER.warning(
                    "INA228 %s transient acquisition error %s/100 at address 0x%02X: %s",
                    self.name,
                    consecutive_errors,
                    self.address,
                    e,
                )

                # precise_sleep(0.0001)

                if consecutive_errors >= 100:
                    LOGGER.exception(
                        "INA228 %s stopping after repeated acquisition errors at address 0x%02X",
                        self.name,
                        self.address,
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
                "INA228 %s: set FTDI latency timer to %s ms",
                self.name,
                self.ftdi_latency_ms,
            )

        except AttributeError as e:
            raise RuntimeError("Could not access pyftdi controller through Blinka.") from e
