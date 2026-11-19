try:
    from enum import IntEnum
    from threading import Event, Thread
    from time import monotonic_ns

    from adafruit_ina228 import INA228
    from adafruit_ina228 import AveragingCount as INA228AveragingCount
    from adafruit_ina228 import ConversionTime as IN228ConvwetionTime
    from adafruit_ina228 import Mode as INA228Mode
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


class Mode(IntEnum):
    SHUTDOWN = INA228Mode.SHUTDOWN
    TRIG_BUS = INA228Mode.TRIG_BUS
    TRIG_SHUNT = INA228Mode.TRIG_SHUNT
    TRIG_BUS_SHUNT = INA228Mode.TRIG_BUS_SHUNT
    TRIG_TEMP = INA228Mode.TRIG_TEMP
    TRIG_TEMP_BUS = INA228Mode.TRIG_TEMP_BUS
    TRIG_TEMP_SHUNT = INA228Mode.TRIG_TEMP_SHUNT
    TRIG_TEMP_BUS_SHUNT = INA228Mode.TRIG_TEMP_BUS_SHUNT
    CONT_BUS = INA228Mode.CONT_BUS
    CONT_SHUNT = INA228Mode.CONT_SHUNT
    CONT_BUS_SHUNT = INA228Mode.CONT_BUS_SHUNT
    CONT_TEMP = INA228Mode.CONT_TEMP
    CONT_TEMP_BUS = INA228Mode.CONT_TEMP_BUS
    CONT_TEMP_SHUNT = INA228Mode.CONT_TEMP_SHUNT
    CONT_TEMP_BUS_SHUNT = INA228Mode.CONT_TEMP_BUS_SHUNT

    # Convenience aliases
    TRIGGERED = TRIG_TEMP_BUS_SHUNT
    CONTINUOUS = CONT_TEMP_BUS_SHUNT


class ConversionTime(IntEnum):
    TIME_50_US = IN228ConvwetionTime.TIME_50_US
    TIME_84_US = IN228ConvwetionTime.TIME_84_US
    TIME_150_US = IN228ConvwetionTime.TIME_150_US
    TIME_280_US = IN228ConvwetionTime.TIME_280_US
    TIME_540_US = IN228ConvwetionTime.TIME_540_US
    TIME_1052_US = IN228ConvwetionTime.TIME_1052_US
    TIME_2074_US = IN228ConvwetionTime.TIME_2074_US
    TIME_4120_US = IN228ConvwetionTime.TIME_4120_US


class AveragingCount(IntEnum):
    COUNT_1 = INA228AveragingCount.COUNT_1
    COUNT_4 = INA228AveragingCount.COUNT_4
    COUNT_16 = INA228AveragingCount.COUNT_16
    COUNT_64 = INA228AveragingCount.COUNT_64
    COUNT_128 = INA228AveragingCount.COUNT_128
    COUNT_256 = INA228AveragingCount.COUNT_256
    COUNT_512 = INA228AveragingCount.COUNT_512
    COUNT_1024 = INA228AveragingCount.COUNT_1024


class INA228Source(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x40, description="INA228 I2C address")
    sample_rate: int = Field(
        default=10, ge=1, le=10000, description=("target sample rate in hz for voltage measurement")
    )
    skip_reset: bool = Field(
        default=False,
        description="Pass skip_reset to the Adafruit INA228 driver.",
    )
    i2c: I2C | None = Field(default=None)
    bus_frequency: int = Field(
        default=400_000,
        ge=1,
        le=1_000_000,
        description="I2C clock speed in Hz, has no effect in Linux",
    )
    ftdi_latency_ms: int | None = Field(
        default=1,
        description="FTDI USB latency timer in ms. Use None to leave default.",
        ge=0,
        le=255,
    )
    mode: Mode = Field(
        default=Mode.CONT_BUS,
        description="INA228 ADC mode. Default is continuous bus-voltage-only.",
    )
    bus_voltage_conv_time: ConversionTime = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 bus voltage conversion time setting.",
    )
    shunt_voltage_conv_time: ConversionTime = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 shunt voltage conversion time setting.",
    )
    temp_conv_time: ConversionTime = Field(
        default=ConversionTime.TIME_50_US,
        description="INA228 temperature conversion time setting.",
    )
    averaging_count: AveragingCount = Field(
        default=AveragingCount.COUNT_1,
        description="INA228 averaging count.",
    )

    _ina228: INA228 | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)

    _ina228_acquisition_thread: Thread | None = PrivateAttr(default=None)
    _ina228_stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [("timestamp", "uint64"), ("bus_voltage", "float32")]

    def _open(self):
        try:
            if self.i2c is None:
                self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
                self._internal_i2c = True

            self._set_ftdi_latency_timer()
            self._ina228 = INA228(self.i2c, address=self.address, skip_reset=self.skip_reset)

            self._configure_ina228()
            self._start_acquisition_thread()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize INA228 at address 0x{self.address:02X}"
            ) from e

    def _close(self):
        self._ina228_stop_acquisition_event.set()

        if self._ina228_acquisition_thread and self._ina228_acquisition_thread.is_alive():
            self._ina228_acquisition_thread.join(timeout=2.0)

        if self._ina228_acquisition_thread and self._ina228_acquisition_thread.is_alive():
            LOGGER.warning(
                "INA228 %s at 0x%02X: Acquisition thread is still alive after closure",
                self.name,
                self.address,
            )

        self._ina228_acquisition_thread = None
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
        if (
            self._ina228_acquisition_thread is not None
            and self._ina228_acquisition_thread.is_alive()
        ):
            return

        self._ina228_stop_acquisition_event.clear()

        self._ina228_acquisition_thread = Thread(
            target=self._ina228_acquisition_thread_func,
            daemon=True,
            name=f"INA228-Acquisition-{self.name}",
        )
        self._ina228_acquisition_thread.start()

    def _ina228_acquisition_thread_func(self):
        period = round(1e9 / self.sample_rate)
        deadline = monotonic_ns()
        errors = 0

        while not self._ina228_stop_acquisition_event.is_set():
            deadline += period

            try:
                request = monotonic_ns()
                voltage = self._ina228.bus_voltage
                answer = monotonic_ns()
                errors = 0

                self._write_sample(((request + answer) // 2, voltage))

                remaining = deadline - monotonic_ns()
                if remaining > 0:
                    precise_sleep(remaining * 1e-9)

            except Exception as error:
                errors += 1
                LOGGER.warning(
                    "INA228 %s acquisition error %d/10 at 0x%02X: %s",
                    self.name,
                    errors,
                    self.address,
                    error,
                )

                if errors >= 10:
                    LOGGER.exception(
                        "INA228 %s stopped after repeated errors at 0x%02X",
                        self.name,
                        self.address,
                    )
                    self._ina228_stop_acquisition_event.set()

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
            raise RuntimeError("Could not access pyftdi controller through Blinka. ") from e
