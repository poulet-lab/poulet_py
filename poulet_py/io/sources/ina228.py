from ast import If


try:
    from threading import Event, Thread
    from time import monotonic_ns, sleep

    from adafruit_ina228 import INA228, AveragingCount, ConversionTime, Mode
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
    maximum_valid_voltage: float = Field(
        default=5.6,
        ge=0.0,
        le=85.0,
        description=("Maximum voltage that doesnt get rejected, check calibrated voltage"),
    )

    sample_rate_Hz: int = Field(
        default=10,
        ge=1,
        le=10000,
        description=("target sample rate for voltage measurement")
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
    calibration_offset_voltage: float = Field(
        default=0.0,
        description="Voltage added for safety checks and temperature conversion.",
    )
    temperature: bool = Field(
        default=True,
        description="Convert the corrected voltage to temperature for safety checks.",
    )
    temperature_warning: float = Field(
        default=41.0,
        ge=25.0,
        le=60.0,
        description="Temperature warning threshold in °C.",
    )
    temperature_maximum: float = Field(
        default=42.0,
        ge=25.0,
        le=60.0,
        description="Maximum temperature threshold in °C (termination criterion).",
    )

    i2c: I2C | None = Field(default=None)

    _ina228: INA228 | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)

    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"), #midpoint between read i2c transaction and value read timepoint
            ("time_roundtrip", "uint64"), #time passed between read i2c request and read timepoint
            ("bus_voltage", "float32"), #bus voltage in V
            ("invalid_value_count", "int16"), #number of invalid values from sensor for debugging
        ]

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
        self._start_acquisition_thread()

    def _close(self):
        self._stop_acquisition_event.set()
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            LOGGER.warning(
                "INA228 %s at 0x%02X: Acquisition thread is still alive after closure",
                self.name,
                self.address,
            )
        self._acquisition_thread = None
        self._ina228 = None
        if self._internal_i2c and self.i2c is not None:
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False


    #calculate time until call of def fire for higher accuracy??
    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True

    def _configure_ina228(self): #initializes INA228 with user settings
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

    def _acquisition_thread_func(self):
        ina228 = self._ina228
        if ina228 is None:
            return

        clock, wait, write, sample_rate = monotonic_ns, precise_sleep, self._write_sample, self.sample_rate_Hz
        stop = self._stop_acquisition_event

        period = round(1_000_000_000 / sample_rate)
        check_every = max(1, round(2 * sample_rate))
        deadline = clock()

        maximum = self.maximum_valid_voltage
        temperature = self.temperature
        offset = self.calibration_offset_voltage

        invalid = errors = samples = 0

        while not stop.is_set():
            deadline += period

            try:
                request = clock()
                voltage = ina228.bus_voltage
                answer = clock()
                errors = 0

                voltage += offset if temperature else 0.0

                if voltage <= maximum:
                    write(
                        (
                            (request + answer) // 2,
                            answer - request,
                            voltage,
                            invalid,
                        )
                    )

                    samples += 1

                    if temperature and samples >= check_every:
                        samples = 0
                        #Conversion: voltage to temp [°C], based on FHC DC Controller datasheet
                        #Offset needs to be calibrated for each FHC DC Controller
                        temperature_c = 25.0 + voltage * 10.0
                        if temperature_c >= self.temperature_maximum: #this is dependant on the animal license
                            LOGGER.error(
                                "INA228 %s temperature too high: %.2f °C",
                                self.name,
                                temperature_c,
                            )
                        elif temperature_c >= self.temperature_warning:
                            LOGGER.warning(
                                "INA228 %s approaching unsafe temperature: %.2f °C",
                                self.name,
                                temperature_c,
                            )
                else:
                    invalid += 1

                remaining = deadline - clock()
                if remaining > 0:
                    wait(remaining * 1e-9)

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
                    return

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
