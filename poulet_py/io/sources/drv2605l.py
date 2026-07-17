try:
    from time import monotonic_ns
    from typing import Any

    from adafruit_bus_device.i2c_device import I2CDevice
    from board import SCL, SDA
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr
    from pyftdi.i2c import I2cNackError

    from poulet_py import LOGGER, BaseSource, DRV2605Stimulus, precise_sleep

except ImportError as e:
    msg = """
Missing DRV2605L source dependencies.

Install options:
- Dedicated:    pip install poulet_py[sources] adafruit-blinka adafruit-circuitpython-busdevice
- Module:       pip install poulet_py[io] adafruit-blinka adafruit-circuitpython-busdevice
- Full:         pip install poulet_py[all] adafruit-blinka adafruit-circuitpython-busdevice
"""
    raise ImportError(msg) from e


DRV2605_ADDR = 0x5A
REG_MODE = 0x01
REG_RTP_INPUT = 0x02
REG_LIBRARY = 0x03
REG_WAVESEQ1 = 0x04
REG_GO = 0x0C
REG_OD_CLAMP = 0x17
MODE_INTTRIG = 0x00
LIBRARY_TS2200A = 0x01


class DRV2605Source(BaseSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=DRV2605_ADDR)
    bus_frequency: int = Field(default=400_000)

    ftdi_latency_ms: int | None = Field(
        default=None,
        ge=1,
        le=255,
        description=("FTDI latency timer in ms"),
    )

    i2c: I2C | None = Field(
        default=None,
        description=(
            "Optional externally supplied I2C bus. If None, the source creates "
            "busio.I2C(board.SCL, board.SDA) in _open()."
        ),
    )
    i2c_retry_attempts: int = Field(
        default=5,
        ge=0,
        description="Number of retry attempts for transient DRV2605L I2C errors.",
    )

    i2c_retry_backoff_s: float = Field(
        default=0.005,
        ge=0.0,
        description="Backoff between DRV2605L I2C retries in seconds.",
    )

    continue_on_i2c_error: bool = Field(
        default=True,
        description="If True, log DRV2605L I2C failures and continue later trials.",
    )
    _device: I2CDevice | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
            ("time_roundtrip", "uint64"),
            ("waveform", "uint8"),
            ("repeat_count", "uint8"),
            ("drive_voltage", "float32"),
        ]

    def _open(self):
        if self.i2c is None:
            LOGGER.info("Opening internal I2C bus for DRV2605L")
            self.i2c = I2C(SCL, SDA, frequency=self.bus_frequency)
            self._internal_i2c = True
            self._set_ftdi_latency_timer()

        self._device = I2CDevice(self.i2c, self.address)
        self._initialize_drv2605()

    def _close(self):
        self.stop()

        self._device = None

        if self._internal_i2c and self.i2c is not None:
            LOGGER.info("Closing internal I2C bus for DRV2605L")
            self.i2c.deinit()
            self.i2c = None
            self._internal_i2c = False

    def _fire(self) -> bool:
        for stimulus in self._stimuli:
            t0 = monotonic_ns()
            if not isinstance(stimulus, DRV2605Stimulus):
                continue

            precise_sleep(stimulus.pre_delay / 1000.0)

            config = stimulus.build()
            waveform = int(config["waveform"])
            repeat_count = int(config["repeat_count"])
            drive_voltage = config["drive_voltage"]
            time_request = monotonic_ns()
            played = self._play_waveform_with_retries(
                waveform=waveform,
                repeat_count=repeat_count,
                drive_voltage=drive_voltage,
            )
            time_answer = monotonic_ns()
            time_roundtrip = time_answer - time_request
            time_read = (time_request + time_answer) // 2
            if played:
                self._write_sample(
                    (
                        time_read,
                        time_roundtrip,
                        waveform,
                        repeat_count,
                        float("nan") if drive_voltage is None else float(drive_voltage),
                    )
                )
            else:
                LOGGER.error(
                    "DRV2605L %s failed to play waveform=%s repeat_count=%s drive_voltage=%s",
                    self.name,
                    waveform,
                    repeat_count,
                    drive_voltage,
                )

                if not self.continue_on_i2c_error:
                    return False

            # Preserve trial timing even if the haptic command failed.
            precise_sleep(
                (stimulus.duration - ((monotonic_ns() - t0) / 1e9) + stimulus.post_delay) / 1000.0
            )

        return True

    def _initialize_drv2605(self):
        self._write_register(REG_MODE, MODE_INTTRIG)
        self._write_register(REG_RTP_INPUT, 0x00)
        self._write_register(REG_LIBRARY, LIBRARY_TS2200A)
        self._write_block(REG_WAVESEQ1, [0] * 8)
        self._write_register(REG_GO, 0)

    def _play_waveform(
        self,
        waveform: int,
        repeat_count: int,
        drive_voltage: float | None = None,
    ):
        repeat_count = max(0, min(int(repeat_count), 7))
        waveform = max(1, min(int(waveform), 123))

        if drive_voltage is not None:
            self._write_register(REG_OD_CLAMP, self._od_clamp_from_voltage(drive_voltage))

        slots = [waveform] * repeat_count + [0] * (8 - repeat_count)

        LOGGER.info(
            "DRV2605L I2C: %.0f Hz (configured=%s)",
            self.i2c._i2c._i2c.frequency,
            self.i2c._i2c._i2c.configured,
        )
        self._write_block(REG_WAVESEQ1, slots)
        self._write_register(REG_GO, 1)

    def stop(self):
        if self._device is None:
            return

        try:
            self._write_register(REG_GO, 0)

        except (I2cIOError, I2cNackError, OSError, TimeoutError) as e:
            LOGGER.warning(
                "DRV2605L %s failed to stop cleanly over I2C: %s",
                self.name,
                e,
            )

    def _write_register(self, register: int, value: int):
        if self._device is None:
            raise RuntimeError("DRV2605L I2C device is not open.")

        with self._device as device:
            device.write(bytes([register & 0xFF, value & 0xFF]))

    def _write_block(self, start_register: int, values: list[int]):
        if self._device is None:
            raise RuntimeError("DRV2605L I2C device is not open.")

        payload = bytes([start_register & 0xFF] + [value & 0xFF for value in values])

        with self._device as device:
            device.write(payload)

    def _play_waveform_with_retries(
        self,
        waveform: int,
        repeat_count: int,
        drive_voltage: float | None = None,
    ) -> bool:
        last_error: Exception | None = None

        for attempt in range(self.i2c_retry_attempts + 1):
            try:
                self._play_waveform(
                    waveform=waveform,
                    repeat_count=repeat_count,
                    drive_voltage=drive_voltage,
                )
                return True

            except (I2cNackError, OSError, TimeoutError) as e:
                last_error = e

                LOGGER.warning(
                    "DRV2605L %s transient I2C error during playback attempt %s/%s: %s",
                    self.name,
                    attempt + 1,
                    self.i2c_retry_attempts + 1,
                    e,
                )

                self._recover_i2c_backend()
                precise_sleep(self.i2c_retry_backoff_s)

        LOGGER.error(
            "DRV2605L %s failed after %s I2C attempts. Last error: %s",
            self.name,
            self.i2c_retry_attempts + 1,
            last_error,
        )

        return False

    def _recover_i2c_backend(self):
        try:
            pyftdi_i2c_controller = self.i2c._i2c._i2c
            pyftdi_i2c_controller.flush()
        except Exception:
            pass

    @staticmethod
    def _od_clamp_from_voltage(voltage: float) -> int:
        return max(0, min(255, round(float(voltage) * 255 / 5.6)))

    def _set_ftdi_latency_timer(self):
        if self.ftdi_latency_ms is None:
            return

        try:
            pyftdi_i2c_controller = self.i2c._i2c._i2c
            ftdi = pyftdi_i2c_controller.ftdi
            ftdi.set_latency_timer(self.ftdi_latency_ms)
            LOGGER.info(f"Set FTDI latency timer to {self.ftdi_latency_ms} ms")
        except AttributeError as e:
            raise RuntimeError("Could not access pyftdi controller through Blinka.") from e
