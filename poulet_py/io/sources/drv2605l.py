try:
    from time import monotonic_ns
    from typing import Any

    from adafruit_bus_device.i2c_device import I2CDevice
    from board import SCL, SDA
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr

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
    """
    Runtime source/controller for the DRV2605L haptic motor driver.

    This follows the same basic hardware-lifecycle pattern as INA228Source:
    - optional externally supplied I2C bus
    - internal bus creation in _open()
    - optional FTDI latency setting for FT232H/Blinka
    - cleanup in _close()
    - stimulation in _fire()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=DRV2605_ADDR)
    bus_frequency: int = Field(default=400_000, gt=1)

    ftdi_latency_ms: int | None = Field(
        default=1,
        ge=1,
        le=255,
        description=(
            "Optional FTDI latency timer in ms. Only applies when using "
            "Blinka through an FT232H backend."
        ),
    )

    i2c: I2C | None = Field(
        default=None,
        description=(
            "Optional externally supplied I2C bus. If None, the source creates "
            "busio.I2C(board.SCL, board.SDA) in _open()."
        ),
    )

    fire_on: tuple[type[DRV2605Stimulus], ...] = Field(
        default=(DRV2605Stimulus,),
        description="Only fire for DRV2605Stimulus objects.",
    )

    _device: I2CDevice | None = PrivateAttr(default=None)
    _internal_i2c: bool = PrivateAttr(default=False)

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
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
            if not isinstance(stimulus, DRV2605Stimulus):
                continue

            precise_sleep(stimulus.pre_delay / 1000.0)

            config = stimulus.build()
            waveform = int(config["waveform"])
            repeat_count = int(config["repeat_count"])
            drive_voltage = config["drive_voltage"]

            self._play_waveform(
                waveform=waveform,
                repeat_count=repeat_count,
                drive_voltage=drive_voltage,
            )

            self._write_sample(
                (
                    monotonic_ns(),
                    waveform,
                    repeat_count,
                    float("nan") if drive_voltage is None else float(drive_voltage),
                )
            )

            precise_sleep((stimulus.duration + stimulus.post_delay) / 1000.0)

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

        self._write_block(REG_WAVESEQ1, slots)
        self._write_register(REG_GO, 1)

    def stop(self):
        if self._device is not None:
            self._write_register(REG_GO, 0)

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
