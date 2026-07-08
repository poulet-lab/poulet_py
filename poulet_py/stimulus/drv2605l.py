try:
    from threading import Event, Thread
    from time import monotonic_ns
    from typing import Literal
    from pyftdi.ftdi import Sequence
    import board
    import busio
    from adafruit_drv2605 import DRV2605
    from busio import I2C
    from pydantic import ConfigDict, Field, PrivateAttr
    from adafruit_bus_device.i2c_device import I2CDevice
    from poulet_py import LOGGER, BaseStimulus, precise_sleep

except ImportError as e:
    msg = """
Missing DRV2605 source dependencies. Install options:
- Dedicated:    pip install poulet_py[sources] adafruit-circuitpython-drv2605 adafruit-blinka
- Module:       pip install poulet_py[io] adafruit-circuitpython-drv2605 adafruit-blinka
- Full:         pip install poulet_py[all] adafruit-circuitpython-drv2605 adafruit-blinka
"""
    raise ImportError(msg) from e

#Minimal vaible to test the DRV2605 driver and its integration with Poulet-Py. 
#This is a simple example that initializes the DRV2605, sets up a sequence of effects, and plays them. 
#For now a user setable sequence length of 1000ms "Alert" at 100% intensity is used. 

#register to set the waveform sequence, 8 slots of 1 byte each, 0-123 for effect number, 0 for empty slot
REG_WAVESEQ1 = 0x04
#register to start the effect sequence
REG_GO = 0x0C
#drive voltage clamp register, 0-255 corresponds to 0-5.6V
REG_OD_CLAMP = 0x17


class DRV2605Stimulus(BaseStimulus):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: int = Field(default=0x5A, description="DRV2605 I2C address")
    bus_frequency: int = Field(
        default=400_000, 
        description="Fast I2C clock speed in Hz")

    repeats: int = Field(
        default=5,
        description="Number of repeats for the effect sequence (1-8)",
        ge=1,
        le=8,
    )
    waveform: int = Field(
        default=16,
        description="100% intensity 1000ms 'Alert' waveform effect (#16)",
        ge=1,
        le=123,
    )
    random: int = Field(
        default=False,
        description="Whether to randomize the waveform length (True/False)",
    )
    custom: int = Field(
        default=False,
        description="Whether to use a custom waveform sequence (True/False)",
    )
    voltage: float = Field(
        default=4.5,
        description="Drive voltage in volts (0-5.6)",
        ge=0,
        le=5.6,
    )


    i2c: I2C | None = Field(default=None)

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
    except Exception:
        i2c = None

    # Optional: keep the official driver for init/config/play compatibility
    # drv = DRV2605(i2c)

    # Raw register access over the same working Adafruit/Blinka I2C bus
    raw_drv = I2CDevice(i2c, address) if i2c is not None else None

    def write_reg(self, register, value):
        if self.raw_drv is None:
            raise RuntimeError("I2C device not initialized")
        with self.raw_drv as dev:
            dev.write(bytes([register, value & 0xFF]))

    def write_block(self, start_register, values):
        if self.raw_drv is None:
            raise RuntimeError("I2C device not initialized")
        with self.raw_drv as dev:
            dev.write(bytes([start_register] + [v & 0xFF for v in values]))

    def od_clamp_from_voltage(self, voltage):
        return max(0, min(255, round(voltage * 255 / 5.6)))

    def set_effect16_repeats(self, repeats, custom=False):
        repeats = max(0, min(int(repeats), 7))
        if custom:
            slots = [16] * repeats + [0] * (8 - repeats)
        else:
            slots = [16] * repeats + [0] * (8 - repeats)
        self.write_block(REG_WAVESEQ1, slots)

    def set_drive_voltage(self, voltage):
        self.write_reg(REG_OD_CLAMP, self.od_clamp_from_voltage(voltage))

    def stimulate(self):
        self.write_reg(REG_GO, 1)

    def build(self, *args, **kwargs) -> Sequence[bytes]:
        self.set_drive_voltage(self.voltage)
        self.set_effect16_repeats(self.repeats, self.custom)
        self.stimulate()
