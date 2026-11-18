from enum import IntEnum, StrEnum


class Reg:
    # ------------------------------------------------------------------
    # GigE Vision bootstrap (standard)
    # ------------------------------------------------------------------
    GEV_VERSION = 0x0000
    GEV_DEVICE_MODE = 0x0004  # bit 0 (LSB) = big-endian device
    GEV_MAC_HIGH = 0x0008
    GEV_MAC_LOW = 0x000C
    GEV_IP_CONFIG_CAPS = 0x0010
    GEV_CURRENT_IP_CONFIG = 0x0014
    GEV_CURRENT_IP = 0x0024
    GEV_CURRENT_SUBNET = 0x0034
    GEV_CURRENT_GATEWAY = 0x0044
    GEV_HEARTBEAT_TIMEOUT = 0x0938  # ms, 500..65000
    GEV_TIMESTAMP_FREQ_HIGH = 0x093C
    GEV_TIMESTAMP_FREQ_LOW = 0x0940
    GEV_TIMESTAMP_CONTROL = 0x0944  # write 1 = reset, 2 = latch
    GEV_TIMESTAMP_VALUE_HIGH = 0x0948
    GEV_TIMESTAMP_VALUE_LOW = 0x094C
    GEV_CCP = 0x0A00
    GEV_PERSISTENT_IP = 0x064C
    GEV_PERSISTENT_SUBNET = 0x065C
    GEV_PERSISTENT_GATEWAY = 0x066C

    # Stream channel 0 (standard)
    SC_HOST_PORT = 0x0D00
    SC_PACKET_SIZE = 0x0D04  # bits 15:0 = size in bytes (512..9216, step 4)
    SC_PACKET_DELAY = 0x0D08  # inter-packet delay, 0..65535
    SC_DEST_ADDR = 0x0D18

    # ------------------------------------------------------------------
    # Device information (strings, read with read_mem)
    # ------------------------------------------------------------------
    VENDOR_NAME = 0x0048  # 32 bytes
    MODEL_NAME = 0x0068  # 32 bytes
    DEVICE_VERSION = 0x0088  # 32 bytes
    SERIAL_NUMBER = 0x00D8  # 16 bytes
    USER_ID = 0x00E8  # 16 bytes, RW
    FIRMWARE_VERSION = 0x10000074  # 20 bytes

    # ------------------------------------------------------------------
    # Sensor / device state (read-only)
    # ------------------------------------------------------------------
    SCAN_TYPE = 0x10000010  # 0 = areascan
    COLOR_TYPE = 0x10000014  # 0 = mono, otherwise Bayer (HC series)
    MAX_THROUGHPUT = 0x10000018  # bytes/s
    PHYSICAL_WIDTH = 0x10000008
    PHYSICAL_HEIGHT = 0x1000000C
    TEN_BIT_IMPLEMENTED = 0x300010BC  # bit 0 (LSB) set = 10-bit output available

    # ------------------------------------------------------------------
    # Image format
    # ------------------------------------------------------------------
    AOI_LEFT = 0x10000024  # OffsetX
    AOI_TOP = 0x10000028  # OffsetY
    AOI_WIDTH = 0x1000002C
    AOI_HEIGHT = 0x10000030
    PIXEL_FORMAT = 0x10000034  # PFNC code, see PixelFormat
    BINNING_H = 0x300010AC  # mono only, 1..2 (digital, no frame-rate gain)
    BINNING_V = 0x300010B0  # mono only, 1..2
    REVERSE_X = 0x300000A0  # mono only, 0/1
    TEST_IMAGE = 0x30000048  # see TestImage

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    ACQUISITION_MODE = 0x10000040  # see AcquisitionMode
    ACQUISITION_FRAME_COUNT = 0x10000044  # 0..32767, MultiFrame only
    ACQUISITION_START = 0x100000A8  # write 1
    ACQUISITION_STOP = 0x100000AC  # write 1 (stops after current frame)
    ACQUISITION_ABORT = 0x100000B0  # write 1 (immediate)
    ACQUISITION_ARM = 0x300010A8  # write 1

    # ------------------------------------------------------------------
    # Timing: frame rate, exposure, trigger
    # ------------------------------------------------------------------
    FRAME_RATE = 0x30000008  # mHz (0.001 Hz); free-run only
    FRAME_RATE_MAX = 0x30001020  # mHz, depends on ROI/binning/pixel format
    FRAME_RATE_MIN = 0x3000104C  # mHz
    SOFTWARE_TRIGGER = 0x3000000C  # write 1 (write-only)
    SHUTTER = 0x30000024  # exposure time in us
    EXPOSURE_MAX = 0x30001024  # us (free-run); 4_000_000 when triggered
    EXPOSURE_MIN = 0x30001040  # us
    EXPOSURE_MODE = 0x30000020  # see ExposureMode (hardware trigger only)
    EXPOSURE_ALIGNMENT = 0x30000030  # see ExposureAlignment (hardware trigger only)
    TRIGGER_DELAY = 0x30000034  # us, 100..4_000_000
    TRIGGER_MODE = 0x30000038  # 0 = off, 1 = on
    TRIGGER_SOURCE = 0x3000003C  # see TriggerSource
    LINE_POLARITY_BASE = 0x30000040  # + 4 * line (inputs 0..1); also trigger activation
    LINE_DEBOUNCE_BASE = 0x30000050  # + 4 * line (inputs 0..1), 0..255 us
    LINE_STATUS_BASE = 0x30000058  # + 4 * line (inputs 0..1), read-only

    # Output lines 2..3 (Line3, Line4): base + (line - 2) * 0x18
    OUT_VALUE_BASE = 0x30000068  # software driven: 0 = open, 1 = close
    OUT_MODE_BASE = 0x3000006C  # 0 = software, 1 = event driven
    OUT_EVENT_BASE = 0x30000070  # see OutputEvent
    OUT_POLARITY_BASE = 0x30000074  # 0 = active close, 1 = active open
    OUT_DELAY_BASE = 0x30000078  # us, 0..65535
    OUT_DURATION_BASE = 0x3000007C  # us, 100..16777251
    OUT_STRIDE = 0x18

    # ------------------------------------------------------------------
    # Gain / black level / processing
    # ------------------------------------------------------------------
    GAIN = 0x30000014  # signed, 0.1 dB steps, 0..120 (0..+12 dB)
    BLACK_LEVEL = 0x3000001C  # 0..1023 (see black_level_increment)
    LUT_ENABLE = 0x30000060  # mono only
    LUT_DATA = 0x31000000  # 1024 entries, two 12-bit values per 32-bit word
    FLATFIELD_MODE = 0x300000B0  # 0 = off, 1 = on, 2 = calibration
    FLATFIELD_OFFSET = 0x31700000  # one byte per pixel
    FLATFIELD_GAIN = 0x31A00000  # one byte per pixel, gain = value / 128

    # ------------------------------------------------------------------
    # Counters / events
    # ------------------------------------------------------------------
    COUNTER_EVENT_SOURCE = 0x30001090  # 0 clock (us), 1 line, 2 end of readout
    COUNTER_LINE_SOURCE = 0x30001094  # 0 = Line1, 1 = Line2
    COUNTER_RESET_MODE = 0x30001098  # 0 software, 1 hardware
    COUNTER_RESET_LINE = 0x3000109C
    EVENT_SELECTOR = 0x100000A0
    EVENT_NOTIFICATION = 0x100000A4

    # ------------------------------------------------------------------
    # Register-set handling and user sets
    # ------------------------------------------------------------------
    REGS_STREAM_START = 0x10000058  # write 1
    REGS_STREAM_END = 0x1000005C  # write 1 (validates the set)
    REGS_VALID = 0x10000060  # read: 1 = consistent
    USERSET_SELECTOR = 0x10000064  # see UserSet
    USERSET_LOAD = 0x10000068  # write 1; reads 1 while busy
    USERSET_SAVE = 0x1000006C  # write 1; reads 1 while busy
    USERSET_DEFAULT = 0x10000070  # set loaded at power-up


# ----------------------------------------------------------------------
# Enumerations (register values)
# ----------------------------------------------------------------------
class PixelFormat(IntEnum):
    MONO_8 = 0x01080001
    MONO_10 = 0x01100003  # 10-bit in a 16-bit container, only if TEN_BIT_IMPLEMENTED
    BAYER_GR_8 = 0x01080008  # colour (HC) models only
    BAYER_GR_10 = 0x0110000C  # colour (HC) models only


class TriggerSourceReg(IntEnum):
    LINE_1 = 0
    LINE_2 = 1
    SOFTWARE = 2


class TriggerActivationReg(IntEnum):
    RISING_EDGE = 0
    FALLING_EDGE = 1


class ExposureModeReg(IntEnum):
    TIMED = 1
    TRIGGER_WIDTH = 2


class ExposureAlignmentReg(IntEnum):
    SYNCHRONOUS = 0
    RESET = 1


class AcquisitionMode(IntEnum):
    CONTINUOUS = 0
    SINGLE_FRAME = 1
    MULTI_FRAME = 2


class TestImage(IntEnum):
    OFF = 0
    HORIZONTAL_WEDGE = 1
    VERTICAL_WEDGE = 2
    DIAGONAL_MOVING_WEDGE = 3
    PURITY = 4


class OutputEvent(IntEnum):
    START_OF_TRIGGER = 0
    START_OF_EXPOSURE = 1
    START_OF_READOUT = 3
    END_OF_READOUT = 4
    END_OF_ACQUISITION = 5
    EVENT_ON_INPUT_LINE_1 = 6
    EVENT_ON_INPUT_LINE_2 = 7
    OVER_TRIGGER = 8


class UserSet(IntEnum):
    FACTORY = 0
    USER_SET_1 = 1


# ----------------------------------------------------------------------
# Limits that are fixed in the XML
# ----------------------------------------------------------------------
GAIN_RAW_MAX = 120  # 0.1 dB steps -> +12 dB
TRIGGER_DELAY_MIN_US = 100
TRIGGER_DELAY_MAX_US = 4_000_000
TRIGGERED_EXPOSURE_MAX_US = 4_000_000
PACKET_SIZE_MIN = 512
PACKET_SIZE_MAX = 9216
PACKET_SIZE_STEP = 4
PACKET_SIZE_MASK = 0xFFFF  # bits 15:0 of SC_PACKET_SIZE


# ----------------------------------------------------------------------
# Enumerations (helpers)
# ----------------------------------------------------------------------
class PixelType(StrEnum):
    MONO_8 = "Mono8"
    MONO_10 = "Mono10"  # 10 bit in a 16 bit container, not all firmware has it

    # Colour (HC) models only
    BAYER_GR_8 = "BayerGR8"
    BAYER_GR_10 = "BayerGR10"

    def to_numpy(self) -> str:
        return "uint8" if self in (self.MONO_8, self.BAYER_GR_8) else "uint16"

    def to_register(self) -> int:
        return {
            self.MONO_8: PixelFormat.MONO_8,
            self.MONO_10: PixelFormat.MONO_10,
            self.BAYER_GR_8: PixelFormat.BAYER_GR_8,
            self.BAYER_GR_10: PixelFormat.BAYER_GR_10,
        }[self].value

    @property
    def is_bayer(self) -> bool:
        return self in (self.BAYER_GR_8, self.BAYER_GR_10)


class TriggerSource(StrEnum):
    INTERNAL = "Internal"  # free-run at `fps`
    SOFTWARE = "Software"
    LINE_1 = "Line1"
    LINE_2 = "Line2"

    def to_register(self) -> int:
        return {
            self.LINE_1: TriggerSourceReg.LINE_1,
            self.LINE_2: TriggerSourceReg.LINE_2,
            self.SOFTWARE: TriggerSourceReg.SOFTWARE,
            self.INTERNAL: -1,
        }[self]


class TriggerActivation(StrEnum):
    RISING_EDGE = "RisingEdge"
    FALLING_EDGE = "FallingEdge"

    def to_register(self) -> int:
        return {
            self.RISING_EDGE: TriggerActivationReg.RISING_EDGE,
            self.FALLING_EDGE: TriggerActivationReg.FALLING_EDGE,
        }[self]


class ExposureMode(StrEnum):
    TIMED = "Timed"
    TRIGGER_WIDTH = "TriggerWidth"  # exposure = trigger pulse width, needs Line1/Line2

    def to_register(self) -> int:
        return {
            self.TIMED: ExposureModeReg.TIMED,
            self.TRIGGER_WIDTH: ExposureModeReg.TRIGGER_WIDTH,
        }[self]


class ExposureAlignment(StrEnum):
    SYNCHRONOUS = "Synchronous"  # readout overlaps exposure, fastest
    RESET = "Reset"  # sensor timing restarts on each trigger

    def to_register(self) -> int:
        return {
            self.SYNCHRONOUS: ExposureAlignmentReg.SYNCHRONOUS,
            self.RESET: ExposureAlignmentReg.RESET,
        }[self]


class OutputLine(StrEnum):
    LINE_3 = "Line3"  # Output 1: Hirose pin 3 (+) / pin 4 (-)
    LINE_4 = "Line4"  # Output 2: Hirose pin 7 (+) / pin 8 (-)

    def to_register(self) -> int:
        return 2 if self == self.LINE_3 else 3


class OutputEventType(StrEnum):
    START_OF_TRIGGER = "StartOfTrigger"
    START_OF_EXPOSURE = "StartOfExposure"
    START_OF_READOUT = "StartOfReadout"
    END_OF_READOUT = "EndOfReadout"

    def to_register(self) -> int:
        return {
            self.START_OF_TRIGGER: OutputEvent.START_OF_TRIGGER,
            self.START_OF_EXPOSURE: OutputEvent.START_OF_EXPOSURE,
            self.START_OF_READOUT: OutputEvent.START_OF_READOUT,
            self.END_OF_READOUT: OutputEvent.END_OF_READOUT,
        }[self].value


class TestPattern(StrEnum):
    OFF = "Off"
    HORIZONTAL_WEDGE = "HorizontalWedge"
    VERTICAL_WEDGE = "VerticalWedge"
    DIAGONAL_MOVING_WEDGE = "DiagonalMovingWedge"
    PURITY = "Purity"

    def to_register(self) -> int:
        return {
            self.OFF: TestImage.OFF,
            self.HORIZONTAL_WEDGE: TestImage.HORIZONTAL_WEDGE,
            self.VERTICAL_WEDGE: TestImage.VERTICAL_WEDGE,
            self.DIAGONAL_MOVING_WEDGE: TestImage.DIAGONAL_MOVING_WEDGE,
            self.PURITY: TestImage.PURITY,
        }[self].value


# ----------------------------------------------------------------------
# Address helpers
# ----------------------------------------------------------------------
def input_line_polarity(line: int) -> int:
    """Polarity register of input line 0 (Line1) or 1 (Line2)."""
    _check_input(line)
    return Reg.LINE_POLARITY_BASE + 4 * line


def input_line_debounce(line: int) -> int:
    _check_input(line)
    return Reg.LINE_DEBOUNCE_BASE + 4 * line


def input_line_status(line: int) -> int:
    _check_input(line)
    return Reg.LINE_STATUS_BASE + 4 * line


def output_line_reg(base: int, line: int) -> int:
    """Address of an output register for line 2 (Line3) or 3 (Line4)."""
    if line not in (2, 3):
        raise ValueError("Output lines are 2 (Line3) and 3 (Line4)")
    return base + (line - 2) * Reg.OUT_STRIDE


def _check_input(line: int) -> None:
    if line not in (0, 1):
        raise ValueError("Input lines are 0 (Line1) and 1 (Line2)")


# ----------------------------------------------------------------------
# Value helpers
# ----------------------------------------------------------------------
def x_increment(binning_h: int) -> int:
    """Step for Width and OffsetX."""
    return 4 // binning_h


def y_increment(is_color: bool) -> int:
    """Step for Height and OffsetY."""
    return 2 if is_color else 1


def pixel_bits(pixel_format: int) -> int:
    """Bits per pixel in the stream: 8, or 16 for the 10-bit formats."""
    return 16 if pixel_format in (PixelFormat.MONO_10, PixelFormat.BAYER_GR_10) else 8


def black_level_increment(pixel_format: int) -> int:
    """BlackLevelRaw step: 1 in 8-bit mode, 4 in 10-bit mode (raw = register * step)."""
    return 1 if pixel_bits(pixel_format) == 8 else 4


# String registers: name -> (address, length in bytes)
STRING_REGISTERS: dict[str, tuple[int, int]] = {
    "vendor": (Reg.VENDOR_NAME, 32),
    "model": (Reg.MODEL_NAME, 32),
    "device_version": (Reg.DEVICE_VERSION, 32),
    "serial": (Reg.SERIAL_NUMBER, 16),
    "user_id": (Reg.USER_ID, 16),
    "firmware": (Reg.FIRMWARE_VERSION, 20),
}
