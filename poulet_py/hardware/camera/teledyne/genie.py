try:
    import socket
    from contextlib import suppress
    from enum import StrEnum
    from threading import Condition, Event, Thread
    from time import monotonic, monotonic_ns, sleep
    from typing import Any, Literal

    from numpy import ndarray, zeros
    from pydantic import Field, PrivateAttr
    from pyGigEVision import GVCPClient, GVCPError, GVSPReceiver, discover

    from poulet_py import LOGGER, AcquisitionType
    from poulet_py.hardware import BaseHardware
    from poulet_py.hardware.camera.teledyne.registers.genie import (
        GAIN_RAW_MAX,
        PACKET_SIZE_MAX,
        PACKET_SIZE_MIN,
        PACKET_SIZE_STEP,
        STRING_REGISTERS,
        TRIGGER_DELAY_MAX_US,
        TRIGGER_DELAY_MIN_US,
        TRIGGERED_EXPOSURE_MAX_US,
        AcquisitionMode,
        ExposureAlignment,
        ExposureMode,
        PixelFormat,
        Reg,
        TestImage,
        UserSet,
        black_level_increment,
        input_line_polarity,
        x_increment,
        y_increment,
    )
    from poulet_py.hardware.camera.teledyne.registers.genie import (
        TriggerActivation as _RegTriggerActivation,
    )
    from poulet_py.hardware.camera.teledyne.registers.genie import (
        TriggerSource as _RegTriggerSource,
    )
except ImportError as e:
    raise ImportError("""
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]   (needs pyGigEVision)
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


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


class TriggerActivationMode(StrEnum):
    RISING_EDGE = "RisingEdge"
    FALLING_EDGE = "FallingEdge"


class ExposureModeType(StrEnum):
    TIMED = "Timed"
    TRIGGER_WIDTH = "TriggerWidth"  # exposure = trigger pulse width, needs Line1/Line2


class ExposureAlignmentType(StrEnum):
    SYNCHRONOUS = "Synchronous"  # readout overlaps exposure, fastest
    RESET = "Reset"  # sensor timing restarts on each trigger


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


_TRIGGER_SOURCE_REG = {
    TriggerSource.LINE_1: _RegTriggerSource.LINE_1,
    TriggerSource.LINE_2: _RegTriggerSource.LINE_2,
    TriggerSource.SOFTWARE: _RegTriggerSource.SOFTWARE,
}
_TRIGGER_ACTIVATION_REG = {
    TriggerActivationMode.RISING_EDGE: _RegTriggerActivation.RISING_EDGE,
    TriggerActivationMode.FALLING_EDGE: _RegTriggerActivation.FALLING_EDGE,
}

_SLICE_S = 0.1  # how long a blocking wait runs before re-checking the stop event


class GenieCamera(BaseHardware):
    camera_ip: str | None = Field(
        default=None, description="Camera IPv4. If None, the camera at `device_index` is discovered"
    )
    device_index: int = Field(default=0, description="Index among discovered cameras")
    local_ip: str = Field(
        default="",
        description="IPv4 of the NIC facing the camera. Empty = pick it from the routing table",
    )
    local_port: int = Field(
        default=0, description="UDP port to receive the stream on. 0 = any free port"
    )
    heartbeat_ms: int = Field(default=3000, ge=500, le=65000)
    trigger_source: TriggerSource = Field(default=TriggerSource.INTERNAL)
    trigger_activation: TriggerActivationMode = Field(default=TriggerActivationMode.RISING_EDGE)
    trigger_delay: int | None = Field(
        default=None, description="Delay from hardware trigger to exposure, us (100..4e6)"
    )
    timeout: int | None = Field(default=None, description="frame wait timeout in ms")
    fps: float = Field(default=30, gt=0, description="Frame rate in Hz (free-run only)")
    pixel_type: PixelType = Field(default=PixelType.MONO_8)
    resolution: tuple[int, int] | None = Field(
        default=None,
        description="Optional camera-native ROI as (width, height). None = full sensor",
    )
    offset: tuple[int, int] | Literal["center"] = Field(
        default="center", description="ROI offset (x, y), or 'center'"
    )
    binning: tuple[int, int] = Field(
        default=(1, 1), description="(horizontal, vertical), 1 or 2. Digital, mono models only"
    )
    reverse_x: bool = Field(default=False, description="Horizontal flip (mono models only)")
    test_pattern: TestPattern = Field(default=TestPattern.OFF)
    exposure: float = Field(default=10000, ge=1, description="Exposure time in us")
    exposure_mode: ExposureModeType = Field(default=ExposureModeType.TIMED)
    exposure_alignment: ExposureAlignmentType = Field(default=ExposureAlignmentType.SYNCHRONOUS)
    gain: float = Field(default=0.0, ge=0, le=12, description="Digital gain in dB")
    black_level: int = Field(
        default=0, ge=0, le=4092, description="Raw black level (step 1 in 8-bit, 4 in 10-bit)"
    )
    packet_size: int = Field(default=1500, ge=PACKET_SIZE_MIN, le=PACKET_SIZE_MAX)
    packet_delay: int = Field(default=0, ge=0, le=65535, description="Inter-packet delay")
    byteswap: bool = Field(default=False, description="Swap bytes of 16-bit pixels")
    load_factory_defaults: bool = Field(
        default=False,
        description="Load the factory user set before configuring (discards the power-up set)",
    )

    _client: GVCPClient | None = PrivateAttr(default=None)
    _receiver: GVSPReceiver | None = PrivateAttr(default=None)
    _ip: str = PrivateAttr(default="")
    _info: dict = PrivateAttr(default_factory=dict)
    _width: int = PrivateAttr(default=0)
    _height: int = PrivateAttr(default=0)
    _tick_hz: int = PrivateAttr(default=1_000_000)
    _is_color: bool = PrivateAttr(default=False)
    _capturing: bool = PrivateAttr(default=False)
    _dropped_frames: int = PrivateAttr(default=0)

    _first_monotonic_time: int = PrivateAttr(default=0)
    _first_sample_time: int = PrivateAttr(default=0)

    @staticmethod
    def get_available_devices(timeout: float = 2.0) -> list[dict[str, str | int]]:
        res = []
        for idx, dev in enumerate(discover(timeout=timeout)):
            res.append(
                {
                    "device_id": idx,
                    "name": f"{dev.get('manufacturer', '')} {dev.get('model', '')}".strip(),
                    "ip": dev["ip"],
                    "serial": dev.get("serial", ""),
                }
            )
        return res

    @property
    def dropped_frames(self) -> int:
        """Frames discarded because packets were missing or the size was wrong."""
        return self._dropped_frames

    def _open(self) -> None:
        try:
            self._connect()
            self._configure()
            self._set_buffer()
            self._set_stream_channel()

        except Exception as e:
            self._teardown()
            raise RuntimeError("Failed to open Genie") from e

    def _close(self) -> None:
        self._stop_acquisition_thread()
        if self._client is not None:
            with suppress(GVCPError, OSError):
                self._stop_capture()
                self._write(Reg.ACQUISITION_ABORT, 1)
        self._capturing = False
        if self._receiver is not None:
            with suppress(Exception):
                self._receiver.close()
            self._receiver = None
        if self._client is not None:
            with suppress(Exception):
                self._client.disconnect()
            self._client = None
        self._buffer = None
        self._buffer_idx = 0

    def info(self) -> dict[str, Any]:
        self._ensure_open()
        return {"device_id": self.device_index, "ip": self._ip, **self._info}

    def software_trigger(self) -> None:
        self._ensure_open()

        if self.trigger_source == TriggerSource.INTERNAL:
            raise RuntimeError("Software trigger needs a trigger_source other than INTERNAL")

        self._write(Reg.SOFTWARE_TRIGGER, 1)

    def _read(self, addr: int) -> int:
        if not self._client:
            raise RuntimeError(f"{type(self).__name__} no open client")

        return self._client.read_reg(addr)

    def _write(self, addr: int, value: int) -> None:
        if not self._client:
            raise RuntimeError(f"{type(self).__name__} no open client")

        self._client.write_reg(addr, value)

    def _read_string(self, addr: int, length: int) -> str:
        if not self._client:
            raise RuntimeError(f"{type(self).__name__} no open client")

        raw = self._client.read_mem(addr, length)
        return raw.split(b"\x00")[0].decode("utf-8", errors="replace").strip()

    def _wait_idle(self, addr: int, what: str, timeout: float = 5.0) -> None:
        end = monotonic() + timeout
        while self._read(addr) == 1:
            if monotonic() > end:
                raise TimeoutError(f"Camera still busy: {what}")
            sleep(0.02)

    def _connect(self) -> None:
        self._ip = self.camera_ip or self._discover_ip()
        self._client = GVCPClient(self._ip, local_ip=self.local_ip or None)
        self._client.connect()
        self._write(Reg.GEV_HEARTBEAT_TIMEOUT, self.heartbeat_ms)

        self._info = {
            key: self._read_string(addr, length) for key, (addr, length) in STRING_REGISTERS.items()
        }
        self._info["name"] = f"{self._info['vendor']} {self._info['model']}".strip()
        if "genie" not in self._info["model"].lower():
            LOGGER.warning(f"Device model '{self._info['model']}' is not a Genie, continuing")

        freq = (self._read(Reg.GEV_TIMESTAMP_FREQ_HIGH) << 32) | self._read(
            Reg.GEV_TIMESTAMP_FREQ_LOW
        )
        self._tick_hz = freq or 1_000_000

    def _discover_ip(self) -> str:
        devices = discover(interface_ip=self.local_ip, timeout=2.0)
        if not devices:
            raise RuntimeError(f"{type(self).__name__} No camera found.")

        if self.device_index >= len(devices):
            raise IndexError(
                f"{type(self).__name__} Device index {self.device_index} out of range. "
                "Use get_available_devices() to list available cameras."
            )
        return devices[self.device_index]["ip"]

    def _configure(self) -> None:
        if self.load_factory_defaults:
            self._write(Reg.USERSET_SELECTOR, UserSet.FACTORY)
            self._write(Reg.USERSET_LOAD, 1)
            self._wait_idle(Reg.USERSET_LOAD, "user set load")

        self._is_color = self._read(Reg.COLOR_TYPE) != 0
        # Image format first: frame-rate and exposure limits depend on it.
        self._set_pixel_format()
        self._set_roi()
        self._write(Reg.TEST_IMAGE, self.test_pattern.to_register())
        self._write(Reg.ACQUISITION_MODE, AcquisitionMode.CONTINUOUS)
        # Then trigger, then timing (exposure limits depend on frame rate).
        self._set_trigger()
        self._set_exposure()
        self._set_gain_black_level()

        if not self._read(Reg.REGS_VALID):
            LOGGER.warning("Camera reports an inconsistent register set after configuration")

        self._width = self._read(Reg.AOI_WIDTH)
        self._height = self._read(Reg.AOI_HEIGHT)

    def _set_pixel_format(self) -> None:
        if self.pixel_type.is_bayer != self._is_color:
            raise ValueError(
                f"Pixel type {self.pixel_type.value} does not match a "
                f"{'colour' if self._is_color else 'mono'} camera"
            )
        if self.pixel_type in (PixelType.MONO_10, PixelType.BAYER_GR_10) and not (
            self._read(Reg.TEN_BIT_IMPLEMENTED) & 1
        ):
            raise ValueError("This camera/firmware does not support 10-bit output")
        self._write(Reg.PIXEL_FORMAT, self.pixel_type.to_register())

    def _set_roi(self) -> None:
        hb, vb = self.binning
        if not (hb in (1, 2) and vb in (1, 2)):
            raise ValueError("Binning must be 1 or 2 in each direction")
        if self._is_color:
            if self.binning != (1, 1) or self.reverse_x:
                raise ValueError("Binning and reverse_x are only available on mono models")
        else:
            self._write(Reg.BINNING_H, hb)
            self._write(Reg.BINNING_V, vb)
            self._write(Reg.REVERSE_X, int(self.reverse_x))

        max_w = self._read(Reg.PHYSICAL_WIDTH) // hb
        max_h = self._read(Reg.PHYSICAL_HEIGHT) // vb
        inc_x = x_increment(hb)
        inc_y = y_increment(self._is_color)

        width, height = self.resolution or (max_w, max_h)
        if not (inc_x <= width <= max_w and inc_y <= height <= max_h):
            raise ValueError(f"Resolution {(width, height)} out of limits, max {(max_w, max_h)}")
        if width % inc_x or height % inc_y:
            raise ValueError(f"Width must be a multiple of {inc_x} and height of {inc_y}")

        if self.offset == "center":
            off_x = (max_w - width) // 2 // inc_x * inc_x
            off_y = (max_h - height) // 2 // inc_y * inc_y
        else:
            off_x, off_y = self.offset
            if off_x % inc_x or off_y % inc_y:
                raise ValueError(f"Offset X must be a multiple of {inc_x} and Y of {inc_y}")
        if off_x + width > max_w or off_y + height > max_h:
            raise ValueError("ROI plus offset exceeds the sensor area")

        # Zero the offsets first so a bigger window is never rejected.
        self._write(Reg.AOI_LEFT, 0)
        self._write(Reg.AOI_TOP, 0)
        self._write(Reg.AOI_WIDTH, width)
        self._write(Reg.AOI_HEIGHT, height)
        self._write(Reg.AOI_LEFT, off_x)
        self._write(Reg.AOI_TOP, off_y)

    @property
    def _is_hardware_trigger(self) -> bool:
        return self.trigger_source in (TriggerSource.LINE_1, TriggerSource.LINE_2)

    def _set_trigger(self) -> None:
        if self.trigger_source == TriggerSource.INTERNAL:
            self._write(Reg.TRIGGER_MODE, 0)
            return

        self._write(Reg.TRIGGER_MODE, 1)
        source = _TRIGGER_SOURCE_REG[self.trigger_source]
        self._write(Reg.TRIGGER_SOURCE, source)

        if self._is_hardware_trigger:
            line = 0 if self.trigger_source == TriggerSource.LINE_1 else 1
            self._write(input_line_polarity(line), _TRIGGER_ACTIVATION_REG[self.trigger_activation])
            if self.trigger_delay is not None:
                if not TRIGGER_DELAY_MIN_US <= self.trigger_delay <= TRIGGER_DELAY_MAX_US:
                    raise ValueError(
                        f"trigger_delay must be {TRIGGER_DELAY_MIN_US}..{TRIGGER_DELAY_MAX_US} us"
                    )
                self._write(Reg.TRIGGER_DELAY, self.trigger_delay)

    def _set_exposure(self) -> None:
        # Timing modes only exist with a hardware trigger; otherwise the camera is Timed.
        if self._is_hardware_trigger:
            mode = (
                ExposureMode.TRIGGER_WIDTH
                if self.exposure_mode == ExposureModeType.TRIGGER_WIDTH
                else ExposureMode.TIMED
            )
            self._write(Reg.EXPOSURE_MODE, mode)
            if mode == ExposureMode.TIMED:
                align = (
                    ExposureAlignment.RESET
                    if self.exposure_alignment == ExposureAlignmentType.RESET
                    else ExposureAlignment.SYNCHRONOUS
                )
                self._write(Reg.EXPOSURE_ALIGNMENT, align)
        elif self.exposure_mode == ExposureModeType.TRIGGER_WIDTH:
            raise ValueError("TRIGGER_WIDTH exposure needs trigger_source LINE_1 or LINE_2")

        # Free-run: frame rate is in mHz and limits depend on ROI, so set it before exposure.
        if self.trigger_source == TriggerSource.INTERNAL:
            lo, hi = self._read(Reg.FRAME_RATE_MIN), self._read(Reg.FRAME_RATE_MAX)
            target = round(self.fps * 1000)
            if not lo <= target <= hi:
                raise ValueError(f"fps {self.fps} out of range {lo / 1000:.3f}..{hi / 1000:.3f} Hz")
            self._write(Reg.FRAME_RATE, target)

        if self.exposure_mode == ExposureModeType.TIMED:
            lo = self._read(Reg.EXPOSURE_MIN)
            hi = (
                TRIGGERED_EXPOSURE_MAX_US
                if self.trigger_source != TriggerSource.INTERNAL
                else self._read(Reg.EXPOSURE_MAX)
            )
            target = round(self.exposure)
            if not lo <= target <= hi:
                raise ValueError(f"exposure {self.exposure} us out of range {lo}..{hi} us")
            self._write(Reg.SHUTTER, target)

    def _set_gain_black_level(self) -> None:
        raw_gain = round(self.gain * 10)
        if not 0 <= raw_gain <= GAIN_RAW_MAX:
            raise ValueError("gain must be 0..12 dB")
        self._write(Reg.GAIN, raw_gain)

        step = black_level_increment(self.pixel_type.to_register())
        if self.black_level % step or self.black_level > 1023 * step:
            raise ValueError(f"black_level must be a multiple of {step}, max {1023 * step}")
        self._write(Reg.BLACK_LEVEL, self.black_level // step)

    def _route_ip(self) -> str:
        """Local IPv4 address that reaches the camera."""
        if self.local_ip:
            return self.local_ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self._ip, 3956))
            return s.getsockname()[0]
        finally:
            s.close()

    def _set_stream_channel(self) -> None:
        size = self.packet_size
        if size % PACKET_SIZE_STEP:
            raise ValueError(f"packet_size must be a multiple of {PACKET_SIZE_STEP}")

        local_ip = self._route_ip()
        self._receiver = GVSPReceiver(
            local_ip=local_ip,
            local_port=self.local_port,
            gvcp_client=self._client,
            packet_size=size,
            byteswap=self.byteswap,
            camera_ip=self._ip,
        )
        self._receiver.start()

        ip_int = int.from_bytes(socket.inet_aton(local_ip), "big")
        self._write(Reg.SC_DEST_ADDR, ip_int)
        self._write(Reg.SC_HOST_PORT, self._receiver.port)
        current = self._read(Reg.SC_PACKET_SIZE)
        self._write(Reg.SC_PACKET_SIZE, (current & 0xFFFF0000) | size)
        self._write(Reg.SC_PACKET_DELAY, self.packet_delay)

    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
        return [
            ("data", self.pixel_type.to_numpy(), (self._height, self._width)),
        ]

    def _timeout_s(self) -> float | None:
        return None if self.timeout is None else self.timeout / 1000

    def _start_capture(self) -> None:
        if not self._receiver:
            raise RuntimeError(f"{type(self).__name__} Receiver is not set")

        if self._capturing:
            return

        if self.acquisition_type == AcquisitionType.FINITE:
            self._receiver.stop()
            self._receiver.flush()
            self._receiver.start()

        self._write(Reg.ACQUISITION_START, 1)
        self._capturing = True

    def _stop_capture(self) -> None:
        if not self._capturing:
            return

        self._capturing = False
        self._write(Reg.ACQUISITION_STOP, 1)

    def _acquire_sample(self) -> ndarray | None:
        if not self._receiver:
            raise RuntimeError(f"{type(self).__name__} Receiver is not set")

        if self.trigger_source == TriggerSource.SOFTWARE:
            self._write(Reg.SOFTWARE_TRIGGER, 1)

        return self._receiver.get_frame(timeout=_SLICE_S)

    def _acquisition_thread_func(self) -> None:
        receiver, stop = self._receiver, self._stop_acquisition_event
        try:
            if not receiver:
                raise RuntimeError(f"{type(self).__name__} Receiver is not set")

            self._start_capture()

            while not stop.is_set():
                item = receiver.get_frame_with_info(timeout=_SLICE_S)
                if item is not None:
                    self._store(*item)

        except Exception as e:
            LOGGER.exception(e)
        finally:
            with self._acquisition_condition:
                self._acquisition_condition.notify_all()
