try:
    from contextlib import suppress
    from time import monotonic, monotonic_ns, sleep
    from typing import Literal

    from numpy import ndarray
    from pydantic import BaseModel, Field, IPvAnyAddress, PrivateAttr
    from pyGigEVision import GVCPClient, GVCPError, GVSPReceiver, discover

    from poulet_py import LOGGER, AcquisitionType
    from poulet_py.hardware import BaseHardware
    from poulet_py.hardware.camera.teledyne.registers.genie import (
        GAIN_RAW_MAX,
        PACKET_SIZE_MAX,
        PACKET_SIZE_MIN,
        PACKET_SIZE_STEP,
        TRIGGER_DELAY_MAX_US,
        TRIGGER_DELAY_MIN_US,
        TRIGGERED_EXPOSURE_MAX_US,
        AcquisitionMode,
        ExposureAlignment,
        ExposureAlignmentType,
        ExposureMode,
        ExposureModeType,
        OutputEventType,
        OutputLine,
        PixelType,
        Reg,
        TestPattern,
        TriggerActivation,
        TriggerSource,
        UserSet,
        black_level_increment,
        input_line_polarity,
        output_line_reg,
        x_increment,
        y_increment,
    )
except ImportError as e:
    raise ImportError("""
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]   (needs pyGigEVision)
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class DeviceInfo(BaseModel):
    device_id: int | None = Field(...)
    ip: IPvAnyAddress = Field(...)
    local_ip: IPvAnyAddress = Field(...)
    manufacturer: str | None = Field(default=None)
    model: str | None = Field(default=None)
    serial: str | None = Field(default=None)
    firmware: str | None = Field(default=None)


class GenieCamera(BaseHardware):
    device: IPvAnyAddress | int = Field(..., description="Camera IPv4. or device's index")
    local_port: int = Field(
        default=0, description="UDP port to receive the stream on. 0 = any free port"
    )
    heartbeat_ms: int = Field(default=3000, ge=500, le=65000)
    trigger_source: TriggerSource = Field(default=TriggerSource.INTERNAL)
    trigger_activation: TriggerActivation = Field(default=TriggerActivation.RISING_EDGE)
    trigger_delay: int | None = Field(
        default=None, description="Delay from hardware trigger to exposure, us (100..4e6)"
    )
    timeout: int = Field(default=5000, description="frame wait timeout in ms")
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
    output_line: OutputLine | None = Field(
        default=None, description="Output line that pulses once per frame. None = leave untouched"
    )
    output_event: OutputEventType = Field(default=OutputEventType.START_OF_EXPOSURE)
    output_active: Literal["open", "close"] = Field(
        default="open",
        description="Opto state during the pulse. With an external pull-up, 'open' gives a "
        "positive pulse and 'close' a negative one",
    )
    packet_size: int = Field(default=1500, ge=PACKET_SIZE_MIN, le=PACKET_SIZE_MAX)
    packet_delay: int = Field(default=0, ge=0, le=65535, description="Inter-packet delay")
    byteswap: bool = Field(default=False, description="Swap bytes of 16-bit pixels")
    load_factory_defaults: bool = Field(
        default=False,
        description="Load the factory user set before configuring (discards the power-up set)",
    )

    _device_info: DeviceInfo = PrivateAttr()

    _client: GVCPClient = PrivateAttr()
    _receiver: GVSPReceiver = PrivateAttr()

    _is_color: bool = PrivateAttr(default=False)
    _resolution: tuple[int, int] = PrivateAttr()
    _capturing: bool = PrivateAttr(default=False)

    @staticmethod
    def get_available_devices(interface_ip: str = "", timeout: float = 2.0) -> list[DeviceInfo]:
        res = []
        for idx, dev in enumerate(discover(interface_ip=interface_ip, timeout=timeout)):
            res.append(
                DeviceInfo(
                    device_id=idx,
                    ip=dev.get("ip", ""),
                    local_ip=dev.get("interface_ip", ""),
                    manufacturer=dev.get("manufacturer", ""),
                    model=dev.get("model", ""),
                    serial=dev.get("serial", ""),
                    firmware=dev.get("device_version", ""),
                )
            )
        return res

    def info(self) -> DeviceInfo:
        self._ensure_open()
        return self._device_info

    def software_trigger(self) -> None:
        self._ensure_open()

        if self.trigger_source == TriggerSource.INTERNAL:
            raise RuntimeError("Software trigger needs a trigger_source other than INTERNAL")

        self._write(Reg.SOFTWARE_TRIGGER, 1)

    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
        return [
            ("data", self.pixel_type.to_numpy(), (self._resolution[1], self._resolution[0])),
        ]

    def _open(self) -> None:
        try:
            self._discover_camera()
            self._connect()
            self._configure()
            self._set_stream_channel()

        except Exception as e:
            self._close()
            raise RuntimeError("Failed to open Genie") from e

    def _close(self) -> None:
        if hasattr(self, "_client"):
            with suppress(GVCPError, OSError):
                self._stop_capture()
                self._write(Reg.ACQUISITION_ABORT, 1)

        self._capturing = False

        if hasattr(self, "_receiver"):
            with suppress(Exception):
                self._receiver.close()
            del self._receiver

        if hasattr(self, "_client"):
            with suppress(Exception):
                self._client.disconnect()
            del self._client

    def _start_capture(self) -> None:
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
        if self.trigger_source == TriggerSource.SOFTWARE:
            self._write(Reg.SOFTWARE_TRIGGER, 1)

        return self._receiver.get_frame(timeout=self.timeout / 1000)

    def _acquisition_thread_func(self) -> None:
        try:
            self._start_capture()

            while not self._stop_acquisition_event.is_set():
                start = monotonic_ns()
                sample = self._acquire_sample()
                end = monotonic_ns()

                if sample is None:
                    continue

                self._sample_to_buffer(sample=sample, timestamp=(end + start) // 2)
        except Exception as e:
            LOGGER.exception(e)
            self._close()
        finally:
            with self._acquisition_condition:
                self._acquisition_condition.notify_all()

    def _read(self, addr: int) -> int:
        return self._client.read_reg(addr)

    def _write(self, addr: int, value: int) -> None:
        self._client.write_reg(addr, value)

    def _read_string(self, addr: int, length: int) -> str:
        raw = self._client.read_mem(addr, length)
        return raw.split(b"\x00")[0].decode("utf-8", errors="replace").strip()

    def _wait_idle(self, addr: int, what: str, timeout: float = 5.0) -> None:
        end = monotonic() + timeout
        while self._read(addr) == 1:
            if monotonic() > end:
                raise TimeoutError(f"Camera still busy: {what}")
            sleep(0.02)

    def _discover_camera(self) -> None:
        devices = self.get_available_devices()
        if not devices:
            raise RuntimeError(f"{type(self).__name__} No camera found.")

        if isinstance(self.device, str):
            for dev in devices:
                if dev.ip == self.device:
                    self._device_info = dev
                    break

        if isinstance(self.device, int):
            if self.device >= len(devices):
                raise IndexError(
                    f"{type(self).__name__} Device index {self.device} out of range. "
                    "Use get_available_devices() to list available cameras."
                )

            self._device_info = devices[self.device]

    def _connect(self) -> None:
        self._client = GVCPClient(
            str(self._device_info.ip), local_ip=str(self._device_info.local_ip)
        )
        if self._client is None:
            raise RuntimeError(f"{type(self).__name__} Could not init Client")

        self._client.connect()

        self._write(Reg.GEV_HEARTBEAT_TIMEOUT, self.heartbeat_ms)

    def _configure(self) -> None:
        if self.load_factory_defaults:
            self._write(Reg.USERSET_SELECTOR, UserSet.FACTORY)
            self._write(Reg.USERSET_LOAD, 1)
            self._wait_idle(Reg.USERSET_LOAD, "user set load")

        self._is_color = self._read(Reg.COLOR_TYPE) != 0

        self._set_pixel_format()
        self._set_roi()
        self._write(Reg.TEST_IMAGE, self.test_pattern.to_register())
        # TODO for single sample?
        self._write(Reg.ACQUISITION_MODE, AcquisitionMode.CONTINUOUS)
        self._set_trigger()
        self._set_exposure()
        self._set_gain_black_level()
        self._set_output()

        if not self._read(Reg.REGS_VALID):
            LOGGER.warning("Camera reports an inconsistent register set after configuration")

        self._resolution = (self._read(Reg.AOI_WIDTH), self._read(Reg.AOI_HEIGHT))

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

    # TODO
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

    # TODO
    def _set_trigger(self) -> None:
        if self.trigger_source == TriggerSource.INTERNAL:
            self._write(Reg.TRIGGER_MODE, 0)
            return

        self._write(Reg.TRIGGER_MODE, 1)
        self._write(Reg.TRIGGER_SOURCE, self.trigger_source.to_register())

        if self._is_hardware_trigger:
            line = 0 if self.trigger_source == TriggerSource.LINE_1 else 1
            self._write(input_line_polarity(line), self.trigger_activation.to_register())
            if self.trigger_delay is not None:
                if not TRIGGER_DELAY_MIN_US <= self.trigger_delay <= TRIGGER_DELAY_MAX_US:
                    raise ValueError(
                        f"trigger_delay must be {TRIGGER_DELAY_MIN_US}..{TRIGGER_DELAY_MAX_US} us"
                    )
                self._write(Reg.TRIGGER_DELAY, self.trigger_delay)

    # TODO
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

    def _set_output(self) -> None:
        if self.output_line is None:
            return

        line = self.output_line.to_register()
        self._write(output_line_reg(Reg.OUT_MODE_BASE, line), 1)
        self._write(output_line_reg(Reg.OUT_EVENT_BASE, line), self.output_event.to_register())
        self._write(
            output_line_reg(Reg.OUT_POLARITY_BASE, line), 1 if self.output_active == "open" else 0
        )
        self._write(output_line_reg(Reg.OUT_DELAY_BASE, line), 0)

        period_us = 1e6 / self.fps
        self._write(output_line_reg(Reg.OUT_DURATION_BASE, line), int(period_us * 0.2))

    def _set_stream_channel(self) -> None:
        size = self.packet_size
        if size % PACKET_SIZE_STEP:
            raise ValueError(f"packet_size must be a multiple of {PACKET_SIZE_STEP}")

        self._receiver = GVSPReceiver(
            local_ip=str(self._device_info.local_ip),
            local_port=self.local_port,
            gvcp_client=self._client,
            packet_size=size,
            byteswap=self.byteswap,
            camera_ip=str(self._device_info.ip),
        )
        self._receiver.start()

        self._write(Reg.SC_DEST_ADDR, int(self._device_info.local_ip))
        self._write(Reg.SC_HOST_PORT, self._receiver.port)
        current = self._read(Reg.SC_PACKET_SIZE)
        self._write(Reg.SC_PACKET_SIZE, (current & 0xFFFF0000) | size)
        self._write(Reg.SC_PACKET_DELAY, self.packet_delay)
