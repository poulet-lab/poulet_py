try:
    from ctypes import byref, c_double, c_int32
    from enum import Enum
    from typing import Any, Literal

    from numpy import ndarray
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER
    from poulet_py.hardware import BaseHardware
    from poulet_py.hardware.camera.hamamatzu._api import (
        DCAM_IDPROP,
        DCAM_IDSTR,
        DCAM_PIXELTYPE,
        DCAMAPI_INIT,
        DCAMBUF_FRAME,
        DCAMCAP_START,
        DCAMCAP_STATUS,
        DCAMDEV_OPEN,
        DCAMDEV_STRING,
        DCAMERR,
        DCAMPROP,
        DCAMWAIT_CAPEVENT,
        DCAMWAIT_OPEN,
        DCAMWAIT_START,
        dcamapi_init,
        dcamapi_uninit,
        dcambuf_alloc,
        dcambuf_release,
        dcamcap_firetrigger,
        dcamcap_start,
        dcamcap_status,
        dcamcap_stop,
        dcamdev_close,
        dcamdev_getstring,
        dcamdev_open,
        dcamprop_getvalue,
        dcamprop_setvalue,
        dcamwait_abort,
        dcamwait_close,
        dcamwait_open,
        dcamwait_start,
    )
except ImportError as e:
    raise ImportError("""
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class DCAM(BaseHardware):
    DTYPE_MAP: dict[DCAM_PIXELTYPE, str] = {
        DCAM_PIXELTYPE.MONO16: "uint16",
        DCAM_PIXELTYPE.MONO8: "uint8",
    }

    device_index: int = Field(default=0, description="Camera device index")
    pixel_type: DCAM_PIXELTYPE = Field(
        default=DCAM_PIXELTYPE.MONO16, description="Camera pixel type."
    )
    sensor_mode: DCAMPROP.SENSORMODE = Field(
        default=DCAMPROP.SENSORMODE.AREA, description="Camera sensor mode."
    )
    shutter_mode: DCAMPROP.SHUTTER_MODE = Field(
        default=DCAMPROP.SHUTTER_MODE.GLOBAL, description="Camera shutter mode."
    )
    readout_speed: DCAMPROP.READOUTSPEED = Field(
        default=DCAMPROP.READOUTSPEED.FASTEST, description="Camera readout speed."
    )
    readout_direction: DCAMPROP.READOUT_DIRECTION = Field(
        default=DCAMPROP.READOUT_DIRECTION.FORWARD, description="Camera readout direction."
    )
    trigger_source: DCAMPROP.TRIGGERSOURCE = Field(
        default=DCAMPROP.TRIGGERSOURCE.INTERNAL, description="Camera trigger source."
    )
    trigger_mode: DCAMPROP.TRIGGER_MODE = Field(
        default=DCAMPROP.TRIGGER_MODE.NORMAL, description="Camera trigger mode."
    )
    trigger_active: DCAMPROP.TRIGGERACTIVE = Field(
        default=DCAMPROP.TRIGGERACTIVE.EDGE, description="Camera trigger active mode."
    )
    trigger_polarity: DCAMPROP.TRIGGERPOLARITY = Field(
        default=DCAMPROP.TRIGGERPOLARITY.POSITIVE, description="Camera trigger polarity."
    )
    trigger_global_exposure: DCAMPROP.TRIGGER_GLOBALEXPOSURE | None = Field(
        default=DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED,
        description=(
            "Global-exposure timing mode. DELAYED is usually appropriate for "
            "rolling-shutter CMOS cameras; set None to leave the camera default unchanged."
        ),
    )
    output_trigger_connector: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Output trigger connector index. Usually 1 for the first timing output.",
    )
    output_trigger_kind: DCAMPROP.OUTPUTTRIGGER_KIND = Field(
        default=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE, description="Output trigger kind."
    )
    output_trigger_source: DCAMPROP.OUTPUTTRIGGER_SOURCE = Field(
        default=DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
        description="Output trigger source for programmable output triggers.",
    )
    output_trigger_polarity: DCAMPROP.OUTPUTTRIGGER_POLARITY = Field(
        default=DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE, description="Output trigger polarity."
    )
    output_trigger_active: DCAMPROP.OUTPUTTRIGGER_ACTIVE = Field(
        default=DCAMPROP.OUTPUTTRIGGER_ACTIVE.EDGE,
        description="Output trigger active mode for programmable output triggers.",
    )
    output_trigger_basesensor: DCAMPROP.OUTPUTTRIGGER_BASESENSOR | None = Field(
        default=DCAMPROP.OUTPUTTRIGGER_BASESENSOR.VIEW1,
        description="Base sensor/view for global-exposure output trigger. Usually VIEW1.",
    )
    output_trigger_delay: float = Field(
        default=0.0, ge=0, description="Output trigger delay in seconds."
    )
    output_trigger_period: float = Field(
        default=0.001, gt=0, description="Output trigger period in seconds."
    )
    binning: DCAMPROP.BINNING = Field(
        default=DCAMPROP.BINNING._1, description="Camera-native binning mode."
    )
    resolution: tuple[int, int] | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional camera-native ROI/resolution as (width, height). "
            "Implemented through DCAM subarray properties, not software resizing."
        ),
    )
    subarray_hpos: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional horizontal ROI start position. If None, ROI is centered when center_roi=True."
        ),
    )
    subarray_vpos: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional vertical ROI start position. If None, ROI is centered when center_roi=True."
        ),
    )
    subarray_mode: DCAMPROP.MODE | None = Field(
        default=None,
        description=(
            "Optional explicit subarray mode. If resolution is set, subarray mode is forced ON. "
            "If resolution is None and this is None, subarray mode is left unchanged."
        ),
    )
    center_roi: bool = Field(
        default=True,
        description="Center ROI automatically when resolution is set and positions are omitted.",
    )
    exposure_time: int = Field(
        default=50,
        gt=1,
        lt=10000,
        description="Exposure time in ms.",
    )
    frame_rate: float | None = Field(
        default=10.0,
        gt=0,
        description=(
            "Requested acquisition frame rate in frames per second. "
            "For timing_mode='masterpulse', this becomes MASTERPULSE_INTERVAL = 1/frame_rate."
        ),
    )
    timing_mode: Literal["internal", "masterpulse"] = Field(
        default="masterpulse",
        description=(
            "Use 'masterpulse' for camera-native fixed-rate acquisition when "
            "INTERNALFRAMERATE is not writable."
        ),
    )
    masterpulse_mode: DCAMPROP.MASTERPULSE_MODE = Field(
        default=DCAMPROP.MASTERPULSE_MODE.CONTINUOUS,
        description="Master pulse mode. CONTINUOUS gives a continuous fixed-rate acquisition clock.",
    )
    masterpulse_triggersource: DCAMPROP.MASTERPULSE_TRIGGERSOURCE = Field(
        default=DCAMPROP.MASTERPULSE_TRIGGERSOURCE.SOFTWARE,
        description="Master pulse trigger source.",
    )
    masterpulse_bursttimes: int = Field(
        default=1,
        description="Only relevant for MASTERPULSE_MODE.BURST.",
        ge=1,
    )
    contrast_gain: int = Field(default=10, description="Camera contrast gain.")
    framebundle_mode: DCAMPROP.MODE = Field(
        default=DCAMPROP.MODE.OFF, description="Frame bundle mode."
    )
    framebundle_number: int = Field(default=1, description="Frame bundle number.")
    number_of_view: int = Field(default=1, description="Number of views.")
    internal_buffer_size: int = Field(default=16, description="Camera internal frame buffer size.")
    timeout: int | Literal["auto"] = Field(default="auto", description="DCAM wait timeout in ms.")
    capture_mode: DCAMCAP_START = Field(
        default=DCAMCAP_START.SEQUENCE, description="DCAM capture mode."
    )

    _api: DCAMAPI_INIT = PrivateAttr(default_factory=DCAMAPI_INIT)
    _device: DCAMDEV_OPEN = PrivateAttr(default_factory=DCAMDEV_OPEN)
    _internal_buffer: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    _frame: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    _wait: DCAMWAIT_OPEN = PrivateAttr(default_factory=DCAMWAIT_OPEN)
    _wait_start: DCAMWAIT_START = PrivateAttr(default_factory=DCAMWAIT_START)

    _timeout: int = PrivateAttr(default=2)
    _software_trigger_cycle: int = PrivateAttr(default=0)
    _framecount_till_software_trigger: int = PrivateAttr(default=0)

    @staticmethod
    def get_available_devices() -> list[dict[str, str]]:
        _api = DCAMAPI_INIT()
        err = dcamapi_init(byref(_api))
        if err.is_failed():
            raise RuntimeError(
                f"{type(DCAM).__name__} Failed to initialize DCAM-API: {DCAMERR(err).name}"
            )

        devices = []

        for i in range(_api.iDeviceCount):
            _device = DCAMDEV_OPEN()
            _device.index = i

            err = dcamdev_open(byref(_device))
            if err.is_failed():
                LOGGER.error(
                    f"{type(DCAM).__name__} Failed to initialize DCAM %s: %s", i, DCAMERR(err).name
                )
                continue

            dcam_info = {}
            for idstr in DCAM_IDSTR:
                dev_str = DCAMDEV_STRING()
                dev_str.iString = idstr
                dev_str.alloctext(256)

                err = dcamdev_getstring(_device.hdcam, byref(dev_str))
                if err.is_failed():
                    LOGGER.error(
                        f"{type(DCAM).__name__} Failed to get device information for %s: %s",
                        idstr,
                        DCAMERR(err).name,
                    )
                    continue

                dcam_info[idstr.name] = dev_str.text.decode()

            devices.append(dcam_info)
            dcamdev_close(_device.hdcam)

        dcamapi_uninit()
        return devices

    def info(self) -> dict[str, str]:
        self._ensure_open()
        dcam_info = {}

        for idstr in DCAM_IDSTR:
            dev_str = DCAMDEV_STRING()
            dev_str.iString = idstr
            dev_str.alloctext(256)

            err = dcamdev_getstring(self._device.hdcam, byref(dev_str))
            if err.is_failed():
                raise RuntimeError(
                    f"{type(self).__name__} Failed to get device information "
                    f"for {idstr}: {DCAMERR(err).name}"
                )

            dcam_info[idstr.name] = dev_str.text.decode()

        return dcam_info

    def software_trigger(self) -> None:
        if self.trigger_source == DCAMPROP.TRIGGERSOURCE.SOFTWARE:
            if self._framecount_till_software_trigger > 0:
                self._framecount_till_software_trigger -= 1

            if self._framecount_till_software_trigger == 0:
                err = dcamcap_firetrigger(self._device.hdcam, c_int32(0))
                if err.is_failed():
                    raise RuntimeError(
                        f"{type(self).__name__} Failed to software trigger: {DCAMERR(err).name}"
                    )
                self._framecount_till_software_trigger = self._software_trigger_cycle

    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
        height = self._internal_buffer.height * self.framebundle_number * self.number_of_view
        return [
            (
                "data",
                self.DTYPE_MAP.get(self.pixel_type, "float32"),
                (height, self._internal_buffer.width),
            ),
        ]

    def _open(self) -> None:
        self._set_api()
        self._set_device()
        self._set_params()
        self._set_internal_buffer()
        self._set_buffer()
        self._set_timeout()
        self._trigger_policy()
        self._open_wait()

    def _close(self) -> None:
        self._close_wait()
        self._release_buffer()
        self._release_internal_buffer()
        self._release_device()
        self._release_api()

    def _get_property(
        self, prop: DCAM_IDPROP | int, errors: Literal["ignore", "raise", "log"] = "log"
    ) -> float | None:
        value = c_double()
        err = dcamprop_getvalue(self._device.hdcam, c_int32(int(prop)), byref(value))
        if err.is_failed():
            if errors == "ignore":
                return None

            try:
                prop_name = DCAM_IDPROP(int(prop)).name
            except ValueError:
                prop_name = f"0x{int(prop):08X}"

            msg = (
                f"{type(self).__name__} Failed to get property "
                f"{prop_name}[{int(prop)}] to {value}: {DCAMERR(err).name}[{err}]"
            )

            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)

            return None

        return value.value

    def _set_property(
        self,
        prop: DCAM_IDPROP | int,
        value: float | int | Enum,
        errors: Literal["ignore", "raise", "log"] = "log",
    ) -> None:
        if isinstance(value, Enum):
            value = value.value

        err = dcamprop_setvalue(self._device.hdcam, c_int32(int(prop)), c_double(float(value)))
        if err.is_failed() and errors != "ignore":
            try:
                prop_name = DCAM_IDPROP(int(prop)).name
            except ValueError:
                prop_name = f"0x{int(prop):08X}"

            msg = (
                f"{type(self).__name__} Failed to set property "
                f"{prop_name}[{int(prop)}] to {value}: {DCAMERR(err).name}[{err}]"
            )

            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)

    def _set_api(self) -> None:
        err = dcamapi_init(byref(self._api))
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to initialize DCAM-API: {DCAMERR(err).name}"
            )

    def _release_api(self) -> None:
        dcamapi_uninit()
        self._api = DCAMAPI_INIT()

    def _set_device(self) -> None:
        self._device.index = self.device_index
        err = dcamdev_open(byref(self._device))
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to initialize DCAM device: {DCAMERR(err).name}"
            )

    def _release_device(self) -> None:
        dcamdev_close(self._device.hdcam)
        self._device = DCAMDEV_OPEN()

    def _set_params(self) -> None:
        self._set_property(DCAM_IDPROP.IMAGE_PIXELTYPE, self.pixel_type)
        self._set_property(DCAM_IDPROP.SENSORMODE, self.sensor_mode)
        self._set_property(DCAM_IDPROP.SHUTTER_MODE, self.shutter_mode)
        self._set_property(DCAM_IDPROP.READOUTSPEED, self.readout_speed)
        self._set_property(DCAM_IDPROP.READOUT_DIRECTION, self.readout_direction)
        self._set_property(
            DCAM_IDPROP.TRIGGERSOURCE,
            DCAMPROP.TRIGGERSOURCE.MASTERPULSE
            if self.timing_mode == "masterpulse"
            else self.trigger_source,
            errors="raise",
        )
        self._set_property(DCAM_IDPROP.TRIGGER_MODE, self.trigger_mode)
        self._set_property(DCAM_IDPROP.TRIGGERACTIVE, self.trigger_active)
        self._set_property(DCAM_IDPROP.TRIGGERPOLARITY, self.trigger_polarity)

        if self.trigger_global_exposure is not None:
            self._set_property(DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE, self.trigger_global_exposure)

        self._set_binning_and_resolution_params()

        self._set_property(DCAM_IDPROP.EXPOSURETIME, self.exposure_time / 1000, errors="raise")

        self._set_timing_mode_params()

        self._set_property(DCAM_IDPROP.CONTRASTGAIN, self.contrast_gain, errors="ignore")
        self._set_property(DCAM_IDPROP.FRAMEBUNDLE_MODE, self.framebundle_mode)

        if self.framebundle_mode == DCAMPROP.MODE.ON:
            self._set_property(DCAM_IDPROP.FRAMEBUNDLE_NUMBER, self.framebundle_number)

        self._set_property(DCAM_IDPROP.NUMBEROF_VIEW, self.number_of_view)

        self._set_output_trigger_params()

    def _set_binning_and_resolution_params(self) -> None:
        self._set_property(DCAM_IDPROP.BINNING, self.binning, errors="raise")

        if self.resolution is None:
            if self.subarray_mode is not None:
                self._set_property(DCAM_IDPROP.SUBARRAYMODE, self.subarray_mode, errors="raise")

            return

        self._set_property(DCAM_IDPROP.SUBARRAYMODE, DCAMPROP.MODE.OFF, errors="ignore")

        width, height = self.resolution
        full_width = int(self._get_property(DCAM_IDPROP.IMAGE_WIDTH) or float("inf"))
        full_height = int(self._get_property(DCAM_IDPROP.IMAGE_HEIGHT) or float("inf"))

        if width > full_width or height > full_height:
            raise ValueError(
                f"{type(self).__name__} Requested DCAM resolution is larger than the current"
                f"full-frame image size: requested={width}x{height}, "
                f"full_frame={full_width}x{full_height}. "
                "Check binning and camera full-frame dimensions."
            )

        hpos = self.subarray_hpos
        vpos = self.subarray_vpos

        if hpos is None:
            hpos = max(0, (full_width - width) // 2) if self.center_roi else 0

        if vpos is None:
            vpos = max(0, (full_height - height) // 2) if self.center_roi else 0

        self._set_property(DCAM_IDPROP.SUBARRAYHPOS, hpos, errors="raise")
        self._set_property(DCAM_IDPROP.SUBARRAYHSIZE, width, errors="raise")
        self._set_property(DCAM_IDPROP.SUBARRAYVPOS, vpos, errors="raise")
        self._set_property(DCAM_IDPROP.SUBARRAYVSIZE, height, errors="raise")
        self._set_property(DCAM_IDPROP.SUBARRAYMODE, DCAMPROP.MODE.ON, errors="raise")

    def _set_timing_mode_params(self) -> None:
        if self.frame_rate is not None:
            requested_interval = 1.0 / self.frame_rate

            if self.timing_mode == "masterpulse":
                self._set_property(
                    DCAM_IDPROP.MASTERPULSE_MODE, self.masterpulse_mode, errors="raise"
                )
                self._set_property(
                    DCAM_IDPROP.MASTERPULSE_TRIGGERSOURCE, self.masterpulse_triggersource
                )

                min_trigger_interval = self._get_property(DCAM_IDPROP.TIMING_MINTRIGGERINTERVAL)
                masterpulse_interval = (
                    min_trigger_interval + 0.001
                    if min_trigger_interval and requested_interval <= min_trigger_interval
                    else requested_interval
                )

                self._set_property(
                    DCAM_IDPROP.MASTERPULSE_INTERVAL, masterpulse_interval, errors="raise"
                )
                self._set_property(DCAM_IDPROP.MASTERPULSE_BURSTTIMES, self.masterpulse_bursttimes)

            else:
                frame_rate_readback = self._set_property(
                    DCAM_IDPROP.INTERNALFRAMERATE, self.frame_rate
                )

                if frame_rate_readback is None:
                    self._set_property(DCAM_IDPROP.INTERNAL_FRAMEINTERVAL, requested_interval)

    def _set_output_trigger_params(self) -> None:
        output_offset = (self.output_trigger_connector - 1) * int(DCAM_IDPROP._OUTPUTTRIGGER)
        self._set_property(
            int(DCAM_IDPROP.OUTPUTTRIGGER_KIND) + output_offset,
            self.output_trigger_kind,
            errors="raise",
        )
        self._set_property(
            int(DCAM_IDPROP.OUTPUTTRIGGER_POLARITY) + output_offset,
            self.output_trigger_polarity,
            errors="raise",
        )

        if (
            self.output_trigger_kind
            in (
                DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
                DCAMPROP.OUTPUTTRIGGER_KIND.ANYROWEXPOSURE,
            )
            and self.output_trigger_basesensor is not None
        ):
            self._set_property(
                int(DCAM_IDPROP.OUTPUTTRIGGER_BASESENSOR) + output_offset,
                self.output_trigger_basesensor,
            )

        if self.output_trigger_kind == DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE:
            self._set_property(
                int(DCAM_IDPROP.OUTPUTTRIGGER_SOURCE) + output_offset,
                self.output_trigger_source,
                errors="raise",
            )
            self._set_property(
                int(DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE) + output_offset, self.output_trigger_active
            )
            self._set_property(
                int(DCAM_IDPROP.OUTPUTTRIGGER_DELAY) + output_offset,
                self.output_trigger_delay,
                errors="raise",
            )
            self._set_property(
                int(DCAM_IDPROP.OUTPUTTRIGGER_PERIOD) + output_offset,
                self.output_trigger_period,
                errors="raise",
            )

    def _set_internal_buffer(self) -> None:
        buffer_size = c_int32(self.internal_buffer_size)
        err = dcambuf_alloc(self._device.hdcam, buffer_size)
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to set device internal buffer: {DCAMERR(err).name}"
            )

        width = self._get_property(DCAM_IDPROP.IMAGE_WIDTH, errors="raise")
        height = self._get_property(DCAM_IDPROP.IMAGE_HEIGHT, errors="raise")
        rowbytes = self._get_property(
            DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
            if self.framebundle_mode == DCAMPROP.MODE.ON
            else DCAM_IDPROP.IMAGE_ROWBYTES,
            errors="raise",
        )
        self._internal_buffer.type = self.pixel_type
        self._internal_buffer.width = int(width or 0)
        self._internal_buffer.height = int(height or 0)
        self._internal_buffer.rowbytes = int(rowbytes or 0)

        self._frame.iFrame = -1
        self._frame.rowbytes = self._internal_buffer.rowbytes
        self._frame.type = self._internal_buffer.type
        self._frame.width = self._internal_buffer.width
        self._frame.height = self._internal_buffer.height

    def _release_internal_buffer(self) -> None:
        err = dcambuf_release(self._device.hdcam, c_int32(0))
        if err.is_failed():
            LOGGER.error(
                f"{type(self).__name__} Failed to release device internal buffer: "
                f"{DCAMERR(err).name}"
            )

    def _set_timeout(self) -> None:
        if self.timeout == "auto":
            interval_sec = 0.0

            if self.timing_mode == "masterpulse":
                try:
                    interval_sec = self._get_property(DCAM_IDPROP.MASTERPULSE_INTERVAL) or 0.0
                except Exception:
                    interval_sec = 0.0

            if interval_sec <= 0:
                try:
                    interval_sec = self._get_property(DCAM_IDPROP.INTERNAL_FRAMEINTERVAL) or 0.0
                except Exception:
                    interval_sec = 0.0

            exposure_sec = self.exposure_time / 1000.0

            self._timeout = max(
                self._timeout,
                int(max(interval_sec, exposure_sec) * 1000.0) + 500,
            )

    def _start_capture(self) -> None:
        err = dcamcap_start(self._device.hdcam, self.capture_mode)
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to start device capture: {DCAMERR(err).name}"
            )

    def _capture_status(self) -> DCAMCAP_STATUS:
        status = c_int32()
        err = dcamcap_status(self._device.hdcam, byref(status))
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to get device capture status: {DCAMERR(err).name}"
            )
        return DCAMCAP_STATUS(status.value)

    def _stop_capture(self) -> None:
        err = dcamcap_stop(self._device.hdcam)
        if err.is_failed():
            LOGGER.error(
                f"{type(self).__name__} Failed to stop device capture: {DCAMERR(err).name}"
            )

    def _trigger_policy(self) -> None:
        self._framecount_till_software_trigger = 0

        if self.trigger_mode == DCAMPROP.TRIGGER_MODE.START:
            self._software_trigger_cycle = 0
        elif self.trigger_mode == DCAMPROP.TRIGGER_MODE.PIV:
            self._software_trigger_cycle = 2
        else:
            self._software_trigger_cycle = 1

    def _open_wait(self) -> None:
        self._wait.hdcam = self._device.hdcam
        err = dcamwait_open(byref(self._wait))
        if err.is_failed():
            raise RuntimeError(
                f"{type(self).__name__} Failed to open dcam wait: {DCAMERR(err).name}"
            )
        if self._wait.hwait == 0:
            raise RuntimeError(
                f"{type(self).__name__} Failed to open dcam wait: {DCAMERR.INVALIDWAITHANDLE.name}"
            )

    def _close_wait(self) -> None:
        err = dcamwait_close(self._wait.hwait)
        if err.is_failed():
            LOGGER.error(f"{type(self).__name__} Failed to close dcam wait: {DCAMERR(err).name}")
        self._wait = DCAMWAIT_OPEN()

    def _abort_wait(self) -> None:
        err = dcamwait_abort(self._wait.hwait)
        if err.is_failed():
            LOGGER.error(f"{type(self).__name__} Failed to abort dcam wait: {DCAMERR(err).name}")

    def _wait_event(self, eventmask: DCAMWAIT_CAPEVENT, timeout: int) -> Any:
        self._wait_start.eventmask = eventmask
        self._wait_start.timeout = timeout

        err = dcamwait_start(self._wait.hwait, byref(self._wait_start))

        if err == DCAMERR.ABORT:
            return False

        if err.is_failed() and err != DCAMERR.TIMEOUT:
            raise RuntimeError(
                f"{type(self).__name__} Failed to start dcam wait event: {DCAMERR(err).name}"
            )

        if err == DCAMERR.TIMEOUT:
            LOGGER.warning(f"{type(self).__name__} Timeout waiting for frame ready event.")
            return False

        return self._wait_start.eventhappened

    def _acquire_sample(self) -> ndarray | None:
        if not self._wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY, self._timeout):
            return

        return self._device.hdcam.dcambuf_copyframe()
