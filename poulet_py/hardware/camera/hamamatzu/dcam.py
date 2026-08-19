try:
    from ctypes import byref, c_double, c_int32, c_void_p
    from enum import Enum
    from threading import Condition, Event, Thread
    from time import monotonic_ns
    from typing import Any, Literal

    from numpy import ndarray, zeros
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, AcquisitionType
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
        dcambuf_copyframe,
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


class DCAM(BaseModel):
    DTYPE_MAP: dict[DCAM_PIXELTYPE, str] = {
        DCAM_PIXELTYPE.MONO16: "uint16",
        DCAM_PIXELTYPE.MONO8: "uint8",
    }

    device_index: int = Field(default=0, description="Camera device index")

    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE,
        description="Type of data acquisition, continuous or finite.",
    )

    pixel_type: DCAM_PIXELTYPE = Field(
        default=DCAM_PIXELTYPE.MONO16,
        description="Camera pixel type.",
    )

    sensor_mode: DCAMPROP.SENSORMODE = Field(
        default=DCAMPROP.SENSORMODE.AREA,
        description="Camera sensor mode.",
    )

    shutter_mode: DCAMPROP.SHUTTER_MODE = Field(
        default=DCAMPROP.SHUTTER_MODE.GLOBAL,
        description="Camera shutter mode.",
    )

    readout_speed: DCAMPROP.READOUTSPEED = Field(
        default=DCAMPROP.READOUTSPEED.FASTEST,
        description="Camera readout speed.",
    )

    readout_direction: DCAMPROP.READOUT_DIRECTION = Field(
        default=DCAMPROP.READOUT_DIRECTION.FORWARD,
        description="Camera readout direction.",
    )

    trigger_source: DCAMPROP.TRIGGERSOURCE = Field(
        default=DCAMPROP.TRIGGERSOURCE.INTERNAL,
        description="Camera trigger source.",
    )

    trigger_mode: DCAMPROP.TRIGGER_MODE = Field(
        default=DCAMPROP.TRIGGER_MODE.NORMAL,
        description="Camera trigger mode.",
    )

    trigger_active: DCAMPROP.TRIGGERACTIVE = Field(
        default=DCAMPROP.TRIGGERACTIVE.EDGE,
        description="Camera trigger active mode.",
    )

    trigger_polarity: DCAMPROP.TRIGGERPOLARITY = Field(
        default=DCAMPROP.TRIGGERPOLARITY.POSITIVE,
        description="Camera trigger polarity.",
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
        default=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
        description="Output trigger kind.",
    )

    output_trigger_source: DCAMPROP.OUTPUTTRIGGER_SOURCE = Field(
        default=DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
        description="Output trigger source for programmable output triggers.",
    )

    output_trigger_polarity: DCAMPROP.OUTPUTTRIGGER_POLARITY = Field(
        default=DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE,
        description="Output trigger polarity.",
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
        default=0.0,
        description="Output trigger delay in seconds.",
        ge=0,
    )

    output_trigger_period: float = Field(
        default=0.001,
        description="Output trigger period in seconds.",
        gt=0,
    )

    binning: DCAMPROP.BINNING = Field(
        default=DCAMPROP.BINNING._1,
        description="Camera-native binning mode.",
    )

    resolution: tuple[int, int] | None = Field(
        default=None,
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
        description="Exposure time in ms.",
        gt=1,
        lt=10000,
    )

    frame_rate: float | None = Field(
        default=10.0,
        description=(
            "Requested acquisition frame rate in frames per second. "
            "For timing_mode='masterpulse', this becomes MASTERPULSE_INTERVAL = 1/frame_rate."
        ),
        gt=0,
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
    buffer_size: int = Field(default=100, description="Software circular buffer size.")
    dcam_internal_buffer_size: int = Field(
        default=16, description="Camera internal frame buffer size."
    )
    timeout: int | Literal["auto"] = Field(default="auto", description="DCAM wait timeout in ms.")
    capture_mode: DCAMCAP_START = Field(
        default=DCAMCAP_START.SEQUENCE, description="DCAM capture mode."
    )

    _is_open: bool = PrivateAttr(default=False)
    _dcam_api: DCAMAPI_INIT = PrivateAttr(default_factory=DCAMAPI_INIT)
    _dcam_device: DCAMDEV_OPEN = PrivateAttr(default_factory=DCAMDEV_OPEN)
    _dcam_internal_buffer: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    _dcam_frame: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    _dcam_wait: DCAMWAIT_OPEN = PrivateAttr(default_factory=DCAMWAIT_OPEN)
    _dcam_wait_event: DCAMWAIT_START = PrivateAttr(default_factory=DCAMWAIT_START)
    _dcam_buffer: ndarray = PrivateAttr()
    _dcam_buffer_idx: int = PrivateAttr(0)
    _dcam_buffer_needle: int = PrivateAttr(0)
    _timeout: int = PrivateAttr(default=2)
    _software_trigger_cycle: int = PrivateAttr(default=0)
    _framecount_till_software_trigger: int = PrivateAttr(default=0)

    _dcam_acquisition_thread: Thread = PrivateAttr()
    _dcam_stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _dcam_acquisition_cond: Condition = PrivateAttr(default_factory=Condition)
    _timeout_errors: int = PrivateAttr(default=0)

    @staticmethod
    def get_available_devices() -> list[dict[str, str]]:
        _api = DCAMAPI_INIT()
        err = dcamapi_init(byref(_api))
        if err.is_failed():
            raise RuntimeError(f"Failed to initialize DCAM-API: {DCAMERR(err).name}")

        devices = []

        for i in range(_api.iDeviceCount):
            _device = DCAMDEV_OPEN()
            _device.index = i

            err = dcamdev_open(byref(_device))
            if err.is_failed():
                LOGGER.error("Failed to initialize DCAM %s: %s", i, DCAMERR(err).name)
                continue

            dcam_info = {}
            for idstr in DCAM_IDSTR:
                dev_str = DCAMDEV_STRING()
                dev_str.iString = idstr
                dev_str.alloctext(256)

                err = dcamdev_getstring(_device.hdcam, byref(dev_str))
                if err.is_failed():
                    LOGGER.error(
                        "Failed to get device information for %s: %s",
                        idstr,
                        DCAMERR(err).name,
                    )
                    continue

                dcam_info[idstr.name] = dev_str.text.decode()

            devices.append(dcam_info)
            dcamdev_close(_device.hdcam)

        dcamapi_uninit()
        return devices

    @property
    def is_open(self):
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return

        try:
            self._log_debug_settings(
                "Requested DCAM configuration before dcamapi_init",
                (
                    "device_index",
                    "pixel_type",
                    "sensor_mode",
                    "shutter_mode",
                    "readout_speed",
                    "readout_direction",
                    "binning",
                    "resolution",
                    "subarray_hpos",
                    "subarray_vpos",
                    "subarray_mode",
                    "center_roi",
                    "trigger_source",
                    "trigger_mode",
                    "trigger_active",
                    "trigger_polarity",
                    "trigger_global_exposure",
                    "exposure_time",
                    "frame_rate",
                    "timing_mode",
                    "masterpulse_mode",
                    "masterpulse_triggersource",
                    "masterpulse_bursttimes",
                    "capture_mode",
                    "output_trigger_connector",
                    "output_trigger_kind",
                    "output_trigger_source",
                    "output_trigger_polarity",
                    "output_trigger_active",
                    "output_trigger_basesensor",
                    "output_trigger_delay",
                    "output_trigger_period",
                ),
            )

            self._set_dcam_api()
            self._set_dcam_device()
            self._set_params()
            self._set_dcam_internal_buffer()
            self._set_buffer()
            self._set_timeout()
            self._trigger_policy()
            self._open_dcam_wait()

            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                self._start_capture()
                self._start_acquisition_thread()

            self._is_open = True

        except Exception as e:
            raise RuntimeError("Failed to open Dcam") from e

    def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            if not self._stop_acquisition_thread():
                # Releasing the wait handle, frame buffer or API while the
                # acquisition thread is still inside the SDK crashes the
                # process, so leak them instead.
                LOGGER.error(
                    "Keeping DCAM resources allocated because the acquisition "
                    "thread is still running; restart the process to reopen "
                    "the camera"
                )
                return

            self._stop_capture()

        self._close_dcam_wait()
        self._release_buffer()
        self._release_dcam_internal_buffer()
        self._release_dcam_device()
        self._release_dcam_api()

    def info(self) -> dict[str, str]:
        self._ensure_open()
        dcam_info = {}

        for idstr in DCAM_IDSTR:
            dev_str = DCAMDEV_STRING()
            dev_str.iString = idstr
            dev_str.alloctext(256)

            err = dcamdev_getstring(self._dcam_device.hdcam, byref(dev_str))
            if err.is_failed():
                raise RuntimeError(
                    f"Failed to get device information for {idstr}: {DCAMERR(err).name}"
                )

            dcam_info[idstr.name] = dev_str.text.decode()

        return dcam_info

    def read_sample(self, timeout: float = 0.01) -> ndarray | None:
        self._ensure_open()
        sample = None

        if self.acquisition_type == AcquisitionType.FINITE:
            self._start_capture()
            if not self._acquire_sample():
                return sample
            self._stop_capture()

        with self._dcam_acquisition_cond:
            if self._dcam_buffer_needle == self._dcam_buffer_idx:
                self._dcam_acquisition_cond.wait(timeout)

            idx = (self._dcam_buffer_idx - 1) % self.buffer_size
            sample = self._dcam_buffer[idx]
            self._dcam_buffer_needle = self._dcam_buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        self._ensure_open()

        if data.shape[0] < n:
            raise ValueError(f"Provided array has {data.shape[0]} rows, need at least {n}")

        deadline = monotonic_ns() + int(timeout * 1e9) if timeout >= 0 else None

        if self.acquisition_type == AcquisitionType.FINITE:
            self._start_capture()

            if n < 0 and timeout < 0:
                raise ValueError("For finite acquisition, either n or timeout must be specified.")

            acquired = 0
            while (
                (n < 0 or acquired < n)
                and (deadline is None or monotonic_ns() < deadline)
                and self._acquire_sample()
            ):
                acquired += 1

            self._stop_capture()

        elif self.acquisition_type == AcquisitionType.CONTINUOUS:
            with self._dcam_acquisition_cond:
                if n == -1 and deadline is None:
                    pass

                elif n == -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    self._dcam_acquisition_cond.wait(remaining)

                elif n != -1 and deadline is None:
                    while self._dcam_buffer_idx - self._dcam_buffer_needle < n:
                        self._dcam_acquisition_cond.wait()

                elif n != -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    while self._dcam_buffer_idx - self._dcam_buffer_needle < n and remaining > 0:
                        self._dcam_acquisition_cond.wait(remaining)
                        remaining = (deadline - monotonic_ns()) / 1e9

        with self._dcam_acquisition_cond:
            avail = self._dcam_buffer_idx - self._dcam_buffer_needle

            if avail <= 0:
                return 0

            count = avail if n < 0 else min(avail, n)
            size = self.buffer_size
            buffer = self._dcam_buffer
            needle = self._dcam_buffer_needle

            if count > size:
                needle = self._dcam_buffer_idx - size
                count = size

            start = needle % size
            end = start + count

            if end <= size:
                data[:count] = buffer[start:end]
            else:
                first = size - start
                data[:first] = buffer[start:]
                data[first:count] = buffer[: count - first]

            self._dcam_buffer_needle = needle + count

        return count

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("DCAM is not open")

    def _set_dcam_api(self) -> None:
        err = dcamapi_init(byref(self._dcam_api))
        if err.is_failed():
            raise RuntimeError(f"Failed to initialize DCAM-API: {DCAMERR(err).name}")

    def _release_dcam_api(self) -> None:
        dcamapi_uninit()
        self._dcam_api = DCAMAPI_INIT()

    def _set_dcam_device(self) -> None:
        self._dcam_device.index = self.device_index
        err = dcamdev_open(byref(self._dcam_device))
        if err.is_failed():
            raise RuntimeError(f"Failed to initialize DCAM device: {DCAMERR(err).name}")

    def _release_dcam_device(self) -> None:
        dcamdev_close(self._dcam_device.hdcam)
        self._dcam_device = DCAMDEV_OPEN()

    def _get_property(
        self, prop: DCAM_IDPROP | int, errors: Literal["ignore", "raise", "log"] = "log"
    ) -> float | None:
        value = c_double()
        err = dcamprop_getvalue(self._dcam_device.hdcam, c_int32(int(prop)), byref(value))
        if err.is_failed():
            if errors == "ignore":
                return None

            try:
                prop_name = DCAM_IDPROP(int(prop)).name
            except ValueError:
                prop_name = f"0x{int(prop):08X}"

            msg = (
                f"Failed to get property {prop_name}[{int(prop)}] to {value}: "
                f"{DCAMERR(err).name}[{err}]"
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

        err = dcamprop_setvalue(self._dcam_device.hdcam, c_int32(int(prop)), c_double(float(value)))
        if err.is_failed() and errors != "ignore":
            try:
                prop_name = DCAM_IDPROP(int(prop)).name
            except ValueError:
                prop_name = f"0x{int(prop):08X}"

            msg = (
                f"Failed to set property {prop_name}[{int(prop)}] to {value}: "
                f"{DCAMERR(err).name}[{err}]"
            )

            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)

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

        width, height = self.resolution
        # TODO move to validation
        if width <= 0 or height <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")

        self._set_property(DCAM_IDPROP.SUBARRAYMODE, DCAMPROP.MODE.OFF, errors="ignore")

        full_width = int(self._get_property(DCAM_IDPROP.IMAGE_WIDTH) or float("inf"))
        full_height = int(self._get_property(DCAM_IDPROP.IMAGE_HEIGHT) or float("inf"))

        if width > full_width or height > full_height:
            raise ValueError(
                "Requested DCAM resolution is larger than the current full-frame image size: "
                f"requested={width}x{height}, full_frame={full_width}x{full_height}. "
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

    def _set_timing_mode_params(self):
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

    def _set_dcam_internal_buffer(self) -> None:
        buffer_size = c_int32(self.dcam_internal_buffer_size)
        err = dcambuf_alloc(self._dcam_device.hdcam, buffer_size)
        if err.is_failed():
            raise RuntimeError(f"Failed to set device internal buffer: {DCAMERR(err).name}")

        width = self._get_property(DCAM_IDPROP.IMAGE_WIDTH, errors="raise")
        height = self._get_property(DCAM_IDPROP.IMAGE_HEIGHT, errors="raise")
        rowbytes = self._get_property(
            DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
            if self.framebundle_mode == DCAMPROP.MODE.ON
            else DCAM_IDPROP.IMAGE_ROWBYTES,
            errors="raise",
        )
        self._dcam_internal_buffer.type = self.pixel_type
        self._dcam_internal_buffer.width = int(width or 0)
        self._dcam_internal_buffer.height = int(height or 0)
        self._dcam_internal_buffer.rowbytes = int(rowbytes or 0)

        self._dcam_frame.iFrame = -1
        self._dcam_frame.rowbytes = self._dcam_internal_buffer.rowbytes
        self._dcam_frame.type = self._dcam_internal_buffer.type
        self._dcam_frame.width = self._dcam_internal_buffer.width
        self._dcam_frame.height = self._dcam_internal_buffer.height

    def _release_dcam_internal_buffer(self) -> None:
        err = dcambuf_release(self._dcam_device.hdcam, c_int32(0))
        if err.is_failed():
            LOGGER.error("Failed to release device internal buffer: %s", DCAMERR(err).name)

    def _set_buffer(self) -> None:
        height = self._dcam_internal_buffer.height * self.framebundle_number * self.number_of_view

        self._dcam_buffer = zeros(
            self.buffer_size,
            dtype=[
                ("timestamp", "uint64"),
                (
                    "dcam",
                    self.DTYPE_MAP.get(self.pixel_type, "float32"),
                    (height, self._dcam_internal_buffer.width),
                ),
            ],
        )

    def _release_buffer(self) -> None:
        del self._dcam_buffer
        self._dcam_buffer_idx = 0

    def _start_acquisition_thread(self) -> None:
        self._dcam_stop_acquisition_event.clear()
        self._dcam_acquisition_thread = Thread(
            target=self._dcam_acquisition_thread_func,
            name="DCAM Acquisition Thread",
            daemon=True,
        )
        self._dcam_acquisition_thread.start()

    def _stop_acquisition_thread(self) -> bool:
        """Stop the acquisition thread, reporting whether it actually exited."""
        self._dcam_stop_acquisition_event.set()
        self._abort_dcam_wait()
        self._dcam_acquisition_thread.join(timeout=self._timeout / 1000.0 + 5.0)

        if self._dcam_acquisition_thread.is_alive():
            LOGGER.error("Streaming thread did not stop gracefully")

        del self._dcam_acquisition_thread
        self._dcam_stop_acquisition_event.clear()
        return True

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

        if self.debug_output:
            LOGGER.info("DCAM FRAME_WAIT_TIMEOUT_MS: %s", self._timeout)

    def _start_capture(self) -> None:
        err = dcamcap_start(self._dcam_device.hdcam, self.capture_mode)
        if err.is_failed():
            raise RuntimeError(f"Failed to start device capture: {DCAMERR(err).name}")

        self._software_trigger()

        if self.debug_output:
            output_offset = (self.output_trigger_connector - 1) * int(DCAM_IDPROP._OUTPUTTRIGGER)
            capture_start_readbacks: dict[str, Any] = {}

            for name, prop in (
                ("OUTPUTTRIGGER_KIND", int(DCAM_IDPROP.OUTPUTTRIGGER_KIND) + output_offset),
                ("OUTPUTTRIGGER_POLARITY", int(DCAM_IDPROP.OUTPUTTRIGGER_POLARITY) + output_offset),
                ("TRIGGER_GLOBALEXPOSURE", DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE),
                ("TIMING_GLOBALEXPOSUREDELAY", DCAM_IDPROP.TIMING_GLOBALEXPOSUREDELAY),
                ("TIMING_INVALIDEXPOSUREPERIOD", DCAM_IDPROP.TIMING_INVALIDEXPOSUREPERIOD),
                ("EXPOSURETIME", DCAM_IDPROP.EXPOSURETIME),
                ("INTERNAL_FRAMEINTERVAL", DCAM_IDPROP.INTERNAL_FRAMEINTERVAL),
            ):
                try:
                    capture_start_readbacks[name] = self._get_property(prop)
                except Exception as exc:
                    capture_start_readbacks[name] = f"unavailable: {exc}"

            self._log_debug_readbacks(
                "Capture-start timing/output readback",
                capture_start_readbacks,
            )

    def _capture_status(self) -> DCAMCAP_STATUS:
        status = c_int32()
        err = dcamcap_status(self._dcam_device.hdcam, byref(status))
        if err.is_failed():
            raise RuntimeError(f"Failed to get device capture status: {DCAMERR(err).name}")
        return DCAMCAP_STATUS(status.value)

    def _stop_capture(self) -> None:
        err = dcamcap_stop(self._dcam_device.hdcam)
        if err.is_failed():
            LOGGER.error("Failed to stop device capture: %s", DCAMERR(err).name)

    def _trigger_policy(self) -> None:
        self._framecount_till_software_trigger = 0

        if self.trigger_mode == DCAMPROP.TRIGGER_MODE.START:
            self._software_trigger_cycle = 0
        elif self.trigger_mode == DCAMPROP.TRIGGER_MODE.PIV:
            self._software_trigger_cycle = 2
        else:
            self._software_trigger_cycle = 1

    def _software_trigger(self) -> None:
        if self.trigger_source == DCAMPROP.TRIGGERSOURCE.SOFTWARE:
            if self._framecount_till_software_trigger > 0:
                self._framecount_till_software_trigger -= 1

            if self._framecount_till_software_trigger == 0:
                err = dcamcap_firetrigger(self._dcam_device.hdcam, c_int32(0))
                if err.is_failed():
                    raise RuntimeError(f"Failed to software trigger: {DCAMERR(err).name}")
                self._framecount_till_software_trigger = self._software_trigger_cycle

    def _open_dcam_wait(self) -> None:
        self._dcam_wait.hdcam = self._dcam_device.hdcam
        err = dcamwait_open(byref(self._dcam_wait))
        if err.is_failed():
            raise RuntimeError(f"Failed to open dcam wait: {DCAMERR(err).name}")
        if self._dcam_wait.hwait == 0:
            raise RuntimeError(f"Failed to open dcam wait: {DCAMERR.INVALIDWAITHANDLE.name}")

    def _close_dcam_wait(self) -> None:
        err = dcamwait_close(self._dcam_wait.hwait)
        if err.is_failed():
            LOGGER.error("Failed to close dcam wait: %s", DCAMERR(err).name)
        self._dcam_wait = DCAMWAIT_OPEN()

    def _abort_dcam_wait(self) -> None:
        err = dcamwait_abort(self._dcam_wait.hwait)
        if err.is_failed():
            LOGGER.error("Failed to abort dcam wait: %s", DCAMERR(err).name)

    def _wait_event(self, eventmask: DCAMWAIT_CAPEVENT, timeout: int):
        self._dcam_wait_event.eventmask = eventmask
        self._dcam_wait_event.timeout = timeout

        err = dcamwait_start(self._dcam_wait.hwait, byref(self._dcam_wait_event))

        if err == DCAMERR.ABORT:
            return False

        if err.is_failed() and err != DCAMERR.TIMEOUT:
            raise RuntimeError(f"Failed to start dcam wait event: {DCAMERR(err).name}")

        if err == DCAMERR.TIMEOUT:
            self._timeout_errors += 1
            LOGGER.warning(
                "Timeout waiting for frame ready event. Timeout errors: %s", self._timeout_errors
            )
            return False

        self._timeout_errors = 0
        return self._dcam_wait_event.eventhappened

    def _dcam_frames_to_buffer(self) -> None:
        with self._dcam_acquisition_cond:
            idx = self._dcam_buffer_idx % self.buffer_size
            ptr = self._dcam_buffer[idx]["dcam"].ctypes.data
            self._dcam_frame.buf = c_void_p(ptr)
            err = dcambuf_copyframe(self._dcam_device.hdcam, byref(self._dcam_frame))
            if err.is_failed():
                raise RuntimeError(f"Failed to copy data: {DCAMERR(err).name}")

            self._dcam_buffer[idx]["timestamp"] = monotonic_ns()
            self._dcam_buffer_idx += 1
            self._dcam_acquisition_cond.notify_all()

    def _acquire_sample(self) -> bool:
        if not self._wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY, self._timeout):
            return False

        self._dcam_frames_to_buffer()
        self._software_trigger()
        return True

    def _dcam_acquisition_thread_func(self) -> None:
        while not self._dcam_stop_acquisition_event.is_set():
            try:
                self._acquire_sample()
            except Exception as e:
                self._dcam_stop_acquisition_event.set()
                LOGGER.exception("DCAM acquisition thread stopped after acquisition error")
                raise e

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
