"""Drop-in DCAM class for poulet_py Hamamatsu ORCA Flash 4.0 V3.

Adds camera-native 10 fps acquisition via Master Pulse and global-exposure
output on the timing connector. Designed to replace the existing DCAM class
module in poulet_py while preserving the original public read_sample/read_many_sample
workflow.
"""
from ctypes import byref, c_double, c_int32, c_void_p
from enum import Enum
from threading import Condition, Event, Thread
from time import monotonic_ns
from typing import Literal

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
    dcamwait_close,
    dcamwait_open,
    dcamwait_start,
)


class DCAM(BaseModel):
    DTYPE_MAP: dict[DCAM_PIXELTYPE, str] = {
        DCAM_PIXELTYPE.MONO16: "uint16",
        DCAM_PIXELTYPE.MONO8: "uint8",
    }

    device_index: int = Field(default=0, description="")
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )
    pixel_type: DCAM_PIXELTYPE = Field(
        default=DCAM_PIXELTYPE.MONO16, description="The pixel type of the camera."
    )
    sensor_mode: DCAMPROP.SENSORMODE = Field(default=DCAMPROP.SENSORMODE.AREA, description="")
    shutter_mode: DCAMPROP.SHUTTER_MODE = Field(
        default=DCAMPROP.SHUTTER_MODE.GLOBAL, description=""
    )
    readout_speed: DCAMPROP.READOUTSPEED = Field(
        default=DCAMPROP.READOUTSPEED.FASTEST, description=""
    )
    readout_direction: DCAMPROP.READOUT_DIRECTION = Field(
        default=DCAMPROP.READOUT_DIRECTION.FORWARD, description=""
    )
    trigger_source: DCAMPROP.TRIGGERSOURCE = Field(
        default=DCAMPROP.TRIGGERSOURCE.INTERNAL, description="The trigger source of the camera."
    )
    trigger_mode: DCAMPROP.TRIGGER_MODE = Field(
        default=DCAMPROP.TRIGGER_MODE.NORMAL, description="The trigger mode of the camera."
    )
    trigger_active: DCAMPROP.TRIGGERACTIVE = Field(
        default=DCAMPROP.TRIGGERACTIVE.EDGE, description=""
    )
    trigger_polarity: DCAMPROP.TRIGGERPOLARITY = Field(
        default=DCAMPROP.TRIGGERPOLARITY.POSITIVE, description=""
    )
    trigger_global_exposure: DCAMPROP.TRIGGER_GLOBALEXPOSURE | None = Field(
        default=DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED,
        description=(
            "Global-exposure timing mode. DELAYED is usually appropriate for rolling-shutter "
            "CMOS cameras; set None to leave the camera default unchanged."
        ),
    )

    # Output trigger / rear timing connector configuration.
    # For first electrical testing, HIGH should hold the selected output line high.
    # For LED gating during exposure, use OUTPUTTRIGGER_KIND.GLOBALEXPOSURE.
    # For a short pulse per frame, use OUTPUTTRIGGER_KIND.PROGRAMABLE with SOURCE=VSYNC.
    output_trigger_connector: int = Field(default=1, ge=1, le=16, description="")
    output_trigger_kind: DCAMPROP.OUTPUTTRIGGER_KIND = Field(
        default=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE, description=""
    )
    output_trigger_source: DCAMPROP.OUTPUTTRIGGER_SOURCE = Field(
        default=DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC, description=""
    )
    output_trigger_polarity: DCAMPROP.OUTPUTTRIGGER_POLARITY = Field(
        default=DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE, description=""
    )
    output_trigger_active: DCAMPROP.OUTPUTTRIGGER_ACTIVE = Field(
        default=DCAMPROP.OUTPUTTRIGGER_ACTIVE.EDGE, description=""
    )
    output_trigger_basesensor: DCAMPROP.OUTPUTTRIGGER_BASESENSOR | None = Field(
        default=DCAMPROP.OUTPUTTRIGGER_BASESENSOR.VIEW1,
        description="Base sensor/view for global-exposure output trigger. Usually VIEW1 for single-view capture.",
    )
    output_trigger_delay: float = Field(default=0.0, description="in seconds", ge=0)
    output_trigger_period: float = Field(default=0.001, description="in seconds", gt=0)
    debug_output: bool = Field(default=False, description="Print requested/applied DCAM settings.")

    # TODO check
    binning: DCAMPROP.BINNING = Field(default=DCAMPROP.BINNING._1, description="")
    # subarray_mode: TODO
    exposure_time: int = Field(default=50, description="Exposure time in ms", gt=1, lt=10000)
    frame_rate: float | None = Field(
        default=10.0,
        description="Requested acquisition frame rate in frames per second. For timing_mode='masterpulse', this becomes MASTERPULSE_INTERVAL = 1/frame_rate.",
        gt=0,
    )
    timing_mode: Literal["internal", "masterpulse"] = Field(
        default="masterpulse",
        description="Use 'masterpulse' for camera-native fixed-rate acquisition when INTERNALFRAMERATE is not writable.",
    )
    masterpulse_mode: DCAMPROP.MASTERPULSE_MODE = Field(
        default=DCAMPROP.MASTERPULSE_MODE.CONTINUOUS,
        description="Master pulse mode. CONTINUOUS gives a continuous fixed-rate acquisition clock.",
    )
    masterpulse_triggersource: DCAMPROP.MASTERPULSE_TRIGGERSOURCE = Field(
        default=DCAMPROP.MASTERPULSE_TRIGGERSOURCE.SOFTWARE,
        description="Master pulse trigger source. In CONTINUOUS mode this is normally not used, but is kept explicit.",
    )
    masterpulse_bursttimes: int = Field(default=1, description="Only relevant for MASTERPULSE_MODE.BURST.", ge=1)
    contrast_gain: int = Field(default=10, description="in ms")
    framebundle_mode: DCAMPROP.MODE = Field(default=DCAMPROP.MODE.OFF, description="")
    framebundle_number: int = Field(default=1, description="")
    number_of_view: int = Field(default=1, description="")

    buffer_size: int = Field(default=100, description="")
    dcam_internal_buffer_size: int = Field(default=10, description="")
    timeout: int | Literal["auto"] = Field(default="auto", description="handle timeout in ms")
    capture_mode: DCAMCAP_START = Field(default=DCAMCAP_START.SEQUENCE, description="")

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
    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _acquisition_cond: Condition = PrivateAttr(default_factory=Condition)

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
                LOGGER.error(f"Failed to initialize DCAM {i}: {DCAMERR(err).name}")
                continue

            dcam_info = {}
            for idstr in DCAM_IDSTR:
                dev_str = DCAMDEV_STRING()
                dev_str.iString = idstr
                dev_str.alloctext(256)

                err = dcamdev_getstring(_device.hdcam, byref(dev_str))
                if err.is_failed():
                    LOGGER.error(
                        f"Failed to get device information for {idstr}: {DCAMERR(err).name}"
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
            if self.debug_output:
                print("\n--- Requested DCAM configuration before dcamapi_init ---", flush=True)
                for name in (
                    "device_index", "pixel_type", "sensor_mode", "shutter_mode",
                    "readout_speed", "readout_direction", "trigger_source",
                    "trigger_mode", "trigger_active", "trigger_polarity",
                    "trigger_global_exposure", "exposure_time", "frame_rate", "timing_mode",
                    "masterpulse_mode", "masterpulse_triggersource", "masterpulse_bursttimes",
                    "capture_mode", "output_trigger_connector", "output_trigger_kind", "output_trigger_source",
                    "output_trigger_polarity", "output_trigger_active", "output_trigger_basesensor",
                    "output_trigger_delay", "output_trigger_period",
                ):
                    value = getattr(self, name)
                    print(f"{name}: {value.name if isinstance(value, Enum) else value}", flush=True)
                print("--- end requested configuration ---\n", flush=True)

            self._set_dcam_api()
            self._set_dcam_device()
            self._set_params()
            self._set_dcam_internal_buffer()
            self._set_buffer()

            self._set_timeout()
            self._trigger_policy()
            self._open_dcam_wait()
            self._start_capture()
            self._software_trigger()

            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                self._start_acquisition_thread()
            self._is_open = True
        except Exception as e:
            raise RuntimeError("Failed to open Dcam") from e

    def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._stop_acquisition_thread()

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

    def read_sample(self) -> ndarray | None:
        self._ensure_open()
        sample = None

        if self.acquisition_type == AcquisitionType.FINITE and not self._acquire_sample():
            return sample

        with self._acquisition_cond:
            idx = (self._dcam_buffer_idx - 1) % self.buffer_size
            sample = self._dcam_buffer[idx]
            self._dcam_buffer_needle = self._dcam_buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        self._ensure_open()
        deadline = monotonic_ns() + int(timeout * 1e9) if timeout >= 0 else None

        if self.acquisition_type == AcquisitionType.FINITE:
            if n < 0 and timeout < 0:
                raise ValueError("For finite acquisition, either n or timeout must be specified.")

            acquired = 0
            while (
                (n < 0 or acquired < n)
                and (deadline is None or monotonic_ns() < deadline)
                and self._acquire_sample()
            ):
                acquired += 1

        elif self.acquisition_type == AcquisitionType.CONTINUOUS:
            with self._acquisition_cond:
                if n == -1 and deadline is None:
                    pass
                elif n == -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    self._acquisition_cond.wait(remaining)
                elif n != -1 and deadline is None:
                    while self._dcam_buffer_idx - self._dcam_buffer_needle < n:
                        self._acquisition_cond.wait()
                elif n != -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    while self._dcam_buffer_idx - self._dcam_buffer_needle < n and remaining > 0:
                        self._acquisition_cond.wait(remaining)
                        remaining = (deadline - monotonic_ns()) / 1e9

        with self._acquisition_cond:
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

    def _get_property(self, prop: DCAM_IDPROP | int) -> float:
        value = c_double()
        err = dcamprop_getvalue(self._dcam_device.hdcam, c_int32(int(prop)), byref(value))
        if err.is_failed():
            raise RuntimeError(f"Failed to get property {prop}: {DCAMERR(err).name}")

        return value.value

    def _set_property(
        self,
        prop: DCAM_IDPROP | int,
        value: float | int | Enum,
        errors: Literal["ignore", "raise", "log"] = "log",
    ) -> float | None:
        if isinstance(value, Enum):
            value = value.value

        err = dcamprop_setvalue(
            self._dcam_device.hdcam,
            c_int32(int(prop)),
            c_double(float(value)),
        )
        if err.is_failed():
            try:
                prop_name = DCAM_IDPROP(int(prop)).name
            except ValueError:
                prop_name = f"0x{int(prop):08X}"

            msg = f"Failed to set property {prop_name}[{int(prop)}] to {value}: {DCAMERR(err).name}[{err}]"
            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)
            return None

        readback = c_double()
        err = dcamprop_getvalue(self._dcam_device.hdcam, c_int32(int(prop)), byref(readback))
        if err.is_failed():
            msg = f"Set property {prop}[{int(prop)}] to {value}, but readback failed: {DCAMERR(err).name}[{err}]"
            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)
            return None

        return readback.value

    def _set_params(self) -> None:
        self._set_property(DCAM_IDPROP.IMAGE_PIXELTYPE, self.pixel_type)
        self._set_property(DCAM_IDPROP.SENSORMODE, self.sensor_mode)
        self._set_property(DCAM_IDPROP.SHUTTER_MODE, self.shutter_mode)
        self._set_property(DCAM_IDPROP.READOUTSPEED, self.readout_speed)
        self._set_property(DCAM_IDPROP.READOUT_DIRECTION, self.readout_direction)

        # For fixed-rate acquisition on this ORCA model, INTERNALFRAMERATE/INTERNAL_FRAMEINTERVAL
        # may be read-only. In timing_mode='masterpulse', use the camera's Master Pulse engine
        # as the acquisition trigger source and set MASTERPULSE_INTERVAL below.
        effective_trigger_source = (
            DCAMPROP.TRIGGERSOURCE.MASTERPULSE
            if self.timing_mode == "masterpulse"
            else self.trigger_source
        )
        self._set_property(DCAM_IDPROP.TRIGGERSOURCE, effective_trigger_source, errors="raise")
        self._set_property(DCAM_IDPROP.TRIGGER_MODE, self.trigger_mode)
        self._set_property(DCAM_IDPROP.TRIGGERACTIVE, self.trigger_active)
        self._set_property(DCAM_IDPROP.TRIGGERPOLARITY, self.trigger_polarity)
        if self.trigger_global_exposure is not None:
            self._set_property(
                DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,
                self.trigger_global_exposure,
                errors="ignore",
            )
        self._set_property(DCAM_IDPROP.BINNING, self.binning)

        timing_readbacks = {
            "REQUESTED_EXPOSURE_MS": self.exposure_time,
            "EXPOSURETIME_SET_SEC": self._set_property(
                DCAM_IDPROP.EXPOSURETIME, self.exposure_time / 1000, errors="raise"
            ),
        }

        # Frame-rate control.
        # INTERNALFRAMERATE/INTERNAL_FRAMEINTERVAL were observed to be NOTWRITABLE on this setup.
        # Therefore timing_mode='masterpulse' uses MASTERPULSE_INTERVAL = 1 / frame_rate.
        if self.frame_rate is not None:
            timing_readbacks["REQUESTED_FRAMERATE_FPS"] = self.frame_rate
            requested_interval = 1.0 / self.frame_rate
            timing_readbacks["REQUESTED_FRAME_INTERVAL_SEC"] = requested_interval
            timing_readbacks["REQUESTED_FRAME_INTERVAL_MS"] = requested_interval * 1000.0

            if self.timing_mode == "masterpulse":
                min_trigger_interval = None
                try:
                    min_trigger_interval = self._get_property(DCAM_IDPROP.TIMING_MINTRIGGERINTERVAL)
                    timing_readbacks["TIMING_MINTRIGGERINTERVAL_BEFORE_MASTERPULSE_SEC"] = min_trigger_interval
                    timing_readbacks["TIMING_MINTRIGGERINTERVAL_BEFORE_MASTERPULSE_MS"] = min_trigger_interval * 1000.0
                except Exception as exc:
                    timing_readbacks["TIMING_MINTRIGGERINTERVAL_BEFORE_MASTERPULSE_SEC"] = f"unavailable: {exc}"

                masterpulse_interval = requested_interval
                if isinstance(min_trigger_interval, float) and masterpulse_interval <= min_trigger_interval:
                    # Leave a small margin rather than requesting an interval the camera cannot obey.
                    masterpulse_interval = min_trigger_interval + 0.001
                    timing_readbacks["MASTERPULSE_INTERVAL_ADJUSTED_REASON"] = (
                        "requested interval was <= TIMING_MINTRIGGERINTERVAL; added 1 ms margin"
                    )

                timing_readbacks["MASTERPULSE_INTERVAL_REQUESTED_SEC"] = masterpulse_interval
                timing_readbacks["MASTERPULSE_INTERVAL_REQUESTED_MS"] = masterpulse_interval * 1000.0
                timing_readbacks["MASTERPULSE_EFFECTIVE_REQUESTED_FPS"] = 1.0 / masterpulse_interval

                timing_readbacks["MASTERPULSE_MODE_SET"] = self._set_property(
                    DCAM_IDPROP.MASTERPULSE_MODE,
                    self.masterpulse_mode,
                    errors="raise",
                )
                timing_readbacks["MASTERPULSE_TRIGGERSOURCE_SET"] = self._set_property(
                    DCAM_IDPROP.MASTERPULSE_TRIGGERSOURCE,
                    self.masterpulse_triggersource,
                    errors="ignore",
                )
                timing_readbacks["MASTERPULSE_INTERVAL_SET_SEC"] = self._set_property(
                    DCAM_IDPROP.MASTERPULSE_INTERVAL,
                    masterpulse_interval,
                    errors="raise",
                )
                timing_readbacks["MASTERPULSE_BURSTTIMES_SET"] = self._set_property(
                    DCAM_IDPROP.MASTERPULSE_BURSTTIMES,
                    self.masterpulse_bursttimes,
                    errors="ignore",
                )
            else:
                frame_rate_readback = self._set_property(
                    DCAM_IDPROP.INTERNALFRAMERATE,
                    self.frame_rate,
                    errors="ignore",
                )
                if frame_rate_readback is None:
                    timing_readbacks["INTERNALFRAMERATE_SET"] = "not writable; trying INTERNAL_FRAMEINTERVAL"
                    frame_interval_readback = self._set_property(
                        DCAM_IDPROP.INTERNAL_FRAMEINTERVAL,
                        requested_interval,
                        errors="ignore",
                    )
                    if frame_interval_readback is None:
                        timing_readbacks["INTERNAL_FRAMEINTERVAL_SET"] = "not writable; leaving frame timing unchanged"
                    else:
                        timing_readbacks["INTERNAL_FRAMEINTERVAL_SET_SEC"] = frame_interval_readback
                else:
                    timing_readbacks["INTERNALFRAMERATE_SET_FPS"] = frame_rate_readback

        # Read current/applied timing values back from the camera. These are printed
        # before live capture starts, so you can verify the actual accepted timing.
        current_timing_props = (
            ("EXPOSURETIME_READBACK_SEC", DCAM_IDPROP.EXPOSURETIME),
            ("INTERNALFRAMERATE_READBACK_FPS", DCAM_IDPROP.INTERNALFRAMERATE),
            ("INTERNAL_FRAMEINTERVAL_READBACK_SEC", DCAM_IDPROP.INTERNAL_FRAMEINTERVAL),
            ("MASTERPULSE_MODE_READBACK", DCAM_IDPROP.MASTERPULSE_MODE),
            ("MASTERPULSE_TRIGGERSOURCE_READBACK", DCAM_IDPROP.MASTERPULSE_TRIGGERSOURCE),
            ("MASTERPULSE_INTERVAL_READBACK_SEC", DCAM_IDPROP.MASTERPULSE_INTERVAL),
            ("MASTERPULSE_BURSTTIMES_READBACK", DCAM_IDPROP.MASTERPULSE_BURSTTIMES),
            ("TIMING_READOUTTIME_SEC", DCAM_IDPROP.TIMING_READOUTTIME),
            ("TIMING_CYCLICTRIGGERPERIOD_SEC", DCAM_IDPROP.TIMING_CYCLICTRIGGERPERIOD),
            ("TIMING_MINTRIGGERBLANKING_SEC", DCAM_IDPROP.TIMING_MINTRIGGERBLANKING),
            ("TIMING_MINTRIGGERINTERVAL_SEC", DCAM_IDPROP.TIMING_MINTRIGGERINTERVAL),
            ("TIMING_EXPOSURE_MODE", DCAM_IDPROP.TIMING_EXPOSURE),
            ("TIMING_INVALIDEXPOSUREPERIOD_SEC", DCAM_IDPROP.TIMING_INVALIDEXPOSUREPERIOD),
            ("TIMING_GLOBALEXPOSUREDELAY_SEC", DCAM_IDPROP.TIMING_GLOBALEXPOSUREDELAY),
            ("TRIGGER_GLOBALEXPOSURE_READBACK", DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE),
        )
        for name, prop in current_timing_props:
            try:
                timing_readbacks[name] = self._get_property(prop)
            except Exception as exc:
                timing_readbacks[name] = f"unavailable: {exc}"

        exposure_sec = timing_readbacks.get("EXPOSURETIME_READBACK_SEC")
        if isinstance(exposure_sec, float):
            timing_readbacks["EXPOSURETIME_READBACK_MS"] = exposure_sec * 1000.0

        frame_interval_sec = timing_readbacks.get("INTERNAL_FRAMEINTERVAL_READBACK_SEC")
        if isinstance(frame_interval_sec, float) and frame_interval_sec > 0:
            timing_readbacks["EFFECTIVE_FPS_FROM_INTERNAL_FRAMEINTERVAL"] = 1.0 / frame_interval_sec
            timing_readbacks["INTERNAL_FRAMEINTERVAL_READBACK_MS"] = frame_interval_sec * 1000.0
import signal
from time import sleep

import cv2
import numpy as np

from poulet_py import DCAM
from poulet_py.hardware.camera.hamamatzu._api import DCAMPROP

NAME = "dcam_example"
running = True
from time import sleep

import cv2
import numpy as np

from poulet_py import DCAM
from poulet_py.hardware.camera.hamamatzu._api import DCAMPROP

NAME = "dcam_example"
running = True




def handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handler)


def show_framedata(data):
    maxval = np.amax(data)
    if data.dtype == np.uint16 and maxval > 0:
        imul = int(65535 / maxval)
        data = data * imul

    cv2.imshow(NAME, data)


devices = DCAM.get_available_devices()

if devices:
    print(devices)
else:
    raise RuntimeError("No Devices")

cv2.namedWindow(
    NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
)

with DCAM(
    device_index=0,
    debug_output=True,

    # First hardware test: the selected rear timing output should be held high.
    output_trigger_connector=1,
    output_trigger_kind=DCAMPROP.OUTPUTTRIGGER_KIND.HIGH,

    # For LED during exposure after the HIGH test works, use:
    # output_trigger_kind=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,

    # For a short frame-synchronous pulse after the HIGH test works, use:
    # output_trigger_kind=DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE,
    # output_trigger_source=DCAMPROP.OUTPUTTRIGGER_SOURCE.VSYNC,
    # output_trigger_period=0.001,
    # output_trigger_delay=0.0,
) as dcam:
    input("Camera is open and output trigger is configured. Measure the timing output now, then press Enter.")

    while running:
        sample = dcam.read_sample()
        if sample is None:
            continue

        show_framedata(sample["dcam"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        sleep(0.001)

cv2.destroyAllWindows()
        masterpulse_interval_sec = timing_readbacks.get("MASTERPULSE_INTERVAL_READBACK_SEC")
        if isinstance(masterpulse_interval_sec, float) and masterpulse_interval_sec > 0:
            timing_readbacks["MASTERPULSE_INTERVAL_READBACK_MS"] = masterpulse_interval_sec * 1000.0
            timing_readbacks["EFFECTIVE_FPS_FROM_MASTERPULSE_INTERVAL"] = 1.0 / masterpulse_interval_sec

        cyclic_period_sec = timing_readbacks.get("TIMING_CYCLICTRIGGERPERIOD_SEC")
        if isinstance(cyclic_period_sec, float) and cyclic_period_sec > 0:
            timing_readbacks["EFFECTIVE_FPS_FROM_CYCLICTRIGGERPERIOD"] = 1.0 / cyclic_period_sec
            timing_readbacks["TIMING_CYCLICTRIGGERPERIOD_MS"] = cyclic_period_sec * 1000.0

        min_trigger_interval_sec = timing_readbacks.get("TIMING_MINTRIGGERINTERVAL_SEC")
        if isinstance(min_trigger_interval_sec, float) and min_trigger_interval_sec > 0:
            timing_readbacks["MAX_FPS_FROM_MINTRIGGERINTERVAL"] = 1.0 / min_trigger_interval_sec
            timing_readbacks["TIMING_MINTRIGGERINTERVAL_MS"] = min_trigger_interval_sec * 1000.0

        readout_sec = timing_readbacks.get("TIMING_READOUTTIME_SEC")
        if isinstance(readout_sec, float):
            timing_readbacks["TIMING_READOUTTIME_MS"] = readout_sec * 1000.0

        global_exposure_delay_sec = timing_readbacks.get("TIMING_GLOBALEXPOSUREDELAY_SEC")
        if isinstance(global_exposure_delay_sec, float):
            timing_readbacks["TIMING_GLOBALEXPOSUREDELAY_MS"] = global_exposure_delay_sec * 1000.0

        invalid_exposure_period_sec = timing_readbacks.get("TIMING_INVALIDEXPOSUREPERIOD_SEC")
        if isinstance(invalid_exposure_period_sec, float):
            timing_readbacks["TIMING_INVALIDEXPOSUREPERIOD_MS"] = invalid_exposure_period_sec * 1000.0

        if isinstance(exposure_sec, float) and isinstance(global_exposure_delay_sec, float):
            # For rolling-shutter global-exposure timing, the useful global-exposure window
            # is often roughly exposure time minus the global-exposure delay. If this is <= 0,
            # OUTPUTTRIGGER_KIND.GLOBALEXPOSURE may read back correctly but produce no visible pulse.
            global_window_sec = exposure_sec - global_exposure_delay_sec
            timing_readbacks["EST_GLOBAL_EXPOSURE_WINDOW_SEC"] = global_window_sec
            timing_readbacks["EST_GLOBAL_EXPOSURE_WINDOW_MS"] = global_window_sec * 1000.0
            if global_window_sec <= 0:
                timing_readbacks["GLOBAL_EXPOSURE_WARNING"] = (
                    "estimated global-exposure window <= 0; increase exposure_time "
                    "or use PROGRAMABLE+EXPOSURE/VSYNC instead"
                )

        if self.debug_output:
            print("--- Current/applied acquisition timing before live capture ---", flush=True)
            for name, value in timing_readbacks.items():
                print(f"{name}: {value}", flush=True)
            print("--- end current/applied acquisition timing ---\n", flush=True)

        self._set_property(DCAM_IDPROP.CONTRASTGAIN, self.contrast_gain, errors="ignore")
        self._set_property(DCAM_IDPROP.FRAMEBUNDLE_MODE, self.framebundle_mode)

        if self.framebundle_mode == DCAMPROP.MODE.ON:
            self._set_property(DCAM_IDPROP.FRAMEBUNDLE_NUMBER, self.framebundle_number)
        self._set_property(DCAM_IDPROP.NUMBEROF_VIEW, self.number_of_view)

        output_offset = (self.output_trigger_connector - 1) * int(DCAM_IDPROP._OUTPUTTRIGGER)
        output_trigger_kind_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_KIND) + output_offset
        output_trigger_polarity_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_POLARITY) + output_offset
        output_trigger_source_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_SOURCE) + output_offset
        output_trigger_active_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE) + output_offset
        output_trigger_basesensor_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_BASESENSOR) + output_offset
        output_trigger_delay_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_DELAY) + output_offset
        output_trigger_period_prop = int(DCAM_IDPROP.OUTPUTTRIGGER_PERIOD) + output_offset

        output_readbacks = {
            "OUTPUTTRIGGER_KIND": self._set_property(
                output_trigger_kind_prop, self.output_trigger_kind, errors="raise"
            ),
            "OUTPUTTRIGGER_POLARITY": self._set_property(
                output_trigger_polarity_prop, self.output_trigger_polarity, errors="raise"
            ),
        }

        if self.output_trigger_kind in (
            DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
            DCAMPROP.OUTPUTTRIGGER_KIND.ANYROWEXPOSURE,
        ) and self.output_trigger_basesensor is not None:
            output_readbacks["OUTPUTTRIGGER_BASESENSOR"] = self._set_property(
                output_trigger_basesensor_prop, self.output_trigger_basesensor, errors="ignore"
            )

        if self.output_trigger_kind == DCAMPROP.OUTPUTTRIGGER_KIND.PROGRAMABLE:
            output_readbacks["OUTPUTTRIGGER_SOURCE"] = self._set_property(
                output_trigger_source_prop, self.output_trigger_source, errors="raise"
            )
            output_readbacks["OUTPUTTRIGGER_ACTIVE"] = self._set_property(
                output_trigger_active_prop, self.output_trigger_active, errors="log"
            )
            output_readbacks["OUTPUTTRIGGER_DELAY"] = self._set_property(
                output_trigger_delay_prop, self.output_trigger_delay, errors="raise"
            )
            output_readbacks["OUTPUTTRIGGER_PERIOD"] = self._set_property(
                output_trigger_period_prop, self.output_trigger_period, errors="raise"
            )

        if self.debug_output:
            print(f"--- Applied output trigger connector {self.output_trigger_connector} ---", flush=True)
            for name, value in output_readbacks.items():
                print(f"{name}: {value}", flush=True)
            print("--- end applied output trigger ---\n", flush=True)

    def _set_dcam_internal_buffer(self) -> None:
        buffer_size = c_int32(self.dcam_internal_buffer_size)
        err = dcambuf_alloc(self._dcam_device.hdcam, buffer_size)
        if err.is_failed():
            raise RuntimeError(f"Failed to set device internal buffer: {DCAMERR(err).name}")

        self._dcam_internal_buffer.type = self.pixel_type
        self._dcam_internal_buffer.width = int(self._get_property(DCAM_IDPROP.IMAGE_WIDTH))
        self._dcam_internal_buffer.height = int(self._get_property(DCAM_IDPROP.IMAGE_HEIGHT))
        self._dcam_internal_buffer.rowbytes = int(
            self._get_property(
                DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
                if self.framebundle_mode == DCAMPROP.MODE.ON
                else DCAM_IDPROP.IMAGE_ROWBYTES
            )
        )

        self._dcam_frame.iFrame = -1
        self._dcam_frame.rowbytes = self._dcam_internal_buffer.rowbytes
        self._dcam_frame.type = self._dcam_internal_buffer.type
        self._dcam_frame.width = self._dcam_internal_buffer.width
        self._dcam_frame.height = self._dcam_internal_buffer.height

    def _release_dcam_internal_buffer(self) -> None:
        err = dcambuf_release(self._dcam_device.hdcam, c_int32(0))
        if err.is_failed():
            LOGGER.error(f"Failed to release device internal buffer: {DCAMERR(err).name}")

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
        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func, name="DCAM Acquisition Thread", daemon=True
        )
        self._acquisition_thread.start()

    def _stop_acquisition_thread(self) -> None:
        self._stop_acquisition_event.set()

        self._acquisition_thread.join(timeout=5)
        if self._acquisition_thread.is_alive():
            LOGGER.warning("Streaming thread did not stop gracefully")

        del self._acquisition_thread
        self._stop_acquisition_event.clear()

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
                print(f"FRAME_WAIT_TIMEOUT_MS: {self._timeout}", flush=True)

    def _start_capture(self) -> None:
        err = dcamcap_start(self._dcam_device.hdcam, self.capture_mode)
        if err.is_failed():
            raise RuntimeError(f"Failed to start device capture: {DCAMERR(err).name}")

        if self.debug_output:
            print("--- Capture started; measuring output trigger during active live capture ---", flush=True)
            output_offset = (self.output_trigger_connector - 1) * int(DCAM_IDPROP._OUTPUTTRIGGER)
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
                    print(f"{name}: {self._get_property(prop)}", flush=True)
                except Exception as exc:
                    print(f"{name}: unavailable: {exc}", flush=True)
            print("--- end capture-start timing/output readback ---\n", flush=True)

    def _capture_status(self) -> DCAMCAP_STATUS:
        status = c_int32()
        err = dcamcap_status(self._dcam_device.hdcam, byref(status))
        if err.is_failed():
            raise RuntimeError(f"Failed to get device capture status: {DCAMERR(err).name}")

        return DCAMCAP_STATUS(status.value)

    def _stop_capture(self) -> None:
        err = dcamcap_stop(self._dcam_device.hdcam)
        if err.is_failed():
            LOGGER.error(f"Failed to stop device capture: {DCAMERR(err).name}")

    def _trigger_policy(self) -> None:
        self._framecount_till_software_trigger = 0

        if self.trigger_mode == DCAMPROP.TRIGGER_MODE.START:
            self._software_trigger_cycle = 0
        elif self.trigger_mode == DCAMPROP.TRIGGER_MODE.PIV:
            self._software_trigger_cycle = 2
        else:  # NORMAL
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
            LOGGER.error(f"Failed to close dcam wait: {DCAMERR(err).name}")

        self._dcam_wait = DCAMWAIT_OPEN()

    def _wait_event(self, eventmask: DCAMWAIT_CAPEVENT, timeout):
        self._dcam_wait_event.eventmask = eventmask
        self._dcam_wait_event.timeout = timeout

        err = dcamwait_start(self._dcam_wait.hwait, byref(self._dcam_wait_event))
        if err.is_failed() and err != DCAMERR.TIMEOUT:
            raise RuntimeError(f"Failed to start dcam wait event: {DCAMERR(err).name}")

        if err == DCAMERR.TIMEOUT:
            self._timeout_errors += 1
            LOGGER.warning(
                f"Timeout waiting for frame ready event. Timeout errors: {self._timeout_errors}"
            )
            # TODO do something with errors
            return False

        self._timeout_errors = 0
        return self._dcam_wait_event.eventhappened

    def _dcam_frames_to_buffer(self) -> None:
        with self._acquisition_cond:
            idx = self._dcam_buffer_idx % self.buffer_size

            ptr = self._dcam_buffer[idx]["dcam"].ctypes.data
            self._dcam_frame.buf = c_void_p(ptr)
            self._dcam_buffer[idx]["timestamp"] = monotonic_ns()

            err = dcambuf_copyframe(self._dcam_device.hdcam, byref(self._dcam_frame))
            if err.is_failed():
                raise RuntimeError(f"Failed to copy data: {DCAMERR(err).name}")

            self._dcam_buffer_idx += 1

            self._acquisition_cond.notify_all()

    def _acquire_sample(self) -> bool:
        if not self._wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY, self._timeout):
            return False

        self._dcam_frames_to_buffer()
        self._software_trigger()
        return True

    def _acquisition_thread_func(self) -> None:
        while not self._stop_acquisition_event.is_set():
            try:
                self._acquire_sample()
            except Exception as e:
                self._stop_acquisition_event.set()
                raise e

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()