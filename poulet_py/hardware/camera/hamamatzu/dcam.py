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
    trigger_polarity: DCAMPROP.TRIGGERENABLE_POLARITY = Field(
        default=DCAMPROP.TRIGGERENABLE_POLARITY.NEGATIVE, description=""
    )
    # TODO check
    binning: DCAMPROP.BINNING = Field(default=DCAMPROP.BINNING._1, description="")
    # subarray_mode: TODO
    exposure_time: int = Field(default=9, description="in ms", gt=1, lt=10000)
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
            self._set_dcam_api()
            self._set_dcam_device()
            self._set_params()
            self._set_dcam_internal_buffer()
            self._set_buffer()

            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                self._start_acquisition_thread()

            self._set_timeout()
            self._trigger_policy()
            self._open_dcam_wait()
            self._start_capture()
            self._software_trigger()

            self._is_open = True
        except Exception as e:
            self.close()
            raise RuntimeError("Failed to open Dcam") from e

    def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False

        self._stop_capture()
        self._close_dcam_wait()

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._stop_acquisition_thread()

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

    def _get_property(self, prop: DCAM_IDPROP) -> float:
        value = c_double()
        err = dcamprop_getvalue(self._dcam_device.hdcam, prop, byref(value))
        if err.is_failed():
            raise RuntimeError(f"Failed to get property {prop}: {DCAMERR(err).name}")

        return value.value

    def _set_property(
        self,
        prop: DCAM_IDPROP,
        value: float | int | Enum,
        errors: Literal["ignore", "raise", "log"] = "log",
    ) -> None:
        err = dcamprop_setvalue(self._dcam_device.hdcam, prop, value)
        if err.is_failed():
            msg = f"Failed to set property {DCAM_IDPROP(prop).name}[{prop}]: {DCAMERR(err).name}[{err}]"
            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)
            else:
                pass

    def _set_params(self) -> None:
        self._set_property(DCAM_IDPROP.IMAGE_PIXELTYPE, self.pixel_type)
        self._set_property(DCAM_IDPROP.SENSORMODE, self.sensor_mode)
        self._set_property(DCAM_IDPROP.SHUTTER_MODE, self.shutter_mode)
        self._set_property(DCAM_IDPROP.READOUTSPEED, self.readout_speed)
        self._set_property(DCAM_IDPROP.READOUT_DIRECTION, self.readout_direction)
        self._set_property(DCAM_IDPROP.TRIGGERSOURCE, self.trigger_source)
        self._set_property(DCAM_IDPROP.TRIGGER_MODE, self.trigger_mode)
        self._set_property(DCAM_IDPROP.TRIGGERACTIVE, self.trigger_active)
        self._set_property(DCAM_IDPROP.TRIGGERPOLARITY, self.trigger_polarity)
        self._set_property(DCAM_IDPROP.BINNING, self.binning)
        self._set_property(DCAM_IDPROP.EXPOSURETIME, self.exposure_time / 1000)
        self._set_property(DCAM_IDPROP.CONTRASTGAIN, self.contrast_gain)
        self._set_property(DCAM_IDPROP.FRAMEBUNDLE_MODE, self.framebundle_mode)

        if self.framebundle_mode == DCAMPROP.MODE.ON:
            self._set_property(DCAM_IDPROP.FRAMEBUNDLE_NUMBER, self.framebundle_number)
        self._set_property(DCAM_IDPROP.NUMBEROF_VIEW, self.number_of_view)

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
            frame_interval = self._get_property(DCAM_IDPROP.INTERNAL_FRAMEINTERVAL) or 0
            self._timeout = max(
                self._timeout, int((self.exposure_time + frame_interval) * 1000.0) + 500
            )

    def _start_capture(self) -> None:
        err = dcamcap_start(self._dcam_device.hdcam, self.capture_mode)
        if err.is_failed():
            raise RuntimeError(f"Failed to start device capture: {DCAMERR(err).name}")

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
