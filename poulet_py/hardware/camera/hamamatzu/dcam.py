from ctypes import byref, c_double, c_int32, c_void_p
from enum import Enum
from threading import Condition, Event, Thread
from time import monotonic_ns
from typing import Literal

from numpy import ndarray, zeros
from pydantic import BaseModel, Field, PrivateAttr

from poulet_py import LOGGER
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
    binning: DCAMPROP.BINNING = Field(default=DCAMPROP.BINNING._1, description="")
    exposure_time: int = Field(default=30, description="in ms")
    contrast_gain: int = Field(default=10, description="in ms")
    framebundle_mode: DCAMPROP.MODE = Field(default=DCAMPROP.MODE.OFF, description="")
    framebundle_number: int = Field(default=1, description="")
    number_of_view: int = Field(default=1, description="")

    buffer_size: int = Field(default=1000, description="")
    dcam_internal_buffer_size: int = Field(default=1000, description="")
    timeout: int | Literal["auto"] = Field(default="auto", description="handle timeout in ms")
    capture_mode: DCAMCAP_START = Field(default=DCAMCAP_START.SEQUENCE, description="")

    __is_open: bool = PrivateAttr(default=False)
    __dcam_api: DCAMAPI_INIT = PrivateAttr(default_factory=DCAMAPI_INIT)
    __dcam_device: DCAMDEV_OPEN = PrivateAttr(default_factory=DCAMDEV_OPEN)
    __dcam_internal_buffer: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    __dcam_frame: DCAMBUF_FRAME = PrivateAttr(default_factory=DCAMBUF_FRAME)
    __dcam_wait: DCAMWAIT_OPEN = PrivateAttr(default_factory=DCAMWAIT_OPEN)
    __dcam_wait_event: DCAMWAIT_START = PrivateAttr(default_factory=DCAMWAIT_START)

    __buffer: ndarray = PrivateAttr()
    __buffer_idx: int = PrivateAttr(0)
    __buffer_needle: int = PrivateAttr(0)

    __timeout: int = PrivateAttr(default=2)
    __software_trigger_cycle: int = PrivateAttr(default=0)
    __framecount_till_software_trigger: int = PrivateAttr(default=0)
    __acquisition_thread: Thread = PrivateAttr()
    __stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    __acquisition_cond: Condition = PrivateAttr(default_factory=Condition)

    __timeout_errors: int = PrivateAttr(default=0)

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
        return self.__is_open

    def open(self) -> None:
        if self.__is_open:
            return
        try:
            self.__set_dcam_api()
            self.__set_dcam_device()
            self.__set_params()
            self.__set_dcam_internal_buffer()
            self.__set_buffer()
            self.__start_acquisition_thread()

            self.__is_open = True
        except Exception as e:
            self.close()
            raise RuntimeError("Failed to open Dcam") from e

    def close(self) -> None:
        if not self.__is_open:
            return

        self.__is_open = False

        self.__stop_acquisition_thread()
        self.__release_buffer()
        self.__release_dcam_internal_buffer()
        self.__release_dcam_device()
        self.__release_dcam_api()

    def info(self) -> dict[str, str]:
        self.__ensure_open()

        dcam_info = {}
        for idstr in DCAM_IDSTR:
            dev_str = DCAMDEV_STRING()
            dev_str.iString = idstr
            dev_str.alloctext(256)

            err = dcamdev_getstring(self.__dcam_device.hdcam, byref(dev_str))
            if err.is_failed():
                raise RuntimeError(
                    f"Failed to get device information for {idstr}: {DCAMERR(err).name}"
                )

            dcam_info[idstr.name] = dev_str.text.decode()
        return dcam_info

    def read_last_sample(self) -> ndarray:
        self.__ensure_open()

        sample = None
        with self.__acquisition_cond:
            idx = (self.__buffer_idx - 1) % self.buffer_size
            sample = self.__buffer[idx]
            self.__buffer_needle = self.__buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        self.__ensure_open()

        def available():
            return self.__buffer_idx - self.__buffer_needle

        with self.__acquisition_cond:
            if n > 0:
                deadline = None if timeout < 0 else monotonic_ns() + timeout

                while available() < n:
                    if deadline is None:
                        self.__acquisition_cond.wait()
                    else:
                        remaining = deadline - monotonic_ns()
                        if remaining <= 0:
                            break

                        self.__acquisition_cond.wait(remaining)

            avail = available()

            if avail <= 0:
                return 0

            count = avail if n < 0 else min(avail, n)

            if count > self.buffer_size:
                self.__buffer_needle = self.__buffer_idx - self.buffer_size
                count = self.buffer_size

            start = self.__buffer_needle % self.buffer_size
            end = start + count

            if end <= self.buffer_size:
                data[:count] = self.__buffer[start:end]
            else:
                first = self.buffer_size - start
                second = count - first

                data[:first] = self.__buffer[start:]
                data[first:count] = self.__buffer[:second]

            self.__buffer_needle += count

        return count

    def __ensure_open(self) -> None:
        if not self.__is_open:
            raise RuntimeError("DCAM is not open")

    def __set_dcam_api(self) -> None:
        err = dcamapi_init(byref(self.__dcam_api))
        if err.is_failed():
            raise RuntimeError(f"Failed to initialize DCAM-API: {DCAMERR(err).name}")

    def __release_dcam_api(self) -> None:
        dcamapi_uninit()
        self.__dcam_api = DCAMAPI_INIT()

    def __set_dcam_device(self) -> None:
        self.__dcam_device.index = self.device_index

        err = dcamdev_open(byref(self.__dcam_device))
        if err.is_failed():
            raise RuntimeError(f"Failed to initialize DCAM device: {DCAMERR(err).name}")

    def __release_dcam_device(self) -> None:
        dcamdev_close(self.__dcam_device.hdcam)
        self.__dcam_device = DCAMDEV_OPEN()

    def __get_property(self, prop: DCAM_IDPROP) -> float:
        value = c_double()
        err = dcamprop_getvalue(self.__dcam_device.hdcam, prop, byref(value))
        if err.is_failed():
            raise RuntimeError(f"Failed to get property {prop}: {DCAMERR(err).name}")

        return value.value

    def __set_property(
        self,
        prop: DCAM_IDPROP,
        value: float | int | Enum,
        errors: Literal["ignore", "raise", "log"] = "log",
    ) -> None:
        err = dcamprop_setvalue(self.__dcam_device.hdcam, prop, value)
        if err.is_failed():
            msg = f"Failed to set property {DCAM_IDPROP(prop).name}[{prop}]: {DCAMERR(err).name}[{err}]"
            if errors == "raise":
                raise RuntimeError(msg)
            elif errors == "log":
                LOGGER.error(msg)
            else:
                pass

    def __set_params(self) -> None:
        # TODO handle non excisting
        self.__set_property(DCAM_IDPROP.IMAGE_PIXELTYPE, self.pixel_type)
        self.__set_property(DCAM_IDPROP.SENSORMODE, self.sensor_mode)
        self.__set_property(DCAM_IDPROP.SHUTTER_MODE, self.shutter_mode)
        self.__set_property(DCAM_IDPROP.READOUTSPEED, self.readout_speed)
        self.__set_property(DCAM_IDPROP.READOUT_DIRECTION, self.readout_direction)
        self.__set_property(DCAM_IDPROP.TRIGGERSOURCE, self.trigger_source)
        self.__set_property(DCAM_IDPROP.TRIGGER_MODE, self.trigger_mode)
        self.__set_property(DCAM_IDPROP.TRIGGERACTIVE, self.trigger_active)
        self.__set_property(DCAM_IDPROP.TRIGGERPOLARITY, self.trigger_polarity)
        self.__set_property(DCAM_IDPROP.BINNING, self.binning)
        self.__set_property(DCAM_IDPROP.EXPOSURETIME, self.exposure_time)
        self.__set_property(DCAM_IDPROP.CONTRASTGAIN, self.contrast_gain)
        self.__set_property(DCAM_IDPROP.FRAMEBUNDLE_MODE, self.framebundle_mode)

        if self.framebundle_mode == DCAMPROP.MODE.ON:
            self.__set_property(DCAM_IDPROP.FRAMEBUNDLE_NUMBER, self.framebundle_number)
        self.__set_property(DCAM_IDPROP.NUMBEROF_VIEW, self.number_of_view)

    def __set_dcam_internal_buffer(self) -> None:
        buffer_size = c_int32(self.dcam_internal_buffer_size)
        err = dcambuf_alloc(self.__dcam_device.hdcam, buffer_size)
        if err.is_failed():
            raise RuntimeError(f"Failed to set device internal buffer: {DCAMERR(err).name}")

        self.__dcam_internal_buffer.type = self.pixel_type
        self.__dcam_internal_buffer.width = int(self.__get_property(DCAM_IDPROP.IMAGE_WIDTH))
        self.__dcam_internal_buffer.height = int(self.__get_property(DCAM_IDPROP.IMAGE_HEIGHT))
        self.__dcam_internal_buffer.rowbytes = int(
            self.__get_property(
                DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
                if self.framebundle_mode == DCAMPROP.MODE.ON
                else DCAM_IDPROP.IMAGE_ROWBYTES
            )
        )

        self.__dcam_frame.iFrame = -1
        self.__dcam_frame.rowbytes = self.__dcam_internal_buffer.rowbytes
        self.__dcam_frame.type = self.__dcam_internal_buffer.type
        self.__dcam_frame.width = self.__dcam_internal_buffer.width
        self.__dcam_frame.height = self.__dcam_internal_buffer.height

    def __release_dcam_internal_buffer(self) -> None:
        err = dcambuf_release(self.__dcam_device.hdcam, c_int32(0))
        if err.is_failed():
            LOGGER.error(f"Failed to release device internal buffer: {DCAMERR(err).name}")

    def __set_buffer(self) -> None:
        height = self.__dcam_internal_buffer.height * self.framebundle_number * self.number_of_view

        self.__buffer = zeros(
            self.buffer_size,
            dtype=[
                ("timestamp", "uint64"),
                (
                    "dcam",
                    self.DTYPE_MAP.get(self.pixel_type, "float32"),
                    (height, self.__dcam_internal_buffer.width),
                ),
            ],
        )

    def __release_buffer(self) -> None:
        del self.__buffer
        self.__buffer_idx = 0

    def __start_acquisition_thread(self) -> None:
        self.__set_timeout()
        self.__trigger_policy()
        self.__open_dcam_wait()
        self.__start_capture()

        self.__acquisition_thread = Thread(
            target=self.__acquisition_thread_func, name="DCAM Acquisition Thread", daemon=True
        )
        self.__acquisition_thread.start()

    def __stop_acquisition_thread(self) -> None:
        self.__stop_acquisition_event.set()

        self.__acquisition_thread.join(timeout=5)
        if self.__acquisition_thread.is_alive():
            LOGGER.warning("Streaming thread did not stop gracefully")

        del self.__acquisition_thread
        self.__stop_acquisition_event.clear()

        self.__stop_capture()
        self.__close_dcam_wait()

    def __set_timeout(self) -> None:
        if self.timeout == "auto":
            frame_interval = self.__get_property(DCAM_IDPROP.INTERNAL_FRAMEINTERVAL) or 0
            self.__timeout = max(
                self.__timeout, int((self.exposure_time + frame_interval) * 1000.0) + 500
            )

    def __start_capture(self) -> None:
        err = dcamcap_start(self.__dcam_device.hdcam, self.capture_mode)
        if err.is_failed():
            raise RuntimeError(f"Failed to start device capture: {DCAMERR(err).name}")

    def __capture_status(self) -> DCAMCAP_STATUS:
        status = c_int32()
        err = dcamcap_status(self.__dcam_device.hdcam, byref(status))
        if err.is_failed():
            raise RuntimeError(f"Failed to get device capture status: {DCAMERR(err).name}")

        return DCAMCAP_STATUS(status.value)

    def __stop_capture(self) -> None:
        err = dcamcap_stop(self.__dcam_device.hdcam)
        if err.is_failed():
            LOGGER.error(f"Failed to stop device capture: {DCAMERR(err).name}")

    def __trigger_policy(self) -> None:
        self.__framecount_till_software_trigger = 0

        if self.trigger_mode == DCAMPROP.TRIGGER_MODE.START:
            self.__software_trigger_cycle = 0
        elif self.trigger_mode == DCAMPROP.TRIGGER_MODE.PIV:
            self.__software_trigger_cycle = 2
        else:  # NORMAL
            self.__software_trigger_cycle = 1

    def __software_trigger(self) -> None:
        if self.trigger_source == DCAMPROP.TRIGGERSOURCE.SOFTWARE:
            if self.__framecount_till_software_trigger > 0:
                self.__framecount_till_software_trigger -= 1

            if self.__framecount_till_software_trigger == 0:
                err = dcamcap_firetrigger(self.__dcam_device.hdcam, c_int32(0))
                if err.is_failed():
                    raise RuntimeError(f"Failed to software trigger: {DCAMERR(err).name}")

                self.__framecount_till_software_trigger = self.__software_trigger_cycle

    def __open_dcam_wait(self) -> None:
        self.__dcam_wait.hdcam = self.__dcam_device.hdcam
        err = dcamwait_open(byref(self.__dcam_wait))
        if err.is_failed():
            raise RuntimeError(f"Failed to open dcam wait: {DCAMERR(err).name}")

        if self.__dcam_wait.hwait == 0:
            raise RuntimeError(f"Failed to open dcam wait: {DCAMERR.INVALIDWAITHANDLE.name}")

    def __close_dcam_wait(self) -> None:
        err = dcamwait_close(self.__dcam_wait.hwait)
        if err.is_failed():
            LOGGER.error(f"Failed to close dcam wait: {DCAMERR(err).name}")

        self.__dcam_wait = DCAMWAIT_OPEN()

    def __wait_event(self, eventmask: DCAMWAIT_CAPEVENT, timeout):
        self.__dcam_wait_event.eventmask = eventmask
        self.__dcam_wait_event.timeout = timeout

        err = dcamwait_start(self.__dcam_wait.hwait, byref(self.__dcam_wait_event))
        if err.is_failed() and err != DCAMERR.TIMEOUT:
            raise RuntimeError(f"Failed to start dcam wait event: {DCAMERR(err).name}")

        if err == DCAMERR.TIMEOUT:
            self.__timeout_errors += 1
            # TODO do something with errors
            return False

        self.__timeout_errors = 0
        return self.__dcam_wait_event.eventhappened

    def __dcam_frames_to_buffer(self) -> None:
        with self.__acquisition_cond:
            idx = self.__buffer_idx % self.buffer_size

            self.__dcam_frame.buf = self.__buffer[idx]["dcam"].ctypes.data_as(c_void_p)
            self.__buffer[idx]["timestamp"] = monotonic_ns()

            err = dcambuf_copyframe(self.__dcam_device.hdcam, byref(self.__dcam_frame))
            if err.is_failed():
                raise RuntimeError(f"Failed to copy data: {DCAMERR(err).name}")

            self.__buffer_idx += 1

            self.__acquisition_cond.notify_all()
            print(self.__buffer[idx])

    def __acquisition_thread_func(self) -> None:
        self.__software_trigger()
        while not self.__stop_acquisition_event.is_set():
            try:
                if not self.__wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY, self.__timeout):
                    continue

                self.__dcam_frames_to_buffer()
                self.__software_trigger()
            except Exception as e:
                self.close()
                raise e

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
