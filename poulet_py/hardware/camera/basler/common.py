try:
    from enum import StrEnum
    from threading import Condition, Event, Thread
    from time import monotonic_ns
    from typing import Any, ClassVar, Generic, Literal, TypeVar

    from numpy import ndarray, zeros
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
    from pypylon.pylon import (
        GrabStrategy_LatestImageOnly,
        InstantCamera,
        TimeoutHandling_Return,
        TlFactory,
    )

    from poulet_py import LOGGER, AcquisitionType
except ImportError as e:
    raise ImportError("""
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class SupportedModels(StrEnum):
    ACA800 = "aca800"
    OTHER = "other"


class PixelTypeMixIn:
    def to_numpy(self) -> str: ...


PixelTypeT = TypeVar("PixelTypeT", bound=PixelTypeMixIn)


class _GenericBaslerCamera(BaseModel, Generic[PixelTypeT]):
    MODEL: ClassVar[SupportedModels] = SupportedModels.OTHER

    model_config = ConfigDict(arbitrary_types_allowed=True)  # TODO find solution for PixelTypeMixin

    model: SupportedModels = Field(
        default=SupportedModels.OTHER, description="camera model for specific options"
    )
    device_index: int = Field(default=0, description="")
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )
    buffer_size: int = Field(default=100, description="")
    timeout: int = Field(default=5000, description="handle timeout in ms")

    fps: int = Field(default=30)
    exposure_time: int | Literal["auto"] = Field(
        default="auto", description="in ms", ge=1, le=10000
    )
    contrast_gain: int | Literal["auto"] = Field(default="auto", description="db", ge=1, le=24)
    resolution: tuple[int, int] | None = Field(
        default=None,
        description=(
            "Optional camera-native ROI/resolution as (width, height). "
            "Implemented through Basler subarray properties, not software resizing."
        ),
    )
    offset: tuple[int, int] | None = Field(
        default=None,
        description=("Optional ROI offset. If None, ROI is centered"),
    )
    pixel_type: PixelTypeT  # overwrite in inherited classes
    trigger_mode: Literal["free_run", "frame", "burst"] = "free_run"
    trigger_line: Literal["Line1", "Line3"] = "Line1"
    trigger_activation: Literal["RisingEdge", "FallingEdge"] = "RisingEdge"
    # TODO add more features https://docs.baslerweb.com/features

    _is_open: bool = PrivateAttr(default=False)
    _basler_tl_factory: TlFactory = PrivateAttr()
    _basler_devices: tuple = PrivateAttr()
    _basler_camera: InstantCamera = PrivateAttr()

    _basler_buffer: ndarray = PrivateAttr()
    _basler_buffer_idx: int = PrivateAttr(0)
    _basler_buffer_needle: int = PrivateAttr(0)

    _basler_acquisition_thread: Thread = PrivateAttr()
    _basler_stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _basler_acquisition_cond: Condition = PrivateAttr(default_factory=Condition)

    @staticmethod
    def get_available_devices() -> list[dict[str, str]]:
        tl_factory = TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        res = []
        for idx, device in enumerate(devices):
            res.append({"device_id": idx, "name": device.GetFriendlyName()})
        return res

    @property
    def is_open(self):
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return
        try:
            self._set_basler_tl_factory()
            self._set_basler_camera()
            self._set_basler_params()
            self._set_buffer()

            if self.acquisition_type == AcquisitionType.CONTINUOUS:
                self._start_capture()
                self._start_acquisition_thread()

            self._is_open = True
        except Exception as e:
            raise RuntimeError("Failed to open Basler") from e

    def close(self) -> None:
        if not self._is_open:
            return

        self._is_open = False

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._stop_acquisition_thread()
            self._stop_capture()

        self._release_buffer()
        self._release_basler_camera()
        self._release_basler_tl_factory()

    def info(self) -> dict[str, Any]:
        self._ensure_open()

        return {
            "device_id": self.device_index,
            "name": self._basler_devices[self.device_index].GetFriendlyName(),
        }

    def read_sample(self) -> ndarray | None:
        self._ensure_open()
        sample = None

        if self.acquisition_type == AcquisitionType.FINITE:
            self._start_capture()
            if not self._acquire_sample():
                return sample
            self._stop_capture()

        with self._basler_acquisition_cond:
            idx = (self._basler_buffer_idx - 1) % self.buffer_size
            sample = self._basler_buffer[idx]
            self._basler_buffer_needle = self._basler_buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        self._ensure_open()
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
            with self._basler_acquisition_cond:
                if n == -1 and deadline is None:
                    pass
                elif n == -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    self._basler_acquisition_cond.wait(remaining)
                elif n != -1 and deadline is None:
                    while self._basler_buffer_idx - self._basler_buffer_needle < n:
                        self._basler_acquisition_cond.wait()
                elif n != -1 and deadline is not None:
                    remaining = (deadline - monotonic_ns()) / 1e9
                    while (
                        self._basler_buffer_idx - self._basler_buffer_needle < n and remaining > 0
                    ):
                        self._basler_acquisition_cond.wait(remaining)
                        remaining = (deadline - monotonic_ns()) / 1e9

        with self._basler_acquisition_cond:
            avail = self._basler_buffer_idx - self._basler_buffer_needle
            if avail <= 0:
                return 0

            count = avail if n < 0 else min(avail, n)

            size = self.buffer_size
            buffer = self._basler_buffer
            needle = self._basler_buffer_needle

            if count > size:
                needle = self._basler_buffer_idx - size
                count = size

            start = needle % size
            end = start + count

            if end <= size:
                data[:count] = buffer[start:end]
            else:
                first = size - start
                data[:first] = buffer[start:]
                data[first:count] = buffer[: count - first]

            self._basler_buffer_needle = needle + count

            return count

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("Basler is not open")

    def _set_basler_tl_factory(self) -> None:
        self._basler_tl_factory = TlFactory.GetInstance()

        self._basler_devices = self._basler_tl_factory.EnumerateDevices()
        if len(self._basler_devices) == 0:
            raise Exception("No camera found.")

    def _release_basler_tl_factory(self) -> None:
        del self._basler_devices
        del self._basler_tl_factory

    def _set_basler_camera(self) -> None:
        if self.device_index >= len(self._basler_devices):
            raise IndexError(
                f"Device index {self.device_index} out of range."
                "Use get_available_devices() to list available cameras."
            )

        self._basler_camera = InstantCamera(
            self._basler_tl_factory.CreateDevice(self._basler_devices[self.device_index])
        )
        self._basler_camera.Open()

    def _release_basler_camera(self) -> None:
        self._basler_camera.Close()
        del self._basler_camera

    def _set_model_dedicated_params(self):
        """
        implement in subclass if needed
        """
        return

    def _set_basler_params(self):
        if self.exposure_time != "auto":
            self._basler_camera.ExposureMode.Value = "Timed"
            self._basler_camera.ExposureAuto.Value = "Off"
            self._basler_camera.ExposureTimeAbs.Value = self.exposure_time

        if self.contrast_gain != "auto":
            self._basler_camera.GainAuto.Value = "Off"
            self._basler_camera.GainRaw.Value = self.contrast_gain

        if self.resolution:
            self._basler_camera.Height.Value = self.resolution[0]
            self._basler_camera.Width.Value = self.resolution[1]
        else:
            self._basler_camera.Height.Value = self._basler_camera.Height.Max
            self._basler_camera.Width.Value = self._basler_camera.Width.Max

        if self.trigger_mode == "free_run":
            self._basler_camera.TriggerSelector.Value = "FrameStart"
            self._basler_camera.TriggerMode.Value = "Off"

            self._basler_camera.AcquisitionFrameRateEnable.Value = True
            self._basler_camera.AcquisitionFrameRateAbs.Value = self.fps

        elif self.trigger_mode == "frame":
            self._basler_camera.TriggerSelector.Value = "FrameStart"
            self._basler_camera.TriggerMode.Value = "On"
            self._basler_camera.TriggerSource.Value = self.trigger_line
            self._basler_camera.TriggerActivation.Value = self.trigger_activation

        elif self.trigger_mode == "gate":
            self._basler_camera.AcquisitionFrameRateEnable.Value = True
            self._basler_camera.AcquisitionFrameRateAbs.Value = self.fps

            self._basler_camera.TriggerSelector.Value = "AcquisitionStart"
            self._basler_camera.TriggerMode.Value = "On"
            self._basler_camera.TriggerSource.Value = self.trigger_line
            self._basler_camera.TriggerActivation.Value = self.trigger_activation

        self._set_model_dedicated_params()

    def _set_buffer(self) -> None:
        resolution = self.resolution or (
            self._basler_camera.Height.Max,
            self._basler_camera.Width.Max,
        )
        self._basler_buffer = zeros(
            self.buffer_size,
            dtype=[
                ("timestamp", "uint64"),
                ("basler", self.pixel_type.to_numpy(), resolution),
            ],
        )

    def _release_buffer(self) -> None:
        del self._basler_buffer
        self._basler_buffer_idx = 0

    def _start_capture(self) -> None:
        self._basler_camera.StartGrabbing(GrabStrategy_LatestImageOnly)

    def _capture_status(self) -> bool:
        return self._basler_camera.IsGrabbing()

    def _stop_capture(self) -> None:
        self._basler_camera.StopGrabbing()

    def _start_acquisition_thread(self) -> None:
        self._acquisition_thread = Thread(
            target=self._acquisition_thread_func, name="Basler Acquisition Thread", daemon=True
        )
        self._acquisition_thread.start()

    def _stop_acquisition_thread(self) -> None:
        self._basler_stop_acquisition_event.set()

        self._acquisition_thread.join(timeout=5)
        if self._acquisition_thread.is_alive():
            LOGGER.warning("Streaming thread did not stop gracefully")

        del self._acquisition_thread
        self._basler_stop_acquisition_event.clear()

    def _basler_frames_to_buffer(self, data: ndarray) -> None:
        with self._basler_acquisition_cond:
            idx = self._basler_buffer_idx % self.buffer_size

            self._basler_buffer[idx]["basler"] = data

            self._basler_buffer[idx]["timestamp"] = monotonic_ns()
            self._basler_buffer_idx += 1

            self._basler_acquisition_cond.notify_all()

    def _acquire_sample(self) -> bool:
        if not self._capture_status():
            return False

        result = self._basler_camera.RetrieveResult(self.timeout, TimeoutHandling_Return)

        if result.GrabSucceeded():
            self._basler_frames_to_buffer(result.GetArray())

        result.Release()

        return True

    def _acquisition_thread_func(self) -> None:
        while not self._basler_stop_acquisition_event.is_set():
            try:
                self._acquire_sample()
            except Exception as e:
                self._basler_stop_acquisition_event.set()
                raise e

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
