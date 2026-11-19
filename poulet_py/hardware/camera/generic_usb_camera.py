try:
    from time import monotonic_ns
    from typing import ClassVar, Literal

    from cv2 import (
        CAP_DSHOW,
        CAP_MSMF,
        CAP_PROP_AUTO_EXPOSURE,
        CAP_PROP_BRIGHTNESS,
        CAP_PROP_CONTRAST,
        CAP_PROP_CONVERT_RGB,
        CAP_PROP_EXPOSURE,
        CAP_PROP_FOURCC,
        CAP_PROP_FPS,
        CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FRAME_WIDTH,
        CAP_PROP_GAIN,
        CAP_PROP_SATURATION,
        CAP_V4L2,
        VideoCapture,
        VideoWriter_fourcc,
    )
    from cv2_enumerate_cameras import CameraInfo, enumerate_cameras
    from numpy import isclose, log2, ndarray
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER
    from poulet_py.hardware import BaseHardware

except ImportError as e:
    raise ImportError("""
Missing 'hardware' module. Install options:
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class GenericUSBCamera(BaseHardware):
    _AUTO_EXPOSURE: ClassVar[dict[int, dict[str, float] | dict[str, int]]] = {
        CAP_V4L2: {"auto": 3, "manual": 1},
        CAP_DSHOW: {"auto": 0.75, "manual": 0.25},
        CAP_MSMF: {"auto": 1, "manual": 0},
    }

    camera_idx: int = Field(...)
    resolution: tuple[int, int] = Field(default=(1280, 720))
    fps: int = Field(default=30)
    exposure: int | Literal["auto"] = Field(default="auto")
    brightness: int | None = Field(default=None)
    contrast: int | None = Field(default=None)
    saturation: int | None = Field(default=None)
    gain: int | None = Field(default=None)
    codec: Literal["MJPG", "YUYV", "NV12", "H264"] | None = Field(default="MJPG")

    _cam: CameraInfo = PrivateAttr()
    _cap: VideoCapture = PrivateAttr()
    _grayscale: bool = PrivateAttr(default=False)

    @staticmethod
    def get_available_devices() -> list[CameraInfo]:
        devices = []
        for dev in enumerate_cameras():
            devices.append(dev)
        return devices

    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
        return [
            (
                "data",
                "uint8",
                (self.resolution[1], self.resolution[0])
                if self._grayscale
                else (self.resolution[1], self.resolution[0], 3),
            ),
        ]

    def _open(self) -> None:
        self._set_capture()
        self._set_parameters()
        self._get_frame_resolution()

    def _close(self) -> None:
        self._release_capture()

    def _set_capture(self) -> None:
        for i in enumerate_cameras():
            if i.index == self.camera_idx:
                self._cam = i
                break

        if not hasattr(self, "_cam"):
            raise RuntimeError(f"{type(self).__name__} Camera not found")

        self._cap = VideoCapture(self._cam.index, self._cam.backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"{type(self).__name__} Could not open camera")

    def _release_capture(self) -> None:
        self._cap.release()
        del self._cap

    def _set_parameter(
        self,
        name: str,
        param_id: int,
        value: float,
        on_error: Literal["raise", "log"] | None = None,
    ) -> None:
        ret = self._cap.set(param_id, value)
        if not ret:
            if on_error == "raise":
                raise RuntimeError(f"{type(self).__name__} could not set parameter {param_id}")
            if on_error == "log":
                LOGGER.error(f"{type(self).__name__} could not set parameter {param_id}")

        cam_val = self._cap.get(param_id)
        if not isclose(cam_val, value):
            if on_error:
                LOGGER.warning(f"Requested value for {name}: {value}, camera set {cam_val}")

    def _set_parameters(self) -> None:
        self._set_parameter(
            "fourcc", CAP_PROP_FOURCC, VideoWriter_fourcc(*self.codec), on_error="log"
        )

        self._set_parameter("width", CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._set_parameter("height", CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._set_parameter("fps", CAP_PROP_FPS, self.fps)

        self._set_exposure()

        for name, prop in (
            ("brightness", CAP_PROP_BRIGHTNESS),
            ("contrast", CAP_PROP_CONTRAST),
            ("saturation", CAP_PROP_SATURATION),
            ("gain", CAP_PROP_GAIN),
        ):
            value = getattr(self, name)
            if value is not None:
                self._set_parameter(name, prop, value, on_error="log")

        self._set_parameter("convert_rgb", CAP_PROP_CONVERT_RGB, 1, on_error="log")

    def _set_exposure(self) -> None:
        modes = self._AUTO_EXPOSURE.get(self._cam.backend)
        if modes is None:
            LOGGER.warning(f"Exposure control not supported on backend {self._cam.backend}")
            return

        if self.exposure == "auto":
            self._set_parameter(
                "auto_exposure", CAP_PROP_AUTO_EXPOSURE, modes["auto"], on_error="log"
            )
        else:
            self._set_parameter(
                "auto_exposure", CAP_PROP_AUTO_EXPOSURE, modes["manual"], on_error="log"
            )
            self._set_parameter(
                "exposure",
                CAP_PROP_EXPOSURE,
                self._exposure_to_backend(self.exposure),
                on_error="log",
            )

    def _exposure_to_backend(self, us: int) -> float:
        if self._cam.backend == CAP_V4L2:
            return us / 100
        return log2(us / 1e6)

    def _get_frame_resolution(self) -> None:
        frame = None
        for _ in range(3):
            ok, frame = self._cap.read()

            if not ok:
                raise RuntimeError(f"{type(self).__name__} Could not read frame")

        if frame is None:
            raise RuntimeError(f"{type(self).__name__} Could not read frame")

        h, w = frame.shape[:2]

        self.resolution = (w, h)
        self._grayscale = frame.ndim == 2 or frame.shape[2] == 1

    def _start_capture(self) -> None:
        if not self._cap.grab():
            raise RuntimeError(f"{type(self).__name__} Couldn't retrieve data")

    def _stop_capture(self) -> None:
        pass

    def _acquire_sample(self) -> ndarray | None:
        ret, frame = self._cap.retrieve()

        if not ret:
            return

        return frame

    def _acquisition_thread_func(self) -> None:
        cap, stop, misses = self._cap, self._stop_acquisition_event, 0

        try:
            while not stop.is_set():
                if not cap.grab():
                    misses += 1
                    if misses > 10:
                        raise RuntimeError(
                            f"{type(self).__name__} Camera stopped delivering frames"
                        )
                    stop.wait(0.005)
                    continue

                ts = monotonic_ns()

                ok, frame = cap.retrieve()
                if not ok:
                    continue

                misses = 0
                self._sample_to_buffer(frame, ts)
        except Exception as e:
            LOGGER.exception(e)
        finally:
            with self._acquisition_condition:
                self._acquisition_condition.notify_all()
