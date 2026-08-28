from __future__ import annotations

import importlib
import platform
import sys
import time
from collections.abc import Callable
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_uint16,
    c_void_p,
    cast,
    sizeof,
)
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from time import monotonic_ns
from typing import Any, Literal

try:
    import numpy as np
    from numpy.typing import NDArray
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

    from poulet_py import LOGGER
except ImportError as e:
    raise ImportError(
        """
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    ) from e


GainMode = Literal["high", "low", "auto"]
FfcMode = Literal["auto", "manual", "external"]


class LeptonRadFluxLinearParams(Structure):
    """Lepton RAD Flux Linear Parameters (eight 16-bit words)."""

    _fields_ = [
        ("sceneEmissivity", c_uint16),
        ("TBkgK", c_uint16),
        ("tauWindow", c_uint16),
        ("TWindowK", c_uint16),
        ("tauAtm", c_uint16),
        ("TAtmK", c_uint16),
        ("reflWindow", c_uint16),
        ("TReflK", c_uint16),
    ]


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    """One Python-owned Y16 frame received from libuvc."""

    timestamp: int
    sequence: int
    raw: NDArray[np.uint16]


def _load_uvctypes() -> Any:
    """Load the frozen ctypes/libuvc compatibility module on demand."""

    try:
        uvc = importlib.import_module("poulet_py.hardware.camera.thermal.uvctypes")
    except Exception as e:
        raise RuntimeError("Failed to load the libuvc compatibility layer") from e

    if not hasattr(uvc, "libuvc"):
        raise RuntimeError("libuvc could not be loaded")
    return uvc


class ThermalCamera(BaseModel):
    """Hardware interface for a PureThermal-attached FLIR Lepton 3.5.

    Frames remain raw TLinear Y16 values. With the camera's expected 0.01 K
    TLinear resolution, Celsius conversion is ``raw * 0.01 - 273.15``. This
    assumption is reported by :meth:`info` but is not yet verified by readback.
    """

    model_config = ConfigDict(validate_assignment=True)

    emissivity: float = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Scene emissivity used by the Lepton TLinear calculation",
    )

    gain_mode: GainMode = Field(
        default="high",
        description="Lepton gain mode (high, low, or auto)",
    )

    ffc_mode: FfcMode = Field(
        default="auto", description="Lepton FFC shutter mode (auto, manual, or external)"
    )

    # only during initialization
    frame_queue_size: int = Field(
        default=4,
        ge=1,
        description="Maximum copied frames waiting for a consumer",
    )

    frame_timeout_s: float = Field(
        default=0.25,
        gt=0,
        description="Default read timeout in seconds",
    )

    _uvc: Any = PrivateAttr(default=None)
    _ctx: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)
    _device_handle: Any = PrivateAttr(default=None)
    _stream_ctrl: Any = PrivateAttr(default=None)
    _callback: Any = PrivateAttr(default=None)
    _frames: Queue[ThermalFrame] = PrivateAttr()
    _flux_linear_params: LeptonRadFluxLinearParams | None = PrivateAttr(default=None)
    _windows_camera: Any = PrivateAttr(default=None)
    _frame_rate: float = PrivateAttr(default=None)

    _is_windows: bool = PrivateAttr(default=False)
    _is_open: bool = PrivateAttr(default=False)
    _is_streaming: bool = PrivateAttr(default=False)
    _dropped_frames: int = PrivateAttr(default=0)
    _width: int = PrivateAttr(default=160)
    _height: int = PrivateAttr(default=120)

    _windows_sequence = PrivateAttr(default=0)
    _LIBUVC_FRAME_FORMAT_GREY16: int = PrivateAttr(default=10)

    def model_post_init(self, __context: Any) -> None:
        self._frames = Queue(maxsize=self.frame_queue_size)
        self._is_windows = platform.system() == "Windows"

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._width, self._height

    def open(self) -> None:
        """Open the camera and negotiate its first Y16 stream format."""

        if self._is_open:
            return

        if self._is_windows:
            if self._windows_camera is None:
                self._windows_camera = CameraWindows(self._receive_windows_frame)
            try:
                self._windows_camera.open()
                self._windows_camera.set_emissivity(self.emissivity)
            except Exception:
                self._windows_camera.close()
                raise
            self._is_open = True
            return

        self._uvc = _load_uvctypes()
        uvc = self._uvc
        ctx = POINTER(uvc.uvc_context)()
        device = POINTER(uvc.uvc_device)()
        device_handle = POINTER(uvc.uvc_device_handle)()
        stream_ctrl = uvc.uvc_stream_ctrl()

        try:
            result = uvc.libuvc.uvc_init(byref(ctx), 0)

            if result < 0:
                raise RuntimeError(f"uvc_init failed: {result}")
            self._ctx = ctx

            # This could allow for different devices, but for now we just use the first one found
            # For that the o placeholder would need to be the correct serial number of the device
            result = uvc.libuvc.uvc_find_device(
                ctx,
                byref(device),
                uvc.PT_USB_VID,
                uvc.PT_USB_PID,
                0,
            )
            if result < 0:
                raise RuntimeError(f"uvc_find_device failed: {result}")
            self._device = device

            result = uvc.libuvc.uvc_open(device, byref(device_handle))
            if result < 0:
                raise RuntimeError(f"uvc_open failed: {result}")
            self._device_handle = device_handle

            frame_formats = uvc.uvc_get_frame_formats_by_guid(
                device_handle,
                uvc.VS_FMT_GUID_Y16,
            )
            if not frame_formats:
                raise RuntimeError("PureThermal device does not expose a Y16 format")

            frame_format = frame_formats[0]
            if frame_format.dwDefaultFrameInterval <= 0:
                raise RuntimeError("PureThermal returned an invalid frame interval")

            self._width = int(frame_format.wWidth)
            self._height = int(frame_format.wHeight)
            self._frame_rate = 1e7 / float(frame_format.dwDefaultFrameInterval)

            result = uvc.libuvc.uvc_get_stream_ctrl_format_size(
                device_handle,
                byref(stream_ctrl),
                self._LIBUVC_FRAME_FORMAT_GREY16,
                self._width,
                self._height,
                int(self._frame_rate),
            )

            if result < 0:
                raise RuntimeError(f"Y16 stream negotiation failed: {result}")

            self._stream_ctrl = stream_ctrl
            self._install_callback()
            self._write_emissivity(self.emissivity)
            self._is_open = True
            LOGGER.info(
                "Opened PureThermal Y16 stream %sx%s at %.3f fps",
                self._width,
                self._height,
                self._frame_rate,
            )
        except Exception:
            self._release_uvc_resources()
            raise

    def start_streaming(self) -> None:
        """Start continuous frame delivery through the libuvc callback."""

        if self._is_streaming:
            return
        if not self._is_open:
            self.open()

        self.clear_frames()
        self._dropped_frames = 0

        if self._is_windows:
            self._windows_camera.start_streaming()
        else:
            result = self._uvc.libuvc.uvc_start_streaming(
                self._device_handle,
                byref(self._stream_ctrl),
                self._callback,
                None,
                0,
            )
            if result < 0:
                raise RuntimeError(f"uvc_start_streaming failed: {result}")

        self._is_streaming = True
        try:
            # Sequence: stream first, then FFC and gain.
            self._apply_ffc_mode(self.ffc_mode)
            self._apply_gain_mode(self.gain_mode)
        except Exception:
            self.stop_streaming()
            raise
        # if frames were recorded during the FFC and gain application, they are now dropped.
        self.clear_frames()
        self._dropped_frames = 0

        LOGGER.info("Started PureThermal stream")

    def stop_streaming(self) -> None:
        """Stop frame delivery without releasing the camera handle."""

        if not self._is_streaming:
            return

        if self._is_windows:
            self._windows_camera.stop_streaming()
        else:
            self._uvc.libuvc.uvc_stop_streaming(self._device_handle)

        self._is_streaming = False
        LOGGER.info("Stopped PureThermal stream")

    def close(self) -> None:
        """Stop streaming and release every camera/libuvc resource."""

        try:
            self.stop_streaming()
        finally:
            if self._is_windows:
                if self._windows_camera is not None:
                    self._windows_camera.close()
            else:
                self._release_uvc_resources()
            self._is_streaming = False
            self._is_open = False
            self._flux_linear_params = None
            self.clear_frames()

    def read_sample(self, timeout: float | None = None) -> ThermalFrame | None:
        """Return the next copied frame, or ``None`` when the timeout expires."""

        self._ensure_streaming()
        wait = self.frame_timeout_s if timeout is None else timeout
        if wait < 0:
            raise ValueError("timeout must be non-negative")

        try:
            return self._frames.get(timeout=wait)
        except Empty:
            return None

    def clear_frames(self) -> None:
        """Discard frames currently waiting in the driver queue."""

        while True:
            try:
                self._frames.get_nowait()
            except Empty:
                return

    def set_emissivity(self, emissivity: float) -> None:
        """Apply scene emissivity while preserving the other flux parameters."""

        self._ensure_open()
        old_value = self.emissivity
        self.emissivity = emissivity
        try:
            self._write_emissivity(self.emissivity)
        except Exception:
            self.emissivity = old_value
            raise
        if self._is_streaming:
            self.clear_frames()

    def set_ffc_mode(self, mode: FfcMode) -> None:
        """Set automatic, manual, or externally controlled FFC shutter mode."""
        self._ensure_open()
        old_mode = self.ffc_mode
        self.ffc_mode = mode
        try:
            self._apply_ffc_mode(self.ffc_mode)
        except Exception:
            self.ffc_mode = old_mode
            raise
        if self._is_streaming:
            self.clear_frames()

    def set_gain_mode(self, mode: GainMode) -> None:
        """Set high, low, or automatic Lepton gain mode."""
        self._ensure_open()
        old_mode = self.gain_mode
        self.gain_mode = mode
        try:
            self._apply_gain_mode(self.gain_mode)
        except Exception:
            self.gain_mode = old_mode
            raise
        if self._is_streaming:
            self.clear_frames()

    def perform_manual_ffc(self) -> None:
        """Run one flat-field correction normalization."""
        # Needs changes to uvctypes to read back changes and not silently assume changes
        self._ensure_open()
        if self._is_windows:
            self._windows_camera.perform_manual_ffc()
        else:
            self._uvc.perform_manual_ffc(self._device_handle)
        if self._is_streaming:
            self.clear_frames()

    def info(self) -> dict[str, Any]:
        """Return the effective hardware configuration known to this driver."""

        self._ensure_open()
        return {
            "width": self._width,
            "height": self._height,
            "frame_rate": self._frame_rate,
            "pixel_format": "Y16",
            "emissivity": self.emissivity,
            "gain_mode": self.gain_mode,
            "ffc_mode": self.ffc_mode,
            "tlinear_resolution_kelvin": 0.01,
            "tlinear_state_verified": False,
            "dropped_frames": self._dropped_frames,
        }

    def _install_callback(self) -> None:
        callback_type = CFUNCTYPE(None, POINTER(self._uvc.uvc_frame), c_void_p)

        def on_frame(frame: Any, _userptr: Any) -> None:
            if not frame:
                return
            timestamp = monotonic_ns()
            contents = frame.contents
            width = int(contents.width)
            height = int(contents.height)
            expected_bytes = 2 * width * height
            if (
                not contents.data
                or width != self._width
                or height != self._height
                or contents.data_bytes != expected_bytes
            ):
                self._dropped_frames += 1
                return

            data_pointer = cast(contents.data, POINTER(c_uint16))
            raw = np.ctypeslib.as_array(data_pointer, shape=(width * height,))
            raw = raw.reshape(height, width).copy()
            self._enqueue_frame(
                ThermalFrame(
                    timestamp=timestamp,
                    sequence=int(contents.sequence),
                    raw=raw,
                )
            )

        # ctypes callbacks must remain referenced for the full stream lifetime.
        self._callback = callback_type(on_frame)

    def _receive_windows_frame(self, raw: NDArray[np.uint16]) -> None:
        timestamp = monotonic_ns()
        self._windows_sequence += 1
        self._enqueue_frame(
            ThermalFrame(
                timestamp=timestamp,
                sequence=self._windows_sequence,
                raw=raw.copy(),
            )
        )

    def _enqueue_frame(self, sample: ThermalFrame) -> None:
        try:
            self._frames.put_nowait(sample)
            return
        except Full:
            pass

        try:
            self._frames.get_nowait()
            self._dropped_frames += 1
        except Empty:
            pass

        try:
            self._frames.put_nowait(sample)
        except Full:
            # A race must never block the libuvc callback.
            self._dropped_frames += 1

    def _write_emissivity(self, emissivity: float) -> None:
        if self._is_windows:
            self._windows_camera.set_emissivity(emissivity)
            return

        if self._flux_linear_params is None:
            params = LeptonRadFluxLinearParams()
            control_id = (0xBC >> 2) + 1
            result = self._uvc.call_extension_unit(
                self._device_handle,
                self._uvc.RAD_UNIT_ID,
                control_id,
                byref(params),
                sizeof(params),
            )
            if result < 0:
                raise RuntimeError(f"Failed to read radiometry parameters: {result}")
            self._flux_linear_params = params

        self._flux_linear_params.sceneEmissivity = round(emissivity * 8192)
        control_id = (0xBD >> 2) + 1
        result = self._uvc.set_extension_unit(
            self._device_handle,
            self._uvc.RAD_UNIT_ID,
            control_id,
            byref(self._flux_linear_params),
            sizeof(self._flux_linear_params),
        )
        if result < 0:
            raise RuntimeError(f"Failed to set emissivity to {emissivity}: {result}")

    def _apply_ffc_mode(self, mode: FfcMode) -> None:
        if self._is_windows:
            self._windows_camera.set_ffc_mode(mode)
            return

        setters = {
            "auto": self._uvc.set_auto_ffc,
            "manual": self._uvc.set_manual_ffc,
            "external": self._uvc.set_external_ffc,
        }
        setters[mode](self._device_handle)

    def _apply_gain_mode(self, mode: GainMode) -> None:
        if self._is_windows:
            self._windows_camera.set_gain_mode(mode)
            return

        setters = {
            "high": self._uvc.set_gain_high,
            "low": self._uvc.set_gain_low,
            "auto": self._uvc.set_gain_auto,
        }
        setters[mode](self._device_handle)

    def _release_uvc_resources(self) -> None:
        if self._uvc is None:
            return

        if self._device_handle:
            self._uvc.libuvc.uvc_close(self._device_handle)
        if self._device:
            self._uvc.libuvc.uvc_unref_device(self._device)
        if self._ctx:
            self._uvc.libuvc.uvc_exit(self._ctx)

        self._callback = None
        self._stream_ctrl = None
        self._device_handle = None
        self._device = None
        self._ctx = None
        self._uvc = None

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("ThermalCamera needs to be opened first")

    def _ensure_streaming(self) -> None:
        if not self._is_streaming:
            raise RuntimeError("ThermalCamera needs to be streaming first")

    def __enter__(self) -> ThermalCamera:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class CameraWindows:
    """Compatibility adapter for the existing Windows LeptonUVC route."""

    def __init__(self, on_frame: Callable[[NDArray[np.uint16]], None]) -> None:
        try:
            import clr
            import pythoncom
        except ImportError as e:
            raise RuntimeError("Windows LeptonUVC dependencies are unavailable") from e

        folder = "x64" if platform.architecture()[0] == "64bit" else "x86"
        source_root = Path(__file__).resolve().parents[3]
        package_root = Path(__file__).resolve().parents[2]
        candidates = (
            source_root / "artifacts" / "leptonUVC" / folder,
            package_root / "artifacts" / "leptonUVC" / folder,
        )
        artifacts = next((path for path in candidates if path.exists()), candidates[0])
        if str(artifacts) not in sys.path:
            sys.path.append(str(artifacts))

        clr.AddReference("LeptonUVC")
        clr.AddReference("ManagedIR16Filters")
        from IR16Filters import IR16Capture, NewBytesFrameEvent
        from Lepton import CCI

        self._pythoncom = pythoncom
        self._cci = CCI
        self._capture_type = IR16Capture
        self._callback_type = NewBytesFrameEvent
        self._on_frame = on_frame
        self._device: Any = None
        self._reader: Any = None
        self._callback: Any = None
        self._latest_frame: NDArray[np.uint16] | None = None
        self._com_initialized = False
        self._streaming = False

    def open(self) -> None:
        self._pythoncom.CoInitialize()
        self._com_initialized = True
        time.sleep(1)
        devices = [d for d in self._cci.GetDevices() if d.Name.startswith("PureThermal")]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one PureThermal device, found {len(devices)}")

        self._device = devices[0].Open()
        self._reader = self._capture_type()
        self._callback = self._callback_type(self._add_frame)
        self._reader.SetupGraphWithBytesCallback(self._callback)

    def _add_frame(self, array: Any, width: int, height: int) -> None:
        raw = np.fromiter(array, dtype=np.uint16).reshape(height, width).copy()
        self._latest_frame = raw
        self._on_frame(raw)

    def start_streaming(self) -> None:
        self._reader.RunGraph()
        self._streaming = True

    def stop_streaming(self) -> None:
        if self._streaming:
            self._reader.StopGraph()
            self._streaming = False

    def close(self) -> None:
        self.stop_streaming()
        if self._com_initialized:
            self._pythoncom.CoUninitialize()
            self._com_initialized = False

    def set_emissivity(self, emissivity: float) -> None:
        params = self._device.rad.GetFluxLinearParams()
        if hasattr(params, "sceneEmissivity"):
            field = "sceneEmissivity"
        elif hasattr(params, "SceneEmissivity"):
            field = "SceneEmissivity"
        else:
            raise AttributeError("FluxLinearParams has no scene-emissivity field")
        setattr(params, field, round(float(emissivity) * 8192))
        self._device.rad.SetFluxLinearParams(params)

    def set_ffc_mode(self, mode: FfcMode) -> None:
        shutter = self._device.sys.GetFfcShutterModeObj()
        shutter.shutterMode = getattr(self._cci.Sys.FfcShutterMode, mode.upper())
        self._device.sys.SetFfcShutterModeObj(shutter)

    def set_gain_mode(self, mode: GainMode) -> None:
        self._device.sys.SetGainMode(getattr(self._cci.Sys.GainMode, mode.upper()))

    def perform_manual_ffc(self) -> None:
        self._device.sys.RunFFCNormalization()

    def get_frame(self) -> NDArray[np.uint16] | None:
        return self._latest_frame
