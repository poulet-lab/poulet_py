try:
    from ctypes import CFUNCTYPE, POINTER, byref, c_uint16, c_void_p, cast, sizeof
    from pathlib import Path
    from platform import architecture, system
    from sys import path
    from time import monotonic_ns
    from typing import Any, Literal

    import numpy as np
    from pydantic import Field, PrivateAttr

    from poulet_py.hardware import BaseHardware
    from poulet_py.hardware.camera.flir.uvctypes import (
        Gain,
        RadFluxLinearParams,
        UvcContext,
        UvcDevice,
        UvcDeviceHandle,
        UVCStreamCtrl,
        libuvc,
    )

    if system == "Windows":
        try:
            from IR16Filters import IR16Capture
            from Lepton import CCI
            from pythoncom import CoInitialize, CoUninitialize
        except ImportError as e:
            raise RuntimeError("Windows LeptonUVC dependencies are unavailable") from e

except ImportError as e:
    raise ImportError(
        """
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    ) from e


class Lepton(BaseHardware):
    device: int | str = Field(...)
    resolution: tuple[int, int] = Field((1280, 720))
    fps: int = Field(default=30)
    emissivity: float = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Scene emissivity used by the Lepton TLinear calculation",
    )
    gain: Gain = Field(default=Gain.HIGH, description="Lepton gain mode (high, low, or auto)")
    ffc_mode: Literal["auto", "manual", "external"] = Field(
        default="auto", description="Lepton FFC shutter mode (auto, manual, or external)"
    )

    _is_windows: bool = PrivateAttr(default=False)
    _com_initialized = False
    _ctx: UvcContext = PrivateAttr(default_factory=UvcContext)
    _device: UvcDevice = PrivateAttr(default_factory=UvcDevice)
    _reader: Any = PrivateAttr()
    _device_handle: UvcDeviceHandle = PrivateAttr(default_factory=UvcDeviceHandle)
    _stream_ctrl: UVCStreamCtrl = PrivateAttr(default_factory=UVCStreamCtrl)
    _frame_rate: float | None = PrivateAttr(default=None)

    @staticmethod
    def get_available_devices() -> list[dict[str, str | int]]:
        res = []
        if system == "Windows":
            CoInitialize()
            devices = CCI.GetDevices()
            for idx, dev in enumerate(devices):
                res.append({"index": idx, "name": dev.Name})
            CoUninitialize()

        return res

    def info(self) -> dict[str, str | int]:
        self._ensure_open()

        return {}

    def model_post_init(self, context: Any, /) -> None:
        self._is_windows = system == "Windows"

        if self._is_windows:
            folder = "x64" if architecture()[0] == "64bit" else "x86"
            source_root = Path(__file__).resolve().parents[4]
            artifacts = source_root / "artifacts" / "leptonUVC" / folder
            if not artifacts.exists():
                raise FileNotFoundError(f"{type(self).__name__} LeptonUVC dlls not found")

            if str(artifacts) not in path:
                path.append(str(artifacts))

    def _open(self) -> None:
        self._set_com_or_ctx()
        self._set_device()
        self._set_reader_device_handle()
        stream_ctrl = UVCStreamCtrl()

        try:
            result = libuvc.uvc_init(byref(self._ctx), 0)

            if result < 0:
                raise RuntimeError(f"uvc_init failed: {result}")

            # This could allow for different devices, but for now we just use the first one found
            # For that the o placeholder would need to be the correct serial number of the device
            result = libuvc.uvc_find_device(
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
                10,
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

        except Exception:
            self._release_uvc_resources()
            raise

    def _close(self) -> None:
        self._release_device()
        self._release_com_or_ctx()

    def _set_com_or_ctx(self) -> None:
        if self._is_windows:
            CoInitialize()
            self._com_initialized = True
            return

        self._ctx = UvcContext()

    def _release_com_or_ctx(self) -> None:
        if self._com_initialized:
            CoUninitialize()
            self._com_initialized = False
            return

        libuvc.uvc_exit(self._ctx)
        self._ctx = UvcContext()

    def _set_device(self) -> None:
        if self._is_windows:
            devices = list(CCI.GetDevices())
            if isinstance(self.device, int):
                if len(devices) < self.device:
                    raise RuntimeError(
                        f"{type(self).__name__} device index higher than found devices"
                    )

            self._device = devices[0].Open()
            return

        self._device = UvcDevice()

    def _release_device(self) -> None:
        if self._is_windows:
            devices = list(CCI.GetDevices())
            if isinstance(self.device, int):
                if len(devices) < self.device:
                    raise RuntimeError(
                        f"{type(self).__name__} device index higher than found devices"
                    )

                self._device = devices[self.device]
            else:
                for dev in devices:
                    if dev.Name == self.device:
                        self._device = dev
                        break

            self._device.Open()
            return

        libuvc.uvc_unref_device(self._device)
        self._device = UvcDevice()

        if self._device_handle:
            self._uvc.libuvc.uvc_close(self._device_handle)

    def _set_reader_or_device_handle(self):
        if self._is_windows:
            self._reader = IR16Capture()
            self._reader.SetupGraphWithBytesCallback(self._callback)
            return

        self._device_handle = UvcDeviceHandle()

    def _start_capture(self) -> None:
        if self._is_windows:
            self._reader.RunGraph()
            return

        result = libuvc.uvc_start_streaming(
            self._device_handle,
            byref(self._stream_ctrl),
            self._callback,
            None,
            0,
        )
        if result < 0:
            raise RuntimeError(f"uvc_start_streaming failed: {result}")

        self._is_streaming = True

    def _stop_capture(self) -> None:
        if self._is_windows:
            self._reader.StopGraph()
            return

        libuvc.uvc_stop_streaming(self._device_handle)

    def _set_emissivity(self) -> None:
        if self._is_windows:
            params = self._device.rad.GetFluxLinearParams()
            if hasattr(params, "sceneEmissivity"):
                field = "sceneEmissivity"
            elif hasattr(params, "SceneEmissivity"):
                field = "SceneEmissivity"
            else:
                raise AttributeError("FluxLinearParams has no scene-emissivity field")
            return
        setattr(params, field, round(float(emissivity) * 8192))
        self._device.rad.SetFluxLinearParams(params)
        self._ensure_open()
        old_value = self.emissivity
        self.emissivity = emissivity
        try:
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
        except Exception:
            self.emissivity = old_value
            raise
        if self._is_streaming:
            self.clear_frames()

    def _set_ffc_mode(self) -> None:
        """Set automatic, manual, or externally controlled FFC shutter mode."""
        self._ensure_open()
        if self._is_windows:
            shutter = self._device.sys.GetFfcShutterModeObj()
            shutter.shutterMode = getattr(self._cci.Sys.FfcShutterMode, mode.upper())
            self._device.sys.SetFfcShutterModeObj(shutter)

        old_mode = self.ffc_mode
        self.ffc_mode = mode
        try:
            setters = {
                "auto": self._uvc.set_auto_ffc,
                "manual": self._uvc.set_manual_ffc,
                "external": self._uvc.set_external_ffc,
            }
            setters[mode](self._device_handle)
        except Exception:
            self.ffc_mode = old_mode
            raise
        if self._is_streaming:
            self.clear_frames()

    def _perform_manual_ffc(self) -> None:
        if self._is_windows:
            self._device.sys.RunFFCNormalization()
        else:
            self._uvc.perform_manual_ffc(self._device_handle)
        if self._is_streaming:
            self.clear_frames()

    def _set_gain_mode(self) -> None:
        if self._is_windows:
            self._device.sys.SetGainMode(getattr(CCI.Sys.GainMode, self.gain_mode.upper()))

        old_mode = self.gain_mode
        self.gain_mode = mode
        try:
            if self._is_windows:
                self._windows_camera.set_gain_mode(mode)
                return

            setters = {
                "high": self._uvc.set_gain_high,
                "low": self._uvc.set_gain_low,
                "auto": self._uvc.set_gain_auto,
            }
            setters[mode](self._device_handle)
        except Exception:
            self.gain_mode = old_mode
            raise
        if self._is_streaming:
            self.clear_frames()

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
