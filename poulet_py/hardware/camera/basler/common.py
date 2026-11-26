try:
    from enum import StrEnum
    from typing import Any, Literal

    from numpy import ndarray
    from pydantic import Field, PrivateAttr
    from pypylon.pylon import (
        GrabStrategy_OneByOne,
        InstantCamera,
        TimeoutHandling_Return,
        TlFactory,
        waitForever,
    )

    from poulet_py import LOGGER
    from poulet_py.hardware import BaseHardware
except ImportError as e:
    raise ImportError(f"""{__name__}:
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class PixelType(StrEnum):
    MONO_8 = "Mono8"
    MONO_10 = "Mono10"

    BAYER_BG_8 = "BayerBG8"
    BAYER_BG_10 = "BayerBG10"
    BAYER_BG_10_PACKED = "BayerBG10Packed"

    YUV_422_PACKED = "YUV422Packed"
    YUV_422_YUYV_PACKED = "YUV422_YUYV_Packed"

    # TODO: add more if needed for other cameras
    # https://docs.baslerweb.com/image-format-converter-vtool#supported-pixel-formats

    def to_numpy(self) -> str:
        if self in (self.MONO_8, self.BAYER_BG_8, self.YUV_422_PACKED, self.YUV_422_YUYV_PACKED):
            return "uint8"

        if self in (self.MONO_10, self.BAYER_BG_10, self.BAYER_BG_10_PACKED):
            return "uint16"

        return "O"


class TriggerType(StrEnum):
    FRAME_START = "FrameStart"
    FRAME_END = "FrameEnd"
    FRAME_ACTIVE = "FrameActive"
    ACQUISITION_START = "AcquisitionStart"
    FRAME_BURST_END = "FrameBurstEnd"
    FRAME_BURST_ACTIVE = "FrameBurstActive"
    EXPOSURE_START = "ExposureStart"
    EXPOSURE_END = "ExposureEnd"
    EXPOPSURE_ACTIVE = "ExposureActive"
    LINE_START = "LineStart"


class TriggerSource(StrEnum):
    INTERNAL = "Internal"
    SOFTWARE = "Software"  # TODO
    # GPIO
    LINE_1 = "Line1"
    LINE_2 = "Line2"
    LINE_3 = "Line3"
    LINE_4 = "Line4"
    # CXP-12 interface card
    CXP_TRIGGER_0 = "CXPTrigger0"
    CXP_TRIGGER_1 = "CXPTrigger1"
    # GMSL host system, more can be available
    LINK_SIGNAL_1 = "LinkSignal1"
    LINK_SIGNAL_2 = "LinkSignal2"
    LINK_SIGNAL_3 = "LinkSignal3"
    LINK_SIGNAL_4 = "LinkSignal4"


class TriggerActivationMode(StrEnum):
    RISING_EDGE = "RisingEdge"
    FALLING_EDGE = "FallingEdge"
    ANY_EDGE = "AnyEdge"
    LEVEL_HIGH = "LevelHigh"
    LEVEL_LOW = "LevelLow"


class AutoFunctionProfile(StrEnum):
    GAIN_MINIMUM = "GainMinimum"
    EXPOSURE_MINIMUM = "ExposureMinimum"
    GAIN_MINIMUM_QUICK = "GainMinimumQuick"
    EXPOSURE_MINIMUM_QUICK = "ExposureMinimumQuick"
    SMART = "Smart"
    ANTIFLICKER_50HZ = "AntiFlicker50Hz"
    ANTIFLICKER_60HZ = "AntiFlicker60Hz"


class BaslerCamera(BaseHardware):
    device_index: int = Field(default=0, description="")
    trigger_type: TriggerType = Field(default=TriggerType.FRAME_START)
    trigger_source: TriggerSource = Field(default=TriggerSource.INTERNAL)
    trigger_activation: TriggerActivationMode = Field(default=TriggerActivationMode.RISING_EDGE)
    timeout: int | None = Field(default=None, description="handle timeout in ms")
    pixel_type: PixelType = Field(default=PixelType.MONO_8)
    fps: int = Field(default=30)
    resolution: tuple[int, int] | None = Field(
        default=None,
        description=(
            "Optional camera-native ROI/resolution as (width, height). "
            "Implemented through Basler subarray properties, not software resizing."
        ),
    )
    offset: tuple[int, int] | Literal["center"] = Field(
        default="center",
        description=("Optional ROI offset. If None, ROI is centered"),
    )
    auto_exposure: Literal["auto", "once"] | None = Field(default=None)
    exposure: float = Field(default=10000, description="in ms", ge=1, le=10000)
    auto_gain: Literal["auto", "once"] | None = Field(default=None)
    gain: float = Field(default=200)
    auto_function_profile: AutoFunctionProfile = Field(default=AutoFunctionProfile.EXPOSURE_MINIMUM)
    precise_time_protocol: bool = Field(default=True)
    # TODO add more features https://docs.baslerweb.com/features

    _tl_factory: TlFactory = PrivateAttr()
    _devices: list[Any] = PrivateAttr()
    _camera: InstantCamera = PrivateAttr()

    @staticmethod
    def get_available_devices() -> list[dict[str, int | str]]:
        tl_factory = TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        res = []
        for idx, device in enumerate(devices):
            res.append({"device_id": idx, "name": device.GetFriendlyName()})
        return res

    def _open(self) -> None:
        self._set_tl_factory()
        self._set_camera()
        self._set_params()

    def _close(self) -> None:
        self._release_camera()
        self._release_tl_factory()

    def info(self) -> dict[str, Any]:
        self._ensure_open()

        return {
            "device_id": self.device_index,
            "name": self._devices[self.device_index].GetFriendlyName(),
        }

    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]:
        width, height = self._camera.Width.Value, self._camera.Height.Value
        return [("data", self.pixel_type.to_numpy(), (height, width))]

    def _start_capture(self) -> None:
        self._camera.StartGrabbing(GrabStrategy_OneByOne)

    def _capture_status(self) -> bool:
        return self._camera.IsGrabbing()

    def _stop_capture(self) -> None:
        self._camera.StopGrabbing()

    def _set_tl_factory(self) -> None:
        self._tl_factory = TlFactory.GetInstance()

        self._devices = self._tl_factory.EnumerateDevices()
        if len(self._devices) == 0:
            raise Exception(f"{type(self).__name__} No camera found.")

    def _release_tl_factory(self) -> None:
        if hasattr(self, "_devices"):
            del self._devices
        if hasattr(self, "_tl_factory"):
            del self._tl_factory

    def _set_camera(self) -> None:
        if self.device_index >= len(self._devices):
            raise IndexError(
                f"{type(self).__name__} Device index {self.device_index} out of range."
                "Use get_available_devices() to list available cameras."
            )

        self._camera = InstantCamera(
            self._tl_factory.CreateDevice(self._devices[self.device_index])
        )
        self._camera.Open()

    def _release_camera(self) -> None:
        if hasattr(self, "_camera"):
            self._camera.Close()
            del self._camera

    def _set_params(self) -> None:
        self._camera.GevIEEE1588.Value = self.precise_time_protocol

        self._set_resolution_offset()
        self._set_exposure_gain()
        self._set_trigger()

    def _set_resolution_offset(self) -> None:
        if self.resolution:
            min_limits = (self._camera.Width.Min, self._camera.Height.Min)
            max_limits = (self._camera.Width.Max, self._camera.Height.Max)
            if not (
                min_limits[0] <= self.resolution[0] <= max_limits[0]
                and min_limits[1] <= self.resolution[1] <= max_limits[1]
            ):
                raise AttributeError(
                    f"{type(self).__name__} Resolution {self.resolution} is out of limits, "
                    f"Lower: {min_limits}, Upper: {max_limits}"
                )
            self._camera.Width.Value = self.resolution[0]
            self._camera.Height.Value = self.resolution[1]
        else:
            self._camera.Width.Value = self._camera.Width.Max
            self._camera.Height.Value = self._camera.Height.Max

        if self.offset == "center":
            self._camera.CenterX.Value = True
            self._camera.CenterY.Value = True
        else:
            self._camera.OffsetX.Value = self.offset[0]
            self._camera.OffsetY.Value = self.offset[1]

    def _set_exposure_gain(self) -> None:
        try:
            self._camera.AutoGainRawLowerLimit.Value = self._camera.AutoGainRawLowerLimit.Min
            self._camera.AutoGainRawUpperLimit.Value = self._camera.AutoGainRawUpperLimit.Max
        except Exception as e:
            LOGGER.warning(f"{type(self).__name__} Could not setup gain limits: {e}")

        try:
            self._camera.AutoExposureTimeLowerLimit.Value = (
                self._camera.AutoExposureTimeLowerLimit.Min
            )
            self._camera.AutoExposureTimeUpperLimit.Value = (
                self._camera.AutoExposureTimeUpperLimit.Max
            )
        except Exception as e:
            LOGGER.warning(f"{type(self).__name__} Could not setup exposure limits: {e}")

        auto_target = (self._camera.AutoTargetValue.Min + self._camera.AutoTargetValue.Max) // 2
        self._camera.AutoTargetValue.Value = auto_target
        self._camera.AutoFunctionAOISelector.Value = "AOI1"
        self._camera.AutoFunctionAOIUsageIntensity.Value = True
        self._camera.AutoFunctionProfile.Value = self.auto_function_profile.value
        try:
            if self.trigger_source != TriggerSource.INTERNAL:
                self._camera.ExposureMode.Value = "Timed"

            if self.auto_exposure == "auto":
                self._camera.ExposureAuto.Value = "Continuous"
            elif self.auto_exposure == "once":
                self._camera.ExposureAuto.Value = "Once"
            else:
                self._camera.ExposureMode.Value = "Timed"
                self._camera.ExposureAuto.Value = "Off"
                self._camera.ExposureTimeAbs.Value = self.exposure
        except Exception as e:
            LOGGER.warning(f"{type(self).__name__} Could not setup exposure: {e}")

        try:
            if self.auto_gain == "auto":
                self._camera.GainAuto.Value = "Continuous"
            elif self.auto_gain == "once":
                self._camera.GainAuto.Value = "Once"
            else:
                self._camera.GainAuto.Value = "Off"
                self._camera.GainRaw.Value = self.gain
        except Exception as e:
            LOGGER.warning(f"{type(self).__name__} Could not setup gain: {e}")

    def _set_trigger(self) -> None:
        if self.trigger_source == TriggerSource.INTERNAL:
            self._camera.TriggerSelector.Value = self.trigger_type.value
            self._camera.TriggerMode.Value = "Off"

        elif self.trigger_source not in (TriggerSource.INTERNAL, TriggerSource.SOFTWARE):
            if self.trigger_source in (
                TriggerSource.LINE_1,
                TriggerSource.LINE_2,
                TriggerSource.LINE_3,
                TriggerSource.LINE_4,
            ):
                self._camera.LineSelector.Value = self.trigger_source.value
                self._camera.LineMode.Value = "Input"

            self._camera.TriggerSelector.Value = self.trigger_type.value
            self._camera.TriggerMode.Value = "On"
            self._camera.TriggerSource.Value = self.trigger_source.value
            self._camera.TriggerActivation.Value = self.trigger_activation.value

        if self.trigger_source == TriggerSource.INTERNAL or self.trigger_type in (
            TriggerType.FRAME_BURST_ACTIVE,
            TriggerType.FRAME_BURST_END,
        ):
            self._camera.AcquisitionFrameRateEnable.Value = True
            self._camera.AcquisitionFrameRateAbs.Value = self.fps

    def _acquire_sample(self) -> ndarray | None:
        if not self._capture_status():
            return None

        result = self._camera.RetrieveResult(
            self.timeout if self.timeout is not None else waitForever, TimeoutHandling_Return
        )

        sample = None
        if result and result.GrabSucceeded():
            sample = result.GetArray()

        result.Release()

        return sample
