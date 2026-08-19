try:
    from enum import StrEnum

    from pydantic import Field

    from poulet_py.hardware.camera.basler.aca800 import ACA800
    from poulet_py.hardware.camera.basler.common import (
        PixelTypeMixIn,
        SupportedModels,
        _GenericBaslerCamera,
    )
except ImportError as e:
    raise ImportError("""
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class PixelType(PixelTypeMixIn, StrEnum):
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


class Basler(_GenericBaslerCamera[PixelType]):
    pixel_type: PixelType = Field(default=PixelType.MONO_8)

    def __new__(cls, model: SupportedModels = SupportedModels.OTHER, **kwargs):
        if cls is Basler:
            if ACA800.MODEL == model:
                return ACA800(model=model, **kwargs)

        return super().__new__(cls)
