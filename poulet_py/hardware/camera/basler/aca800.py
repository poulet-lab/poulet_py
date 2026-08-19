try:
    from enum import StrEnum

    from pydantic import Field

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


class ACA800PixelType(PixelTypeMixIn, StrEnum):
    MONO_8 = "Mono8"
    MONO_10 = "Mono10"
    BAYER_BG_8 = "BayerBG8"
    BAYER_BG_10 = "BayerBG10"
    BAYER_BG_10_PACKED = "BayerBG10Packed"
    YUV_422_PACKED = "YUV422Packed"
    YUV_422_YUYV_PACKED = "YUV422_YUYV_Packed"

    def to_numpy(self) -> str:
        if self in (self.MONO_8, self.BAYER_BG_8, self.YUV_422_PACKED, self.YUV_422_YUYV_PACKED):
            return "uint8"

        if self in (self.MONO_10, self.BAYER_BG_10, self.BAYER_BG_10_PACKED):
            return "uint16"

        return "O"


class ACA800(_GenericBaslerCamera[ACA800PixelType]):
    MODEL = SupportedModels.ACA800

    pixel_type: ACA800PixelType = Field(default=ACA800PixelType.MONO_8)
