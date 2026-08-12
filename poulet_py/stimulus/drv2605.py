try:
    from typing import Any, Self

    from pydantic import Field

    from poulet_py import BaseStimulus

except ImportError as e:
    raise ImportError(
        """
Missing DRV2605L stimulus dependencies.

Install options:
- Dedicated: pip install poulet_py[stimulus]
- Module:    pip install poulet_py[io]
- Full:      pip install poulet_py[all]
"""
    ) from e


class DRV2605WaveformStimulus(BaseStimulus):
    wait: bool = Field(default=False)
    waveform: int | list[int] = Field(default=0x00, ge=0x01, le=0x7F)
    overdrive_time_offset: int = Field(default=0x00, ge=0x00, le=0xFF)
    sustain_positive_time_offset: int = Field(default=0x00, ge=0x00, le=0xFF)
    sustain_negative_time_offset: int = Field(default=0x00, ge=0x00, le=0xFF)
    break_time_offset: int = Field(default=0x00, ge=0x00, le=0xFF)

    def build(self, *args: Any, **kwargs: Any) -> Self:
        return self


class DRV2605RTPStimulus(BaseStimulus):
    duration: int = Field(default=500, ge=1, le=5000)
    voltage: float = Field(default=2.0, gt=0.0, le=5.6)
    repeat_count: int = Field(default=1, ge=0, le=8)

    def build(self, *args: Any, **kwargs: Any) -> Self:
        return self
