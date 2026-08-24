try:
    from typing import Any, Literal

    from pydantic import Field, model_validator

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


class DRV2605Stimulus(BaseStimulus):
    """Validated RTP or ROM-waveform command for DRV2605Source."""

    mode: Literal["rtp", "play_waveform"] = Field(default="rtp")
    duration: int = Field(default=500, ge=1, le=7000)
    drive_voltage: float = Field(default=2.0, gt=0.0, le=5.6)
    waveform: int = Field(default=16, ge=1, le=123)
    repeat_count: int = Field(default=1, ge=0, le=8)

    @model_validator(mode="after")
    def _validate_mode_fields(self):
        if self.mode == "rtp" and not 50 <= self.duration <= 5000:
            raise ValueError("RTP duration must be between 50 and 5000 ms.")
        if self.mode == "play_waveform" and self.repeat_count == 0:
            raise ValueError("play_waveform requires repeat_count between 1 and 8.")
        return self

    def build(self, *args: Any, **kwargs: Any) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "duration": self.duration,
            "drive_voltage": self.drive_voltage,
            "waveform": self.waveform,
            "repeat_count": self.repeat_count,
        }