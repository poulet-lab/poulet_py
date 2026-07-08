try:
    from typing import Any

    from pydantic import Field

    from poulet_py import BaseStimulus

except ImportError as e:
    msg = """
Missing DRV2605L stimulus dependencies.

Install options:
- Dedicated:    pip install poulet_py[stimulus]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DRV2605Stimulus(BaseStimulus):
    """
    Stimulus description for a DRV2605L haptic motor driver.

    This class does not touch hardware. It only describes what should be played.
    The hardware execution is handled by DRV2605Source.
    """

    duration: int = Field(
        default=1000,
        ge=1,
        le=7000,
        description=(
            "Nominal stimulus duration in ms. If repeat_count is not given, "
            "duration // 1000 is used as the number of waveform repetitions, "
            "clamped to 0-7."
        ),
    )

    waveform: int = Field(
        default=16,
        ge=1,
        le=123,
        description="DRV2605L ROM waveform/effect ID. Effect 16 is the 1000 ms alert.",
    )

    repeat_count: int | None = Field(
        default=None,
        ge=0,
        le=7,
        description=(
            "Number of times to place the waveform into the 8-slot waveform "
            "sequencer. Slot 8 is kept as 0 terminator, so valid range is 0-7. "
            "If None, duration // 1000 is used."
        ),
    )

    drive_voltage: float | None = Field(
        default=None,
        ge=0.0,
        le=5.6,
        description=(
            "Optional OD_CLAMP voltage in V. If None, the current DRV2605L "
            "configuration is left unchanged."
        ),
    )

    def build(self, *args: Any, **kwargs: Any) -> dict[str, int | float | None]:
        if self.repeat_count is None:
            repeat_count = max(0, min(int(self.duration // 1000), 7))
        else:
            repeat_count = self.repeat_count

        return {
            "waveform": self.waveform,
            "repeat_count": repeat_count,
            "drive_voltage": self.drive_voltage,
        }