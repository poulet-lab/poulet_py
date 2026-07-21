try:
    from typing import Any, Literal

    from pydantic import Field

    from poulet_py import BaseStimulus

except ImportError as e:
    msg = """
Missing DRV2605L stimulus dependencies.

Install options:
- Dedicated: pip install poulet_py[stimulus]
- Module:    pip install poulet_py[io]
- Full:      pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DRV2605Stimulus(BaseStimulus):
    """
    DRV2605L stimulus description.

    RTP is the default mode. ``drive_voltage`` is converted to the unsigned
    RTP_INPUT command. ``DRV2605Source.maximum_voltage`` independently sets
    OD_CLAMP and defaults to 3.3 V.
    """

    mode: Literal["rtp", "play_waveform"] = Field(
        default="rtp",
        description="Use RTP input or the internal ROM waveform sequencer.",
    )

    specify_mode: bool = Field(
        default=False,
        description=(
            "Force mode-specific registers to be rewritten for this stimulus. "
            "Otherwise cached configuration is reused."
        ),
    )

    duration: int = Field(
        default=500,
        ge=50,
        le=10000,
        description="Motor command duration in ms. RTP mode requires 50-5000 ms.",
    )

    drive_voltage: float = Field(
        default=5.6,
        ge=0.0,
        le=5.6,
        description=(
            "Requested RTP command on the existing 0-5.6 V API scale. "
            "OD_CLAMP independently limits maximum output."
        ),
    )

    waveform: int = Field(
        default=16,
        ge=1,
        le=123,
        description="ROM waveform ID for play_waveform mode.",
    )

    repeat_count: int | None = Field(
        default=1,
        ge=0,
        le=7,
        description=(
            "Number of waveform-sequencer slots. If None, duration // 1000 "
            "is used and clamped to 0-7."
        ),
    )

    def build(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, int | float | str | bool]:
        del args, kwargs
        repeat_count = (
            max(0, min(int(self.duration // 1000), 7))
            if self.repeat_count is None
            else self.repeat_count
        )

        return {
            "mode": self.mode,
            "specify_mode": self.specify_mode,
            "duration": self.duration,
            "drive_voltage": self.drive_voltage,
            "waveform": self.waveform,
            "repeat_count": repeat_count,
        }
