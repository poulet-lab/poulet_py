try:
    from collections.abc import Sequence
    from typing import Literal, Self

    from pydantic import BaseModel, Field, model_validator

    from poulet_py import BaseTrigger
    from poulet_py.utils.stimulator import Trial

except ImportError as e:
    raise ImportError("""
Missing 'stimulator' module. Install options:
- Dedicated:    pip install poulet_py[stm]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e


class Block(BaseModel):
    """
    Represents a block of trials within an experiment.

    A block contains one or more trials that are repeated and ordered according
    to specified parameters. ISI and trigger behavior can be configured per block.

    Notes
    -----
    ISI values provided as a range will be converted to a list for random selection.
    """

    name: str | None = Field(default=None, description="Name of the block")
    trials: Sequence[Trial] = Field(..., description="Trials in the block")

    trial_repetitions: int = Field(
        default=1, ge=1, description="Number of times to repeat each trial"
    )
    trial_order: Literal["random", "sequential"] = Field(
        default="random", description="Order of trials"
    )

    trigger: BaseTrigger | None = Field(default=None, description="Trigger for the trial")
    trigger_policy: Literal["abort", "skip"] = Field(
        default="abort",
        description="Policy for handling triggers, abort the current trial or skip it",
    )

    isi: int | Sequence[int] = Field(default=0, description="Inter-stimulus interval")

    @model_validator(mode="after")
    def validate_isi(self) -> Self:
        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self
