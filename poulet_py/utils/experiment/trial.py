try:
    from collections.abc import Sequence
    from typing import Self

    from pydantic import BaseModel, Field, model_validator

    from poulet_py import BaseStimulus

except ImportError as e:
    raise ImportError("""
Missing 'stimulator' module. Install options:
- Dedicated:    pip install poulet_py[stm]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e


class Trial(BaseModel):
    """
    Represents a single trial within an experiment block.

    A trial consists of one or more stimuli that are presented simultaneously
    during the trial execution.
    """

    name: str | None = Field(default=None, description="Name of the trial")
    stimuli: BaseStimulus | Sequence[BaseStimulus] = Field(..., description="Stimuli for the trial")
    isi: int | Sequence[int] = Field(default=0, description="Inter-stimulus interval")

    @model_validator(mode="after")
    def validate_stimuli(self) -> Self:
        if not isinstance(self.stimuli, Sequence):
            self.stimuli = [self.stimuli]

        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self
