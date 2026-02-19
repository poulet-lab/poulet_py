try:
    from secrets import choice
    from time import sleep
    from typing import Literal

    from pydantic import BaseModel, Field

    from poulet_py import (
        LOGGER,
        BaseSink,
        BaseSource,
        BaseStimulus,
        BaseTrigger,
        generate_stimulus_sequence,
    )
except ImportError as e:
    msg = """
Missing 'exp' module. Install options:
- Dedicated:    pip install poulet_py[exp]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class Experiment(BaseModel):
    stimuli: list[BaseStimulus] = Field(...)
    n_repetitions: int = Field(default=1, ge=1)
    stimulus_order: Literal["random", "sequential"] = Field(default="random")
    isi: int | list[int] = Field(default=0)
    trigger: BaseTrigger | None = Field(default=None)
    trigger_policy: Literal["abort", "skip"] = "abort"

    sources: list[BaseSource] = Field(...)
    sinks: list[BaseSink] = Field(...)

    def open(self):

        self._auto_wire()

        for source in self.sources:
            source.open()

        for sink in self.sinks:
            sink.open()

    def close(self):
        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.close()

    def run(self):
        stimuli = generate_stimulus_sequence(
            self.n_repetitions, stimuli_options=self.stimuli, mode=self.stimulus_order
        )

        for stim in stimuli:
            if self.trigger and not self.trigger.wait():
                if self.trigger_policy == "skip":
                    LOGGER.error("Trigger Failed, canceling stimulation")
                    continue
                elif self.trigger_policy == "abort":
                    msg = "Trigger failed — aborting experiment."
                    raise RuntimeError(msg)

            for source in self.sources:
                source.next(stim)

            isi = self._get_isi()

            if isi:
                sleep(isi / 1000)

    def _get_isi(self) -> int:
        """Get the inter-stimulus period (random if list provided)."""
        if isinstance(self.isi, list):
            return choice(self.isi)
        return self.isi

    def _auto_wire(self):

        for source in self.sources:
            # Skip if user already defined routing
            if source._subscribers:
                continue

            for sink in self.sinks:
                source.subscribe(sink)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
