try:
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor, wait
    from secrets import choice
    from typing import Literal, Self

    from pydantic import BaseModel, Field, PrivateAttr, model_validator
    from tqdm.auto import tqdm

    from poulet_py import BaseSink, BaseSource, BaseStimulus, BaseTrigger, EventBus, repeat, LOGGER

except ImportError as e:
    msg = """
Missing 'experiment' module. Install options:
- Dedicated:    pip install poulet_py[exp]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class ExperimentTrial(BaseModel):
    stimuli: BaseStimulus | Sequence[BaseStimulus]


class ExperimentBlock(BaseModel):
    name: str = Field(...)
    trials: Sequence[ExperimentTrial] = Field(...)
    trial_repetitions: int = Field(default=1, ge=1)
    trial_order: Literal["random", "sequential"] = Field(default="random")
    trigger: BaseTrigger | None = Field(default=None)
    trigger_policy: Literal["abort", "skip"] = "abort"
    isi: int | Sequence[int] = Field(default=0)

    @model_validator(mode="after")
    def validate_isi(self) -> Self:
        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self


class ExperimentRuntime(BaseModel):
    name: str = Field(...)
    blocks: Sequence[ExperimentBlock] = Field(...)
    block_repetitions: int = Field(default=1, ge=1)
    block_order: Literal["random", "sequential"] = Field(default="sequential")
    isi: int | Sequence[int] = Field(default=0)

    sources: Sequence[BaseSource] = Field(...)
    sinks: Sequence[BaseSink] = Field(...)

    bus: EventBus = Field(default_factory=EventBus)

    _open: bool = PrivateAttr(False)

    @model_validator(mode="after")
    def validate_isi(self) -> Self:
        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self

    def open(self):
        self.bus.open()

        for source in self.sources:
            source.open(self.bus)

        for sink in self.sinks:
            sink.open(self.bus)

        self._open = True

    def close(self):
        self._open = False

        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.close()

        self.bus.close()

    def run(self):
        if not self._open:
            msg = "Experiment Runtime should open first"
            raise RuntimeError(msg)

        blocks: Sequence[ExperimentBlock] = repeat(
            self.blocks, self.block_repetitions, mode=self.block_order
        )

        for block in blocks:
            block.trials = repeat(block.trials, block.trial_repetitions, mode=block.trial_order)

        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            for i, block in enumerate(tqdm(blocks, desc="Block", smoothing=True, position=0)):
                for j, trial in enumerate(
                    tqdm(block.trials, desc="Trial", smoothing=True, position=1, leave=False)
                ):

                    if block.trigger and not block.trigger.wait():
                        msg = "Trigger failed"
                        raise RuntimeError(msg)

                    isi = ExperimentRuntime.get_isi(block.isi if block.isi != 0 else self.isi)
                    stimuli = (
                        trial.stimuli if isinstance(trial.stimuli, Sequence) else [trial.stimuli]
                    )

                    for st in stimuli:
                        st._isi = isi

                    futures = []
                    for source in self.sources:
                        futures.append(executor.submit(source.fire, stimuli))

                    wait(futures)

                    for f in futures:
                        f.result()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def get_isi(isi: int | Sequence[int]) -> int:
        """Get the inter-stimulus period (random if list provided)."""
        if isinstance(isi, range):
            return choice(list(isi))

        if isinstance(isi, Sequence):
            return choice(isi)

        return isi
