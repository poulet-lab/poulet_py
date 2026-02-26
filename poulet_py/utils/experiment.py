try:
    from concurrent.futures import ThreadPoolExecutor, wait
    from secrets import choice
    from time import sleep
    from typing import Literal

    from pydantic import BaseModel, Field, PrivateAttr
    from tqdm.auto import tqdm

    from poulet_py import (
        BaseSink,
        BaseSource,
        BaseStimulus,
        BaseTrigger,
        EventBus,
        repeat,
    )
except ImportError as e:
    msg = """
Missing 'exp' module. Install options:
- Dedicated:    pip install poulet_py[exp]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class ExperimentTrial(BaseModel):
    stimuli: BaseStimulus | list[BaseStimulus]
    isi: int | list[int] = Field(default=0)


class ExperimentBlock(BaseModel):
    name: str = Field(...)
    trials: list[ExperimentTrial] = Field(...)
    repetitions: int = Field(default=1, ge=1)
    order: Literal["random", "sequential"] = Field(default="random")
    trigger: BaseTrigger | None = Field(default=None)
    trigger_policy: Literal["abort", "skip"] = "abort"
    isi: int | list[int] = Field(default=0)


class ExperimentRuntime(BaseModel):
    name: str = Field(...)
    blocks: list[ExperimentBlock] = Field(...)
    repetitions: int = Field(default=1, ge=1)
    order: Literal["random", "sequential"] = Field(default="sequential")
    isi: int | list[int] = Field(default=0)

    sources: list[BaseSource] = Field(...)
    sinks: list[BaseSink] = Field(...)

    bus: EventBus = Field(default_factory=EventBus)

    _open: bool = PrivateAttr(False)

    def open(self):
        for source in self.sources:
            source.open(self.bus)

        for sink in self.sinks:
            sink.open(self.bus)

    def close(self):
        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.close()

    def run(self):
        if not self._open:
            raise RuntimeError("Experiment Runtime should open first")

        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            blocks: list[ExperimentBlock] = repeat(self.blocks, self.repetitions, mode=self.order)

            for block in tqdm(blocks, desc="Block", smoothing=True):
                trials: list[ExperimentTrial] = repeat(
                    block.trials, block.repetitions, mode=block.order
                )

                for trial in tqdm(trials, desc="Trial", smoothing=True, leave=False):
                    if block.trigger and not block.trigger.wait():
                        msg = "Trigger failed"
                        raise RuntimeError(msg)

                    stimuli = trial.stimuli if isinstance(trial.stimuli, list) else [trial.stimuli]
                    isi = ExperimentRuntime.get_isi(
                        trial.isi if trial.isi != 0 else block.isi if block.isi != 0 else self.isi
                    )

                    futures = []
                    for source in self.sources:
                        futures.append(executor.submit(source.next, stimuli, isi))

                    wait(futures)

                    for f in futures:
                        f.result()

    def __enter__(self):
        self.open()
        self._open = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._open = False
        self.close()

    @staticmethod
    def get_isi(isi) -> int:
        """Get the inter-stimulus period (random if list provided)."""
        if isinstance(isi, list):
            return choice(isi)
        return isi
