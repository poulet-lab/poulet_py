try:
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor, wait
    from secrets import choice
    from threading import Event
    from typing import Literal, Self

    from prompt_toolkit import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.shortcuts.progress_bar import ProgressBar
    from pydantic import BaseModel, Field, PrivateAttr, model_validator

    from poulet_py import (
        LOGGER,
        BaseSink,
        BaseSource,
        BaseStimulus,
        BaseTrigger,
        EventBus,
        precise_sleep,
        repeat,
    )

except ImportError as e:
    msg = """
Missing 'experiment' module. Install options:
- Dedicated:    pip install poulet_py[exp]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class ExperimentTrial(BaseModel):
    name: str | None = Field(default=None, description="Name of the trial")
    stimuli: BaseStimulus | Sequence[BaseStimulus] = Field(..., description="Stimuli for the trial")

    @model_validator(mode="after")
    def validate_stimuli(self):
        if not isinstance(self.stimuli, Sequence):
            self.stimuli = [self.stimuli]
        return self


class ExperimentBlock(BaseModel):
    name: str | None = Field(default=None, description="Name of the block")
    trials: Sequence[ExperimentTrial] = Field(..., description="Trials in the block")

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


class ExperimentRuntime(BaseModel):
    name: str = Field(..., description="Name of the experiment")
    blocks: Sequence[ExperimentBlock] = Field(..., description="Blocks in the experiment")
    block_repetitions: int = Field(
        default=1, ge=1, description="Number of times to repeat each block"
    )
    block_order: Literal["random", "sequential"] = Field(
        default="sequential", description="Order of blocks"
    )
    isi: int | Sequence[int] = Field(default=0, description="Inter-stimulus interval")

    sources: Sequence[BaseSource] = Field(..., description="Sources for the experiment")
    sinks: Sequence[BaseSink] = Field(..., description="Sinks for the experiment")

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _is_open: bool = PrivateAttr(default=False)
    _started: Event = PrivateAttr(default_factory=Event)
    _paused: Event = PrivateAttr(default_factory=Event)
    _aborted: Event = PrivateAttr(default_factory=Event)
    _stopped: Event = PrivateAttr(default_factory=Event)

    _key_bindings: KeyBindings = PrivateAttr()

    @model_validator(mode="after")
    def validate_isi(self) -> Self:
        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self

    @staticmethod
    def get_isi(isi):
        if isinstance(isi, Sequence):
            return choice(isi)
        return isi

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus):
        if self._is_open:
            msg = "Cannot change bus while experiment is open"
            raise RuntimeError(msg)
        self._bus = value

    def open(self):
        if self._is_open:
            return

        self.bus.open()

        for source in self.sources:
            source.bus = self.bus
            source.open()

        for sink in self.sinks:
            sink.bus = self.bus
            sink.open()

        self._key_bindings = self._create_key_bindings()

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        self._is_open = False

        del self._key_bindings

        self._started.clear()
        self._paused.clear()
        self._aborted.clear()
        self._stopped.clear()

        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.close()

        self.bus.close()

    def run(self):
        self._ensure_open()
        bottom_toolbar = HTML(
            " <b>[enter]</b> Start Experiment <b>[esc]</b> Pause / Resume <b>[ctrl-x]</b> Abort."
        )

        if self._aborted.is_set():
            return

        blocks = self._expand()
        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            with ProgressBar(key_bindings=self._key_bindings, bottom_toolbar=bottom_toolbar) as pb:
                with patch_stdout():
                    self._wait_not_started()

                    for block, trials in pb(blocks, total=len(blocks), label="Blocks"):
                        for trial_idx, trial in pb(
                            enumerate(trials), total=len(trials), label="Trial"
                        ):
                            if self._aborted.is_set():
                                break

                            self._wait_paused()

                            if block.trigger and not block.trigger.wait():
                                if block.trigger_policy == "skip":
                                    LOGGER.warning(
                                        f"Trigger failed for trial {trial.name} in block {block.name}, skipping trial."
                                    )
                                    continue

                                msg = f"Trigger failed for trial {trial.name} in block {block.name}"
                                raise RuntimeError(msg)

                            st_info = {
                                f"{type(st).__name__}": st.model_dump(
                                    exclude_unset=True, exclude_none=True
                                )
                                for st in trial.stimuli
                            }
                            LOGGER.info("Trial %d: %s", trial_idx, st_info)
                            futures = [executor.submit(s.fire, trial.stimuli) for s in self.sources]
                            wait(futures)

                            isi = self.get_isi(block.isi or self.isi)
                            precise_sleep(isi / 1000.0)

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.Enter, eager=True)
        def _(event):
            if not self._started.is_set():
                self._started.set()
                LOGGER.info("Experiment started...")

        @kb.add(Keys.Escape, eager=True)
        def _(event):
            if self._started.is_set():
                if self._paused.is_set():
                    self._paused.clear()
                    LOGGER.info("RESUMED")
                else:
                    self._paused.set()
                    LOGGER.info("PAUSED")

        @kb.add(Keys.ControlX, eager=True)
        def _(event):
            LOGGER.info("ABORTING...")
            self._aborted.set()

        return kb

    def _wait_not_started(self):
        while not self._started.is_set() and not self._aborted.is_set():
            self._started.wait(0.1)

    def _wait_paused(self):
        while self._paused.is_set() and not self._aborted.is_set():
            self._paused.wait(0.1)

    def _expand(self):
        blocks = repeat(self.blocks, self.block_repetitions, mode=self.block_order)

        return [
            (block, repeat(block.trials, block.trial_repetitions, mode=block.trial_order))
            for block in blocks
        ]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
