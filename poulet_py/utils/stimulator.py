"""
Stimulator management classes for defining and running experimental protocols.

Classes
-------
StimulatorTrial
    Container for a single trial.
StimulatorBlock
    Container for a block of trials with repetition and ordering controls.
StimulatorRuntime
    Main experiment orchestrator managing blocks, sources, sinks, and execution.
"""

try:
    from collections.abc import Callable, Sequence
    from secrets import choice
    from threading import Barrier, Event
    from time import time_ns
    from types import TracebackType
    from typing import Literal, Self

    from prompt_toolkit import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.shortcuts.progress_bar import ProgressBar
    from pydantic import BaseModel, Field, PrivateAttr, model_validator

    from poulet_py import (
        LOGGER,
        BaseEvent,
        BaseSink,
        BaseSource,
        BaseStimulus,
        BaseTrigger,
        EventBus,
        EventHandler,
        precise_sleep,
        repeat,
    )

except ImportError as e:
    raise ImportError("""
Missing 'stimulator' module. Install options:
- Dedicated:    pip install poulet_py[stm]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e


class StimulatorTrial(BaseModel):
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


class StimulatorBlock(BaseModel):
    """
    Represents a block of trials within an experiment.

    A block contains one or more trials that are repeated and ordered according
    to specified parameters. ISI and trigger behavior can be configured per block.

    Notes
    -----
    ISI values provided as a range will be converted to a list for random selection.
    """

    name: str | None = Field(default=None, description="Name of the block")
    trials: Sequence[StimulatorTrial] = Field(..., description="Trials in the block")

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


class ResponceMonitor(EventHandler):
    event_callable: Callable[[BaseEvent], bool] = Field(...)
    trial_callable: Callable[[StimulatorTrial], StimulatorTrial] | None = Field(default=None)

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _is_open: bool = PrivateAttr(default=False)
    _should_stop: bool = PrivateAttr(default=False)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus) -> None:
        if self._is_open:
            raise RuntimeError(f"Cannot change bus while {type(self).__name__} is open")

        self._bus = value
        self._external_bus = True

    def open(self) -> None:
        if self._is_open:
            return

        if not self._external_bus:
            self.bus.open()

        self.bus.subscribe(self)

        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        if not self._external_bus:
            self.bus.close()

        self._is_open = False

    def on_event(self, event: BaseEvent) -> None:
        self._ensure_open()

        stop = self.event_callable(event)
        if not self._should_stop and stop:
            self._should_stop = stop

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"{type(self)} need to be opened first")

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class StimulatorRuntime(BaseModel):
    """
    Main experiment orchestrator managing execution, data sources, and sinks.

    Controls the entire experiment lifecycle including opening/closing connections,
    running trial blocks, handling triggers, and managing user interaction through
    keyboard shortcuts.

    Examples
    --------
    >>> stm = StimulatorRuntime(
    ...     name="My Stimulator", blocks=[my_block], sources=[my_source], sinks=[my_sink]
    ... )
    >>> with stm:
    ...     stm.run()
    """

    name: str = Field(..., description="Name of the stimulator")
    blocks: Sequence[StimulatorBlock] = Field(..., description="Blocks in the stimulator")
    block_repetitions: int = Field(
        default=1, ge=1, description="Number of times to repeat each block"
    )
    block_order: Literal["random", "sequential"] = Field(
        default="sequential", description="Order of blocks"
    )
    isi: int | Sequence[int] = Field(default=0, description="Inter-stimulus interval")

    sources: Sequence[BaseSource] = Field(..., description="Sources for the stimulator")
    sinks: Sequence[BaseSink] = Field(..., description="Sinks for the stimulator")
    responce_monitor: ResponceMonitor | None = Field(default=None)

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _is_open: bool = PrivateAttr(default=False)
    _started: Event = PrivateAttr(default_factory=Event)
    _paused: Event = PrivateAttr(default_factory=Event)
    _aborted: Event = PrivateAttr(default_factory=Event)
    _stopped: Event = PrivateAttr(default_factory=Event)
    _barrier: Barrier | None = PrivateAttr(default=None)

    _key_bindings: KeyBindings = PrivateAttr()
    _start_time_of_experiment: int = PrivateAttr(default_factory=time_ns)

    @model_validator(mode="after")
    def validate_isi(self) -> Self:
        if isinstance(self.isi, range):
            self.isi = list(self.isi)
        return self

    @staticmethod
    def get_isi(isi: Sequence[int] | int) -> int:
        """
        Get inter-stimulus interval value.

        If ISI is a sequence, randomly selects one value. Otherwise returns
        the single value directly.

        Parameters
        ----------
        isi : int or Sequence of int
            ISI specification, either a single value or sequence of values.

        Returns
        -------
        int
            Selected ISI value in milliseconds.
        """
        if isinstance(isi, Sequence):
            return choice(isi)
        return isi

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus) -> None:
        if self._is_open:
            raise RuntimeError("Cannot change bus while stimulator is open")

        self._bus = value
        self._external_bus = True

    def open(self) -> None:
        """
        Open all connections and prepare stimulator for execution.

        Initializes the event bus (if not externally provided), opens all
        source and sink connections, and sets up keyboard bindings for
        user interaction control.

        Notes
        -----
        Idempotent - calling multiple times has no additional effect after
        first successful open.
        """
        if self._is_open:
            return

        if not self._external_bus:
            self.bus.open()

        self._barrier = Barrier(len(self.sources))

        for source in self.sources:
            source.bus = self.bus
            source.open()
            source.barrier = self._barrier

        for sink in self.sinks:
            sink.bus = self.bus
            sink.open()

        if self.responce_monitor:
            self.responce_monitor.bus = self.bus
            self.responce_monitor.open()

        self._key_bindings = self._create_key_bindings()

        self._is_open = True

    def close(self) -> None:
        """
        Close all connections and clean up stimulator resources.

        Closes all sinks and sources, clears event states, and closes
        the event bus if it was internally created.
        """
        if not self._is_open:
            return

        self._is_open = False

        del self._key_bindings

        self._started.clear()
        self._paused.clear()
        self._aborted.clear()
        self._stopped.clear()

        if self.responce_monitor:
            self.responce_monitor.close()

        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.barrier = None
            source.close()

        if not self._external_bus:
            self.bus.close()

    def run(self) -> None:
        """
        Execute the stimulator protocol.

        Expands blocks and trials according to repetition and ordering settings,
        then iterates through each trial, handling triggers, ISI timing,
        and user interaction (pause/resume/abort).

        Each trial's stimuli are presented simultaneously through all sources
        using a thread pool executor.

        Raises
        ------
        RuntimeError
            If stimulator is not opened before running, or if a trigger fails
            with abort policy.
        """
        self._ensure_open()

        if self._aborted.is_set():
            return

        blocks = self._expand_trials()

        with ProgressBar(
            key_bindings=self._key_bindings, bottom_toolbar=self._generate_bottom_toolbar
        ) as pb:
            with patch_stdout():
                self._wait_not_started()

                for block, trials in pb(blocks, total=len(blocks), label="Blocks"):
                    for trial_idx, trial in pb(
                        enumerate(trials), total=len(trials), label="Trials"
                    ):
                        current_trial = trial

                        if self._aborted.is_set() or (
                            self.responce_monitor and self.responce_monitor.should_stop
                        ):
                            break

                        if self.responce_monitor and self.responce_monitor.trial_callable:
                            current_trial = self.responce_monitor.trial_callable(trial)

                        if not current_trial:
                            continue

                        if block.trigger and not block.trigger.wait():
                            if block.trigger_policy == "skip":
                                LOGGER.warning(
                                    f"Trigger failed for trial {current_trial.name} in block {block.name}, skipping trial."
                                )
                                continue

                            raise RuntimeError(
                                f"Trigger failed for trial {current_trial.name} in block {block.name}"
                            )

                        current_trial.isi = self.get_isi(current_trial.isi or block.isi or self.isi)

                        trial_info = current_trial.model_dump(exclude_unset=True, exclude_none=True)
                        LOGGER.info("Trial %d: %s", trial_idx, trial_info)

                        for s in self.sources:
                            s.fire(current_trial.stimuli)

                        for s in self.sources:
                            s.wait()

                        precise_sleep(current_trial.isi / 1000.0)
                        self._wait_paused()

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"{type(self).__name__} need to be opened first")

    def _generate_bottom_toolbar(self) -> HTML:
        shortcuts = "<b>[ctrl-x]</b> Abort"
        status = "Idle"

        if not self._started.is_set():
            shortcuts = "<b>[enter]</b> Start " + shortcuts
        elif self._started.is_set() and not self._paused.is_set():
            shortcuts = "<b>[esc]</b> Pause " + shortcuts
            status = "Running"
        elif self._started.is_set() and self._paused.is_set():
            shortcuts = "<b>[esc]</b> Resume " + shortcuts
            status = "Paused"

        return HTML(
            f"<b>Stimulator:</b> {self.name} | <b>Status:</b> {status} | <b>Shortcuts:</b> {shortcuts}"
        )

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.Enter, eager=True)
        def _(event) -> None:
            if not self._started.is_set():
                self._started.set()

        @kb.add(Keys.Escape, eager=True)
        def _(event) -> None:
            if self._started.is_set():
                if self._paused.is_set():
                    self._paused.clear()
                else:
                    self._paused.set()

        @kb.add(Keys.ControlX, eager=True)
        def _(event) -> None:
            self._aborted.set()

        return kb

    def _wait_not_started(self) -> None:
        while not self._started.is_set() and not self._aborted.is_set():
            self._started.wait(0.1)

    def _wait_paused(self) -> None:
        while self._paused.is_set() and not self._aborted.is_set():
            self._paused.wait(0.1)

    def _expand_trials(self) -> Sequence[tuple[StimulatorBlock, Sequence[StimulatorTrial]]]:
        blocks = repeat(self.blocks, self.block_repetitions, mode=self.block_order)

        return [
            (block, repeat(block.trials, block.trial_repetitions, mode=block.trial_order))
            for block in blocks
        ]

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
