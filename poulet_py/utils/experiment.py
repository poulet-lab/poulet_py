try:
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor
    from secrets import choice
    from threading import Event
    from typing import Literal, Self

    from pydantic import BaseModel, Field, PrivateAttr, model_validator
    from pynput.keyboard import Key, KeyCode, Listener
    from tqdm.auto import tqdm

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


class ExperimentController(BaseModel):
    _keyboard_listener: Listener = PrivateAttr()
    _started: Event = PrivateAttr(default_factory=Event)
    _paused: Event = PrivateAttr(default_factory=Event)
    _aborted: Event = PrivateAttr(default_factory=Event)
    _stopped: Event = PrivateAttr(default_factory=Event)
    _ctrl_key_pressed: bool = PrivateAttr(default=False)

    def open(self):
        self._keyboard_listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._keyboard_listener.start()

    def close(self):
        self._keyboard_listener.stop()
        del self._keyboard_listener
        self._started.clear()
        self._paused.clear()
        self._aborted.clear()
        self._stopped.clear()

    @property
    def started(self):
        return self._started

    @property
    def paused(self):
        return self._paused

    @property
    def aborted(self):
        return self._aborted

    @property
    def stopped(self):
        return self._stopped

    def _on_press(self, key):
        if key in (Key.ctrl_l, Key.ctrl_r):
            self._ctrl_key_pressed = True
            return

        if key == Key.enter:
            self._started.set()
            return

        if key == Key.esc:
            if not self._started.is_set():
                self._aborted.set()
            else:
                if self._paused.is_set():
                    self._paused.clear()
                    LOGGER.info("Experiment RESUMED. Press ESC to pause/resume, Ctrl+Q to abort.")
                else:
                    self._paused.set()
                    LOGGER.info(
                        "Experiment PAUSED. Press ESC to pause/resume, F1 for menu, Ctrl+Q to abort."
                    )
            return

        if self._ctrl_key_pressed and key == KeyCode.from_char("q"):
            LOGGER.info("Abort requested")
            self._aborted.set()

        # if key == Key.f1:
        #     if (not self._started.is_set()) or (self._started.is_set() and self._paused.is_set()):
        #         self._show_interactive_info()
        #     return

        # if self._control_mode and not self._in_menu:
        #     key_str = self._key_to_string(key)
        #     if key_str and self._ctrl_key_pressed:
        #         full_key = f"ctrl+{key_str}"
        #         if full_key in self._active_source_controls:
        #             LOGGER.info(f"Executing source control: {full_key}")
        #             self._active_source_controls[full_key]()
        #             return
        #     elif key_str and key_str in self._active_source_controls:
        #         LOGGER.info(f"Executing source control: {key_str}")
        #         self._active_source_controls[key_str]()
        #         return

    def _on_release(self, key):
        if key in (Key.ctrl_l, Key.ctrl_r):
            self._ctrl_key_pressed = False

    # def _wait_for_keypress(self, valid_keys: Sequence[KeyCode]) -> KeyCode | None:
    #     key_event = Event()
    #     pressed_key = None

    #     def on_press(key):
    #         nonlocal pressed_key
    #         if key in valid_keys:
    #             pressed_key = key
    #             key_event.set()
    #             return

    #     temp_keyboard_listener = Listener(on_press=on_press)
    #     temp_keyboard_listener.start()
    #     key_event.wait()
    #     temp_keyboard_listener.stop()
    #     del temp_keyboard_listener

    #     return pressed_key

    # def _show_interactive_info(self):
    #     self._in_menu = True
    #     current_menu = "main"
    #     selected_source_idx = None

    #     while self._in_menu and not self._exp_aborted:
    #         if current_menu == "main":
    #             current_menu = self._show_main_menu()
    #         elif current_menu == "source_select":
    #             result = self._show_source_selection_menu()
    #             if result == "back":
    #                 current_menu = "main"
    #             elif result is not None and isinstance(result, int):
    #                 selected_source_idx = result
    #                 current_menu = "source_controls"
    #             else:
    #                 current_menu = "main"
    #         elif current_menu == "source_controls" and selected_source_idx is not None:
    #             result = self._show_source_controls_menu(selected_source_idx)
    #             if result == "back":
    #                 current_menu = "source_select"
    #             else:
    #                 current_menu = "main"

    #     self._in_menu = False

    # def _show_main_menu(self) -> str:
    #     """Show main info menu. Returns next menu _"""
    #     LOGGER.info("\n" + "=" * 50)
    #     LOGGER.info(f"EXPERIMENT: {self.name}")
    #     LOGGER.info(f"Blocks: {len(self.blocks)}, Repetitions: {self.block_repetitions}")
    #     LOGGER.info(f"Sources: {len(self.sources)}, Sinks: {len(self.sinks)}")
    #     LOGGER.info("-" * 50)
    #     LOGGER.info("Controls:")
    #     LOGGER.info("  1 - Select source for control")
    #     LOGGER.info("  2 - View all sources")
    #     LOGGER.info("  q - Return to experiment")
    #     LOGGER.info("=" * 50)

    #     one = KeyCode.from_char("1")
    #     two = KeyCode.from_char("2")
    #     q = KeyCode.from_char("q")

    #     choice = self._wait_for_keypress([one, two, q])

    #     if choice == one:
    #         return "source_select"
    #     elif choice == two:
    #         self._show_all_sources()
    #         return "main"
    #     else:
    #         return "main"  # 'q' exits menu

    # def _show_all_sources(self):
    #     LOGGER.info("\n" + "-" * 50)
    #     LOGGER.info("ALL SOURCES:")
    #     for i, source in enumerate(self.sources):
    #         name = source.name
    #         tp = type(source).__name__

    #         control_status = "✓ Controls" if bool(source._keyboard_controls()) else "✗ No controls"
    #         LOGGER.info(f"  {i} - {name} ({tp}) [{control_status}]")
    #     LOGGER.info("-" * 50)
    #     sleep(2)

    # def _show_source_selection_menu(self) -> int | str | None:
    #     """Show source selection menu. Returns source index, 'back', or None."""
    #     LOGGER.info("\n" + "-" * 50)
    #     LOGGER.info("SELECT SOURCE FOR CONTROLS:")

    #     controllable_sources = []
    #     for idx, source in enumerate(self.sources):
    #         controls = source._keyboard_controls()
    #         if controls:
    #             controllable_sources.append((idx, source))

    #     if not controllable_sources:
    #         LOGGER.info("No sources with keyboard controls available.")
    #         sleep(1)
    #         return None

    #     for i, (idx, source) in enumerate(controllable_sources, 1):
    #         name = source.get_control_menu_name()
    #         LOGGER.info(f"  {i} - {name} (index: {idx})")

    #     LOGGER.info("  b - Back to main menu")
    #     LOGGER.info("-" * 50)

    #     valid_choices = [
    #         KeyCode.from_char(str(i)) for i in range(1, len(controllable_sources) + 1)
    #     ] + [KeyCode.from_char("b")]
    #     choice = self._wait_for_keypress(valid_choices)

    #     if choice is None:
    #         return None
    #     elif choice == KeyCode.from_char("b"):
    #         return "back"
    #     else:
    #         selected = int(choice.char if choice.char else "0") - 1
    #         return controllable_sources[selected][0]

    # def _show_source_controls_menu(self, source_idx: int) -> str:
    #     source = self.sources[source_idx]

    #     controls = source._keyboard_controls()
    #     if not controls:
    #         LOGGER.info(f"No controls available for {source.name}.")
    #         sleep(1)
    #         return "back"

    #     LOGGER.info(f"\n" + "-" * 50)
    #     LOGGER.info(f"CONTROLS FOR: {source.name} ({type(source).__name__})")
    #     LOGGER.info("Available shortcuts:")

    #     key_list = list(controls.keys())
    #     for i, (key_combo, (description, _)) in enumerate(controls.items(), 1):
    #         LOGGER.info(f"  {key_combo:<15} - {description}")

    #     # TODO no activate in experiment, just show controls and execute on keypress
    #     LOGGER.info("\nOptions:")
    #     LOGGER.info("  a - Activate these controls in experiment")
    #     LOGGER.info("  b - Back to source selection")
    #     LOGGER.info("-" * 50)

    #     a = KeyCode.from_char("a")
    #     b = KeyCode.from_char("b")
    #     choice = self._wait_for_keypress([a, b])

    #     if choice == a:
    #         # Activate source controls globally
    #         self._active_source_controls = {
    #             key: callback for key, (_, callback) in controls.items()
    #         }
    #         self._control_mode = True
    #         LOGGER.info(f"Activated controls for {source.name}. Press ESC to exit control mode.")
    #         LOGGER.info("Available shortcuts: " + ", ".join(controls.keys()))
    #         return "back"  # Return to experiment with controls active

    #     return "back"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


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
    _controller: ExperimentController = PrivateAttr(default_factory=ExperimentController)

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

    @property
    def controller(self) -> ExperimentController:
        return self._controller

    def open(self):
        if self._is_open:
            return

        self.controller.open()
        self.bus.open()

        for source in self.sources:
            source.bus = self.bus
            source.open()

        for sink in self.sinks:
            sink.bus = self.bus
            sink.open()

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        for sink in self.sinks:
            sink.close()

        for source in self.sources:
            source.close()

        self.bus.close()
        self.controller.close()

        self._is_open = False

    def run(self):
        self._ensure_open()

        LOGGER.info("Press ENTER to start, ESC to pause, Ctrl+Q to abort")

        self.controller.started.wait()

        if self.controller.aborted.is_set():
            return

        blocks = self._expand()

        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            for block, trials in tqdm(blocks, desc="Block", smoothing=True, position=0):
                for trial in tqdm(trials, desc="Trial", smoothing=True, position=1, leave=False):
                    if block.trigger and not block.trigger.wait():
                        if block.trigger_policy == "skip":
                            LOGGER.warning(
                                f"Trigger failed for trial {trial.name} in block {block.name}, skipping trial."
                            )
                            continue
                        msg = f"Trigger failed for trial {trial.name} in block {block.name}"
                        raise RuntimeError(msg)

                    st_info = {
                        f"{type(st).__name__}": st.model_dump(exclude_unset=True, exclude_none=True)
                        for st in trial.stimuli
                    }
                    LOGGER.info("Stimuli: %s", st_info)

                    list(
                        executor.map(
                            lambda source, stimuli=trial.stimuli: source.fire(stimuli), self.sources
                        )
                    )

                    isi = self.get_isi(block.isi or self.isi)
                    precise_sleep(isi / 1000.0)

                    self._wait_if_paused()

                    if self.controller.aborted.is_set():
                        break

                if self.controller.aborted.is_set():
                    break

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def _wait_if_paused(self):
        while self.controller.paused.is_set() and not self.controller.aborted.is_set():
            self.controller.paused.wait(0.1)

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
