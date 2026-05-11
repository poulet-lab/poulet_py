try:
    from collections.abc import Iterator
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import BaseData

except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

"""
    raise ImportError(msg) from e


class Trials:
    def __init__(self, trials: list[BaseData]):
        self._trials = trials

    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self) -> Iterator[BaseData]:
        return iter(self._trials)

    def __getitem__(self, key: Any):
        # index
        if isinstance(key, int):
            return self._trials[key]

        # slice
        if isinstance(key, slice):
            return Trials(self._trials[key])

        # name-based (assuming path.name is unique)
        if isinstance(key, str):
            for t in self._trials:
                if t.path.name == key:
                    return t
            raise KeyError(f"No trial named '{key}'")

        raise TypeError(f"Unsupported key type: {type(key)}")

    def filter(self, **criteria) -> "Trials":
        def match(trial: BaseData) -> bool:
            for key, value in criteria.items():
                attr = getattr(trial, key, None)

                # callable predicate support
                if callable(value):
                    if not value(attr):
                        return False
                else:
                    if attr != value:
                        return False
            return True

        return Trials([t for t in self._trials if match(t)])

    def map(self, fn):
        return [fn(t) for t in self._trials]

    def first(self) -> BaseData:
        return self._trials[0]

    def __repr__(self) -> str:
        return f"Trials(n={len(self._trials)})"


class Session(BaseModel):

    path: Path = Field(..., description="Path to the session folder")
    start: datetime | int | None = Field(default=None)
    end: datetime | int | None = Field(default=None)
    data_type: type[BaseData] = Field(
        default=..., description="Type of data in the session, e.g. 'widefield' or 'ephys'"
    )

    _trials: list[BaseData] = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)
    _trial_index: slice | None = PrivateAttr(default=None)
    _elapsed_seconds_end: int | None = PrivateAttr(default=None)
    _elapsed_seconds_window: tuple[int, int] | None = PrivateAttr(default=None)

    @property
    def trials(self) -> "Trials":
        self._ensure_open()
        return Trials(self._trials)

    @property
    def trial_range(self) -> slice | None:
        return self._trial_index

    def set_trial_range(
        self,
        start: int | None,
        end: int | None,
        step: int | None = None,
    ) -> None:
        for name, value in (("start", start), ("end", end), ("step", step)):
            if value is not None and not isinstance(value, int):
                msg = f"{name} must be an integer or None"
                raise TypeError(msg)
        if step == 0:
            msg = "step must not be zero"
            raise ValueError(msg)

        self._trial_index = slice(start, end, step)

    def set_elapsed_seconds(self, start: Any, end: Any | None = None) -> None:
        if not isinstance(start, int):
            msg = "start must be an integer"
            raise TypeError(msg)
        if start < 0:
            msg = "start must be >= 0"
            raise ValueError(msg)

        if end is None:
            self._elapsed_seconds_end = start
            self._elapsed_seconds_window = None
            return

        if not isinstance(end, int):
            msg = "end must be an integer"
            raise TypeError(msg)
        if end < 0:
            msg = "end must be >= 0"
            raise ValueError(msg)
        if start > end:
            msg = "elapsed seconds window must be ordered as (start, end)"
            raise ValueError(msg)

        self._elapsed_seconds_end = None
        self._elapsed_seconds_window = (start, end)

    def clear_elapsed_seconds_filter(self) -> None:
        self._elapsed_seconds_end = None
        self._elapsed_seconds_window = None

    def _is_elapsed_seconds_filter_enabled(self) -> bool:
        return self._elapsed_seconds_end is not None or self._elapsed_seconds_window is not None

    def _resolve_session_anchor_time(self, trial_paths: list[Path]) -> datetime:
        parser = getattr(self.data_type, "_parse_folder_datetime", None)
        if not callable(parser):
            msg = (
                f"{self.data_type} does not support elapsed-seconds filtering. "
                "Expected a _parse_folder_datetime(folder_name) helper."
            )
            raise TypeError(msg)

        first_trial_name = trial_paths[0].name
        anchor_time = parser(first_trial_name)
        if anchor_time is None:
            msg = (
                "Unable to compute session anchor time from first trial folder "
                f"'{first_trial_name}'. Expected folder names in yyMMdd_HHmmss, "
                "yyyyMMdd_HHmmss, or HHmmss format."
            )
            raise ValueError(msg)
        return anchor_time

    def _select_trial_paths(self, trial_paths: list[Path]) -> list[Path]:
        if self._trial_index is None:
            return trial_paths

        selected_paths = trial_paths[self._trial_index]
        if not selected_paths:
            msg = (
                f"Trial range {self._trial_index.start}:{self._trial_index.stop}:"
                f"{self._trial_index.step} selected no trials out of "
                f"{len(trial_paths)} discovered"
            )
            raise ValueError(msg)
        return selected_paths

    def open(self) -> None:
        if self._is_open:
            return

        if not self.path.exists():
            msg = f"Session path does not exist: {self.path}"
            raise FileNotFoundError(msg)

        trial_paths = [p for p in self.path.iterdir() if p.is_dir()]

        if not trial_paths:
            trial_paths = [self.path]

        trial_paths = self._select_trial_paths(sorted(trial_paths))
        session_anchor_time = (
            self._resolve_session_anchor_time(trial_paths)
            if self._is_elapsed_seconds_filter_enabled()
            else None
        )

        trials: list[BaseData] = []

        for path in trial_paths:
            trial = self.data_type(
                path=path,
                start=self.start,
                end=self.end,
                elapsed_seconds_end=self._elapsed_seconds_end,
                elapsed_seconds_window=self._elapsed_seconds_window,
                session_anchor_time=session_anchor_time,
            )
            if (
                (
                    self._is_elapsed_seconds_filter_enabled()
                    or self.start is not None
                    or self.end is not None
                )
                and not trial._should_open()
            ):
                continue
            trial.open()
            trials.append(trial)

        self._trials = trials

        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        trials = self._trials if isinstance(self._trials, list) else [self._trials]

        for trial in trials:
            trial.close()

        self._trials = []
        self._is_open = False

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
