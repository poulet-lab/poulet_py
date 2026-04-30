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


#TODO
class TrialCollection:
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
            return TrialCollection(self._trials[key])

        # name-based (assuming path.name is unique)
        if isinstance(key, str):
            for t in self._trials:
                if t.path.name == key:
                    return t
            raise KeyError(f"No trial named '{key}'")

        raise TypeError(f"Unsupported key type: {type(key)}")

    def filter(self, **criteria) -> "TrialCollection":
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

        return TrialCollection([t for t in self._trials if match(t)])

    def map(self, fn):
        return [fn(t) for t in self._trials]

    def first(self) -> BaseData:
        return self._trials[0]

    def __repr__(self) -> str:
        return f"TrialCollection(n={len(self._trials)})"


class Session(BaseModel):
    # TODO: function to index trials

    path: Path = Field(..., description="Path to the session folder")
    start: datetime | int | None = Field(default=None)
    end: datetime | int | None = Field(default=None)
    data_type: type[BaseData] = Field(
        default=..., description="Type of data in the session, e.g. 'widefield' or 'ephys'"
    )

    _trials: list[BaseData] = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)

    @property
    def trials(self) -> "TrialCollection":
        self._ensure_open()
        return TrialCollection(self._trials)

    def open(self) -> None:
        if self._is_open:
            return

        if not self.path.exists():
            msg = f"Session path does not exist: {self.path}"
            raise FileNotFoundError(msg)

        trial_paths = [p for p in self.path.iterdir() if p.is_dir()]

        if not trial_paths:
            trial_paths = [self.path]

        trials: list[BaseData] = []

        for path in sorted(trial_paths):
            trial = self.data_type(path=path, start=self.start, end=self.end)
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
