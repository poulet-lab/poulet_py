from collections.abc import Sequence

try:
    from collections.abc import Iterator
    from typing import Generic, TypeVar
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

"""
    raise ImportError(msg) from e

T = TypeVar("T")


class Index(Generic[T]):
    def __init__(self, data: list[T] | None = None):
        self._data: list[T] = data if data is not None else []

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __contains__(self, item: T) -> bool:
        return item in self._data

    @property
    def iloc(self) -> "_ILocIndexer[T]":
        return _ILocIndexer(self._data)

    @property
    def loc(self) -> "_LocIndexer[T]":
        return _LocIndexer(self._data)

    def __getitem__(
        self, key: int | str | slice | Sequence[int] | Sequence[str]
    ) -> T | Sequence[T]:
        if isinstance(key, int):
            return self.iloc[key]
        elif isinstance(key, str):
            return self.loc[key]
        elif isinstance(key, slice):
            if self._is_integer_slice(key):
                return self.iloc[key]
            elif self._is_string_slice(key):
                return self.loc[key]
            else:
                msg_0 = (
                    "Mixed integer/string slices not supported. "
                    "Use .iloc for integer slices or .loc for string slices."
                )
                raise TypeError(msg_0)
        elif isinstance(key, Sequence):
            if not key:
                return []
            if all(isinstance(k, int) for k in key):
                return self.iloc[key]
            elif all(isinstance(k, str) for k in key):
                return self.loc[key]
            else:
                msg_1 = (
                    "Mixed integer/string lists not supported. "
                    "Use .iloc for integer lists or .loc for string lists."
                )
                raise TypeError(msg_1)

        msg_2 = f"Unsupported key type: {type(key)}"
        raise TypeError(msg_2)

    def __setitem__(self, key: int, value: T) -> None:
        if not isinstance(key, int):
            msg_0 = "Only integer assignment is supported"
            raise TypeError(msg_0)
        self._data[key] = value

    def __delitem__(self, key: int | slice) -> None:
        if isinstance(key, (int, slice)):
            del self._data[key]
        else:
            msg_0 = "Only integer or slice deletion is supported"
            raise TypeError(msg_0)

    def append(self, item: T) -> None:
        self._data.append(item)

    def extend(self, items: Sequence[T]) -> None:
        self._data.extend(items)

    def insert(self, index: int, item: T) -> None:
        self._data.insert(index, item)

    def remove(self, item: T) -> None:
        self._data.remove(item)

    def pop(self, index: int = -1) -> T:
        return self._data.pop(index)

    def clear(self) -> None:
        self._data.clear()

    def index(self, item: T, *args) -> int:
        return self._data.index(item, *args)

    def count(self, item: T) -> int:
        return self._data.count(item)

    def reverse(self) -> None:
        self._data.reverse()

    def sort(self, *, key=None, reverse: bool = False) -> None:
        self._data.sort(key=key, reverse=reverse)

    def copy(self) -> "Index[T]":
        return Index(self._data.copy())

    def _is_integer_slice(self, s: slice) -> bool:
        return all(isinstance(x, (int, type(None))) for x in (s.start, s.stop, s.step))

    def _is_string_slice(self, s: slice) -> bool:
        return all(isinstance(x, (str, type(None))) for x in (s.start, s.stop, s.step))

    def to_list(self) -> list[T]:
        """Return the underlying list."""
        return self._data.copy()

    def __repr__(self) -> str:
        return f"Index({self._data})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Index):
            return self._data == other._data
        elif isinstance(other, list):
            return self._data == other
        return NotImplemented


class _ILocIndexer(Generic[T]):
    def __init__(self, data: Sequence[T]):
        self._data = data

    def __getitem__(self, key: int | slice | Sequence[int]) -> T | Sequence[T]:
        if isinstance(key, int):
            if key < 0:
                key = len(self._data) + key

            if 0 <= key < len(self._data):
                return self._data[key]

            msg_0 = f"Index {key} out of range for container of size {len(self._data)}"
            raise IndexError(msg_0)

        elif isinstance(key, slice):
            return self._data[key]

        elif isinstance(key, Sequence):
            return [self[i] for i in key]

        msg_1 = f"Unsupported key type for iloc: {type(key)}"
        raise TypeError(msg_1)

    def __repr__(self) -> str:
        return f"_ILocIndexer(size={len(self._data)})"


class _LocIndexer(Generic[T]):
    def __init__(self, data: Sequence[T]):
        self._data = data
        # Build label index for faster lookups
        self._label_to_index: dict[str, int] = {}
        for i, item in enumerate(data):
            if hasattr(item, "path"):
                self._label_to_index[str(item.path)] = i

    def _get_label(self, item: T) -> str:
        """Extract label from an item. Override for custom label logic."""
        if hasattr(item, "path"):
            return str(item.path)

        msg_0 = f"Items of type {type(item)} must have a 'path' attribute for loc indexing"
        raise AttributeError(msg_0)

    def __getitem__(self, key: str | slice | Sequence[str]) -> T | Sequence[T]:
        if isinstance(key, str):
            if key in self._label_to_index:
                return self._data[self._label_to_index[key]]
            msg_0 = f"Label '{key}' not found"
            raise KeyError(msg_0)

        elif isinstance(key, slice):
            # Convert string labels to indices
            start_idx = None
            stop_idx = None

            if key.start is not None:
                if key.start not in self._label_to_index:
                    msg_0 = f"Start label '{key.start}' not found"
                    raise KeyError(msg_0)
                start_idx = self._label_to_index[key.start]

            if key.stop is not None:
                if key.stop not in self._label_to_index:
                    msg_1 = f"Stop label '{key.stop}' not found"
                    raise KeyError(msg_1)
                stop_idx = self._label_to_index[key.stop]

            # Apply the integer slice with original step
            return self._data[start_idx : stop_idx : key.step]

        elif isinstance(key, list):
            results = []
            for label in key:
                results.append(self._data[self[label]])

            return results

        msg_2 = f"Unsupported key type for loc: {type(key)}"
        raise TypeError(msg_2)

    def __contains__(self, label: str) -> bool:
        return label in self._label_to_index

    def keys(self) -> list[str]:
        return list(self._label_to_index.keys())

    def __repr__(self) -> str:
        return f"_LocIndexer(labels={list(self._label_to_index.keys())})"


# class TimeFilter:
#     """Handles time-based filtering of trials."""

#     def __init__(self):
#         self._start: datetime | int | None = None
#         self._end: datetime | int | None = None
#         self._elapsed_seconds_end: int | None = None
#         self._elapsed_seconds_window: tuple[int, int] | None = None
#         self._anchor_time: datetime | None = None

#     def set_range(self, start: datetime | int, end: datetime | int):
#         self._start = start
#         self._end = end
#         return self

#     def set_elapsed(self, start_seconds: int, end_seconds: int | None = None):
#         if end_seconds is None:
#             self._elapsed_seconds_end = start_seconds
#             self._elapsed_seconds_window = None
#         else:
#             self._elapsed_seconds_end = None
#             self._elapsed_seconds_window = (start_seconds, end_seconds)
#         return self

#     def apply(self, indices: list[TrialIndex]) -> list[TrialIndex]:
#         """Filter trial indices based on time criteria."""
#         if all(
#             v is None
#             for v in [
#                 self._start,
#                 self._end,
#                 self._elapsed_seconds_end,
#                 self._elapsed_seconds_window,
#             ]
#         ):
#             return indices

#         filtered = []
#         for idx in indices:
#             if self._matches_time_criteria(idx):
#                 filtered.append(idx)
#         return filtered

#     def _matches_time_criteria(self, index: TrialIndex) -> bool:
#         # Implement your time matching logic here
#         return True


# class Query(BaseModel):
#     trials: "Trials"
#     _conditions: list[Callable] = []
#     _time_filter = TimeFilter()

#     def where(self, **criteria) -> "Query":
#         """Add filter conditions."""

#         def check(trial: BaseData) -> bool:
#             for key, value in criteria.items():
#                 trial.filter(key, value)

#                     return False
#             return True

#         self._conditions.append(check)
#         return self

#     def time_between(self, start: datetime | int, end: datetime | int) -> "Query":
#         self._time_filter.set_range(start, end)
#         return self

#     def elapsed_seconds_up_to(self, seconds: int) -> "Query":
#         self._time_filter.set_elapsed(seconds)
#         return self

#     def select(self) -> list["BaseData"]:
#         """Execute the query and return results."""
#         # Apply time filtering first
#         indices = self._time_filter.apply(self.trials._indices)

#         # Load and filter
#         results = []
#         for idx in indices:
#             trial = self.trials._backend.load_trial(idx)
#             trial_data = BaseData.from_trial_index(idx, **trial)

#             # Apply all conditions
#             if all(cond(trial_data) for cond in self._conditions):
#                 results.append(trial_data)

#         return results
