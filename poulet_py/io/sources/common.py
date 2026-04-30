try:
    from abc import ABC, abstractmethod
    from collections.abc import Callable, Sequence
    from enum import Enum
    from threading import Lock
    from time import time_ns
    from typing import Literal

    from numpy import ndarray, zeros
    from numpy.typing import DTypeLike
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import BaseStimulus, EventBus, SinkEvent
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class AcquisitionType(str, Enum):
    NONE = "None"
    CONTINUOUS = "continuous"
    FINITE = "finite"


class BaseSource(BaseModel, ABC):
    name: str = Field(..., description="Name of the data source")
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )
    fire_on: Literal["all"] | list[type] = Field(
        default="all", description="List of stimuli to fire on, or 'all' for all stimuli"
    )
    buffer_size: int = Field(default=500, description="Size of the circular buffer", ge=1)

    _bus: EventBus = PrivateAttr(default_factory=EventBus)

    _stimuli: Sequence[BaseStimulus] = PrivateAttr(default_factory=list)
    _max_stimulus_duration_ms: int = PrivateAttr(0)
    _is_open: bool = PrivateAttr(default=False)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _buffer: ndarray = PrivateAttr()
    _buffer_idx: int = PrivateAttr(default=0)
    _buffer_dtype: DTypeLike = PrivateAttr()
    _last_timestamp: int = PrivateAttr(default_factory=time_ns)

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _set_buffer_dtype(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _fire(self) -> bool: ...

    def _keyboard_controls(self) -> dict[str, tuple[str, Callable]]:
        """
        Return a dict of keyboard shortcuts and their handlers.
        Format: {key_combination: (description, callback)}
        Example: {'ctrl+r': ('Reset device', self.reset_device)}
        """
        ...

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus):
        if self._is_open:
            msg = f"Cannot change bus while {self.name} is open"
            raise RuntimeError(msg)
        self._bus = value

    def open(self):
        if self._is_open:
            return

        self.bus.open()
        self._set_buffer()
        self._open()
        self._is_open = True

    def close(self):
        self._close()
        self._del_buffer()
        self._bus.close()
        self._is_open = False

    def fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        self._ensure_open()

        self._stimuli = stimuli
        self._supports()

        if not self._stimuli and self.acquisition_type == AcquisitionType.FINITE:
            return False

        self._calculate_stimulus_duration()

        if self.acquisition_type == AcquisitionType.FINITE:
            with self._lock:
                self._last_timestamp = time_ns()

        self._fire()
        self._publish()

        return True

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def _set_buffer(self):
        self._set_buffer_dtype()

        if (
            not isinstance(self._buffer_dtype, list)
            or ("timestamp", "uint64") not in self._buffer_dtype
        ):
            msg = "Buffer dtype must include a 'timestamp' field of type uint64"
            raise ValueError(msg)

        self._buffer = zeros(self.buffer_size, dtype=self._buffer_dtype)
        self._buffer_idx = 0

    def _del_buffer(self):
        del self._buffer
        self._buffer_idx = 0
        self._last_timestamp = time_ns()

    def _supports(self):
        if self.fire_on == "all":
            return
        self._stimuli = [st for st in self._stimuli if type(st) in self.fire_on]

    def _calculate_stimulus_duration(self):
        pre_delay = 0
        duration = 0
        post_delay = 0

        for st in self._stimuli:
            pre_delay = max(pre_delay, st.pre_delay)
            duration = max(duration, st.duration)
            post_delay = max(post_delay, st.post_delay)

        self._max_stimulus_duration_ms = pre_delay + duration + post_delay

    def _publish(self) -> bool:
        with self._lock:
            idx = self._buffer_idx % self.buffer_size
            start = self._last_timestamp
            end = self._buffer["timestamp"][idx - 1]

            if end == 0:
                end = self._buffer["timestamp"][idx]

            self._last_timestamp = end

        mask = (self._buffer["timestamp"] >= start) & (self._buffer["timestamp"] <= end)
        chunk = self._buffer[mask].copy()

        if chunk.size == 0:
            return False

        self.bus.emit(
            SinkEvent(
                name=self.name, payload=chunk, meta={"acquisition": self.acquisition_type.value}
            )
        )

        return True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
