try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from enum import Enum
    from threading import Lock
    from typing import Literal

    from numpy import dtype, ndarray, zeros
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import BaseEvent, BaseStimulus, EventBus, SinkEvent
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
    acquisition_type: AcquisitionType = Field(default=AcquisitionType.FINITE, description="")
    fire_on: Literal["all"] | list[BaseStimulus] = Field(default="all")
    buffer_size: int = Field(default=500, description="Size of the circular buffer", ge=1)
    bus: EventBus | None = Field(default=None)

    _stimuli: Sequence[BaseStimulus] = PrivateAttr(default_factory=list)
    _max_stimulus_duration_ms: int = PrivateAttr(0)
    _is_open: bool = PrivateAttr(default=False)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _buffer: ndarray = PrivateAttr()
    _buffer_idx: int = PrivateAttr(default=0)
    _buffer_dtype: Sequence[tuple[str, dtype | str | Sequence[tuple[str, dtype | str]]]] = (
        PrivateAttr()
    )
    _last_timestamp: int = PrivateAttr(default=0)

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _set_buffer_dtype(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _fire(self) -> bool: ...

    def assert_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def open(self, bus: EventBus | None = None):
        if self._is_open:
            return

        if not bus and not self.bus:
            msg = "Event bus must be defined"
            raise RuntimeError(msg)

        if bus and not self.bus:
            self.bus = bus

        self._open()
        self._set_buffer()
        self._is_open = True

    def close(self):
        self._close()
        self._is_open = False

    def fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        self.assert_open()

        self._stimuli = stimuli
        self._supports()

        if not self._stimuli and self.acquisition_type == AcquisitionType.FINITE:
            return False

        self._calculate_stimulus_duration()

        if self.acquisition_type == AcquisitionType.FINITE:
            with self._lock:
                self._last_timestamp = self._buffer["timestamp"][
                    (self._buffer_idx % self.buffer_size) - 1
                ]

        self._fire()
        self._publish()

        return True

    def publish(self, event: BaseEvent):
        if self.bus is None:
            msg = "No Event bus is attached"
            raise RuntimeError(msg)

        self.bus.emit(event)

    def _set_buffer(self):
        self._set_buffer_dtype()

        if ("timestamp", "uint64") not in self._buffer_dtype:
            self._buffer_dtype = [("timestamp", "uint64"), *self._buffer_dtype]

        self._buffer = zeros(self.buffer_size, dtype=self._buffer_dtype)
        self._last_timestamp = 0
        self._buffer_idx = 0

    def _supports(self):
        if self.fire_on == "all":
            return
        self._stimuli = [st for st in self._stimuli if st in self.fire_on]

    def _calculate_stimulus_duration(self):
        pre_delay = 0
        duration = 0
        post_delay = 0
        isi = 0

        for st in self._stimuli:
            pre_delay = max(pre_delay, st.pre_delay)
            duration = max(duration, st.duration)
            post_delay = max(post_delay, st.post_delay)
            isi = max(isi, st._isi)

        self._max_stimulus_duration_ms = pre_delay + duration + post_delay + isi

    def _publish(self) -> bool:
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            return self._publish_continuous()
        elif self.acquisition_type == AcquisitionType.FINITE:
            return self._publish_finite()

        return False

    def _publish_continuous(self) -> bool:
        with self._lock:
            start = self._last_timestamp
            end = self._buffer["timestamp"][(self._buffer_idx % self.buffer_size) - 1]
            self._last_timestamp = end
            print(f"{self.name} publish continuous: start={start}, end={end}")

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] < end)
        chunk = self._buffer[mask].copy()
        print(f"{self.name} publish continuous: chunk size={chunk.size}")
        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(
                name=self.name, payload={self.name: chunk}, meta={"acquisition": "continuous"}
            )
        )

        return True

    def _publish_finite(self) -> bool:
        with self._lock:
            start = self._last_timestamp
            end = start + self._max_stimulus_duration_ms * 1_000_000
            self._last_timestamp = end

        mask = (self._buffer["timestamp"] > start) & (self._buffer["timestamp"] <= end)
        chunk = self._buffer[mask].copy()

        if chunk.size == 0:
            return False

        self.publish(
            SinkEvent(name=self.name, payload={self.name: chunk}, meta={"acquisition": "finite"})
        )

        return True

    def __enter__(self):
        self.open(bus=self.bus)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
