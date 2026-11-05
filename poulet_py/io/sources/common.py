try:
    from abc import ABC, abstractmethod
    from collections.abc import Callable
    from enum import Enum
    from threading import Lock
    from typing import Literal

    from numpy import concatenate, dtype, ndarray, zeros
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
    fire_on: Literal["all"] | list[type[BaseStimulus]] = Field(
        default="all", description="List of stimuli to fire on, or 'all' for all stimuli"
    )
    buffer_size: int = Field(default=500, description="Size of the circular buffer", ge=1)

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _stimuli: tuple[BaseStimulus, ...] = PrivateAttr(default_factory=tuple)
    _max_stimulus_duration_ms: int = PrivateAttr(0)
    _is_open: bool = PrivateAttr(default=False)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _buffer: ndarray = PrivateAttr()
    _buffer_idx: int = PrivateAttr(default=0)
    _last_publish_idx: int = PrivateAttr(default=0)
    _buffer_dtype: DTypeLike = PrivateAttr()
    _total_written: int = PrivateAttr(default=0)
    _last_published_written: int = PrivateAttr(default=0)

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _set_buffer_dtype(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _fire(self) -> bool: ...

    def _keyboard_controls(self) -> dict[str, tuple[str, Callable]]:
        return {}

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus):
        if self._is_open:
            msg = f"Cannot change bus while {self.name} is open"
            raise RuntimeError(msg)

        self._bus = value
        self._external_bus = True

    def open(self):
        if self._is_open:
            return

        if not self._external_bus:
            self.bus.open()

        self._set_buffer()
        self._open()
        self._is_open = True

    def close(self):

        self._close()
        self._del_buffer()

        if not self._external_bus:
            self.bus.close()

        self._is_open = False

    def fire(self, stimuli: tuple[BaseStimulus, ...]) -> bool:
        self._ensure_open()

        self._stimuli = stimuli
        self._supports()

        if not self._stimuli and self.acquisition_type == AcquisitionType.FINITE:
            return False

        self._calculate_stimulus_duration()

        if self.acquisition_type == AcquisitionType.FINITE:
            with self._lock:
                self._last_publish_idx = self._buffer_idx

        self._fire()
        self._publish()

        return True

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self).__name__} needs to be opened first"
            raise RuntimeError(msg)

    def _set_buffer(self):
        if self.acquisition_type == AcquisitionType.NONE:
            return

        self._set_buffer_dtype()
        dt = dtype(self._buffer_dtype)

        if not dt.names or "timestamp" not in dt.names:
            msg = "Buffer dtype must include a 'timestamp' field of type uint64"
            raise ValueError(msg)

        self._buffer = zeros(self.buffer_size, dtype=dt)
        self._buffer_idx = 0
        self._last_publish_idx = 0

    def _del_buffer(self):
        if hasattr(self, "_buffer"):
            del self._buffer

        self._buffer_idx = 0
        self._last_publish_idx = 0

    def _write_sample(self, sample: tuple | dict):
        if self.acquisition_type == AcquisitionType.NONE:
            return

        with self._lock:
            idx = self._buffer_idx % self.buffer_size

            if isinstance(sample, dict):
                for k, v in sample.items():
                    self._buffer[idx][k] = v
            else:
                self._buffer[idx] = sample

            self._buffer_idx += 1

    def _write_samples(self, samples: ndarray):
        if self.acquisition_type == AcquisitionType.NONE:
            return

        n = len(samples)

        with self._lock:
            start = self._buffer_idx % self.buffer_size
            end = start + n

            if end <= self.buffer_size:
                self._buffer[start:end] = samples
            else:
                split = self.buffer_size - start

                self._buffer[start:] = samples[:split]
                self._buffer[: end % self.buffer_size] = samples[split:]

            self._buffer_idx += n

    def _supports(self):
        if self.fire_on == "all":
            return
        self._stimuli = tuple(st for st in self._stimuli if isinstance(st, tuple(self.fire_on)))

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
        if self.acquisition_type == AcquisitionType.NONE:
            return False

        chunk = self._get_new_chunk()

        if chunk is None or chunk.size == 0:
            return False

        self.bus.emit(
            SinkEvent(
                name=self.name, payload=chunk, meta={"acquisition": self.acquisition_type.value}
            )
        )

        return True

    def _get_new_chunk(self) -> ndarray | None:
        with self._lock:
            current = self._buffer_idx
            last = self._last_publish_idx

            if current == last:
                return None

            unread = current - last

            if unread > self.buffer_size:
                # overflow: unread data overwritten
                last = current - self.buffer_size

            start = last % self.buffer_size
            end = current % self.buffer_size

            chunk = (
                self._buffer[start:end]
                if start < end
                else concatenate(
                    (
                        self._buffer[start:],
                        self._buffer[:end],
                    )
                )
            )

            self._last_publish_idx = current

            return chunk.copy()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
