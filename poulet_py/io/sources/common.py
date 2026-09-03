try:
    from abc import ABC, abstractmethod
    from collections.abc import Callable
    from enum import Enum
    from threading import Barrier, Event, Lock, Thread
    from time import sleep
    from typing import Literal

    from numpy import concatenate, dtype, ndarray, zeros
    from numpy.typing import DTypeLike
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, BaseStimulus, EventBus, SinkEvent
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class AcquisitionType(str, Enum):
    NONE = "None"
    CONTINUOUS = "continuous"
    FINITE = "finite"


class BaseSource(BaseModel, ABC):
    name: str = Field(..., description="Name of the data source")
    fire_on: Literal["all"] | tuple[type[BaseStimulus]] = Field(
        default="all", description="List of stimuli to fire on, or 'all' for all stimuli"
    )
    buffer_size: int = Field(default=1000, description="Size of the circular buffer", ge=1)

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _stimuli: tuple[BaseStimulus, ...] = PrivateAttr(default_factory=tuple)
    _max_stimulus_duration_ms: int = PrivateAttr(0)
    _is_open: bool = PrivateAttr(default=False)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _source_buffer: ndarray = PrivateAttr()
    _source_buffer_idx: int = PrivateAttr(default=0)
    _source_buffer_needle: int = PrivateAttr(default=0)
    _source_buffer_dtype: DTypeLike = PrivateAttr()
    _total_written: int = PrivateAttr(default=0)
    _last_published_written: int = PrivateAttr(default=0)

    _fire_thread: Thread = PrivateAttr()
    _publish_thread: Thread = PrivateAttr()
    _start_fire: Event = PrivateAttr(default_factory=Event)
    _done_fire: Event = PrivateAttr(default_factory=Event)
    _stop_thread: Event = PrivateAttr(default_factory=Event)
    _barrier: Barrier | None = PrivateAttr(default=None)

    @abstractmethod
    def _open(self): ...
    @abstractmethod
    def _close(self): ...
    @abstractmethod
    def _set_buffer_dtype(self): ...

    def _fire(self) -> bool:
        return False

    def _acquire(self) -> None:
        return

    def _keyboard_controls(self) -> dict[str, tuple[str, Callable]]:
        return {}

    @property
    def has_crashed(self):
        if self._is_open and self._stop_thread.is_set():
            return True
        return False

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus):
        if self._is_open:
            raise RuntimeError(f"Cannot change bus while {self.name} is open")

        self._bus = value
        self._external_bus = True

    @property
    def barrier(self) -> Barrier | None:
        return self._barrier

    @barrier.setter
    def barrier(self, value: Barrier | None):
        self._barrier = value

    def open(self) -> None:
        if self._is_open:
            return

        if not self._external_bus:
            self.bus.open()

        self._open()
        self._set_source_buffer()

        self._fire_thread = Thread(
            target=self._fire_loop, daemon=True, name=f"{self.name}-fire-loop"
        )
        self._fire_thread.start()

        self._publish_thread = Thread(
            target=self._publish_loop, daemon=True, name=f"{self.name}-publish-loop"
        )
        self._publish_thread.start()

        self._is_open = True

    def close(self) -> None:
        self._stop_thread.set()
        self._start_fire.set()

        if self._fire_thread.is_alive():
            self._fire_thread.join()

        if self._publish_thread.is_alive():
            self._publish_thread.join()

        self._del_source_buffer()
        self._close()

        if not self._external_bus:
            self.bus.close()

        self._is_open = False

    def fire(self, stimuli: tuple[BaseStimulus, ...]) -> bool:
        self._ensure_open()
        self._stimuli = stimuli

        self._done_fire.clear()
        self._start_fire.set()

        return True

    def wait(self) -> bool:
        self._done_fire.wait()
        return True

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"{type(self).__name__} needs to be opened first")

    def _ensure_buffer_size(self, n: int):
        with self._lock:
            if n > self.buffer_size:
                new_size = int(n * 1.5)
                LOGGER.warning(
                    f"[{self.name}] Incoming batch size ({n}) exceeds buffer_size ({self.buffer_size}). "
                    f"Automatically expanding circular buffer to {new_size} to prevent data loss."
                )

                dt = dtype(self._source_buffer_dtype)
                new_buffer = zeros(new_size, dtype=dt)

                current = self._source_buffer_idx
                needle = self._source_buffer_needle
                unread_count = current - needle

                if unread_count > 0:
                    unread_count = min(unread_count, self.buffer_size)
                    start_read = (current - unread_count) % self.buffer_size
                    end_read = current % self.buffer_size

                    if start_read < end_read:
                        new_buffer[:unread_count] = self._source_buffer[start_read:end_read]
                    else:
                        split_read = self.buffer_size - start_read
                        new_buffer[:split_read] = self._source_buffer[start_read:]
                        new_buffer[split_read:unread_count] = self._source_buffer[:end_read]

                self._source_buffer = new_buffer
                self.buffer_size = new_size
                self._source_buffer_needle = 0
                self._source_buffer_idx = unread_count

    def _set_source_buffer(self) -> None:
        self._set_buffer_dtype()
        dt = dtype(self._source_buffer_dtype)

        if not dt.names or "timestamp" not in dt.names:
            raise ValueError("Buffer dtype must include a 'timestamp' field of type uint64")

        self._source_buffer = zeros(self.buffer_size, dtype=dt)
        self._source_buffer_idx = 0
        self._source_buffer_needle = 0

    def _del_source_buffer(self) -> None:
        if hasattr(self, "_source_buffer"):
            del self._source_buffer

        self._source_buffer_idx = 0
        self._source_buffer_needle = 0

    def _write_sample(self, sample: tuple | dict) -> None:
        with self._lock:
            idx = self._source_buffer_idx % self.buffer_size

            if isinstance(sample, dict):
                for k, v in sample.items():
                    self._source_buffer[idx][k] = v
            else:
                self._source_buffer[idx] = sample

            self._source_buffer_idx += 1

    def _write_samples(self, samples: ndarray) -> None:
        n = len(samples)
        if n == 0:
            return

        self._ensure_buffer_size(n)

        with self._lock:
            start = self._source_buffer_idx % self.buffer_size
            end = start + n

            if end <= self.buffer_size:
                self._source_buffer[start:end] = samples
            else:
                split = self.buffer_size - start
                remainder = (end % self.buffer_size) or self.buffer_size

                self._source_buffer[start:] = samples[:split]
                if remainder > 0:
                    self._source_buffer[:remainder] = samples[split:]

            self._source_buffer_idx += n

    def _supports(self) -> None:
        if self.fire_on == "all":
            return
        self._stimuli = tuple(st for st in self._stimuli if isinstance(st, self.fire_on))

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
        chunk = self._get_new_chunk()

        if chunk is None or chunk.size == 0:
            return False

        self.bus.emit(SinkEvent(name=self.name, payload=chunk))

        return True

    def _get_new_chunk(self) -> ndarray | None:
        #TODO write in chunks
        with self._lock:
            current = self._source_buffer_idx
            last = self._source_buffer_needle

            if current == last:
                return None

            unread = current - last

            if unread > self.buffer_size:
                last = current - self.buffer_size

            start = last % self.buffer_size
            end = current % self.buffer_size

            chunk = (
                self._source_buffer[start:end]
                if start < end
                else concatenate(
                    (
                        self._source_buffer[start:],
                        self._source_buffer[:end],
                    )
                )
            )

            self._source_buffer_needle = current

            return chunk.copy()

    def _fire_loop(self):
        while not self._stop_thread.is_set():
            try:
                self._start_fire.wait()
                self._start_fire.clear()

                if self._stop_thread.is_set():
                    break

                self._supports()
                self._calculate_stimulus_duration()

                if self._barrier:
                    self._barrier.wait()

                if self._stimuli:
                    self._fire()

                self._done_fire.set()
            except Exception as e:
                LOGGER.exception(e)
                self._stop_thread.set()

    def _publish_loop(self):
        while not self._stop_thread.is_set():
            try:
                self._acquire()
                self._publish()
                sleep(0.01)
            except Exception as e:
                LOGGER.exception(e)
                self._stop_thread.set()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
