try:
    from abc import ABC, abstractmethod
    from threading import Condition, Event, Thread
    from time import monotonic_ns
    from types import TracebackType
    from typing import Self

    from numpy import ndarray, zeros
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, AcquisitionType
except ImportError as e:
    raise ImportError("""
Missing 'hardware' module. Install options:
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class BaseHardware(BaseModel, ABC):
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )
    buffer_size: int = Field(default=100, description="Size of the internal sampling queue", ge=1)

    _is_open: bool = PrivateAttr(default=False)

    _buffer_idx: int = PrivateAttr(default=0)
    _buffer_needle: int = PrivateAttr(default=0)
    _buffer: ndarray = PrivateAttr()

    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _acquisition_condition: Condition = PrivateAttr(default_factory=Condition)
    _wake_at: int = PrivateAttr(1)  # consumer sets this before waiting

    @abstractmethod
    def _get_sample_spec(self) -> list[tuple[str, str] | tuple[str, str, tuple[int, ...]]]: ...

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _close(self) -> None: ...

    @abstractmethod
    def _acquisition_thread_func(self) -> None: ...

    @abstractmethod
    def _start_capture(self) -> None: ...

    @abstractmethod
    def _stop_capture(self) -> None: ...

    @abstractmethod
    def _acquire_sample(self) -> ndarray | None: ...

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return

        self._open()
        self._set_buffer()

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._start_acquisition_thread()

        self._is_open = True

    def close(self) -> None:
        self._is_open = False

        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            self._stop_acquisition_thread()

        self._release_buffer()
        self._close()

    def read_sample(self, timeout: float | None = None) -> ndarray | None:
        self._ensure_open()
        sample = None

        if self.acquisition_type == AcquisitionType.FINITE:
            self._start_capture()
            ts = monotonic_ns()
            sample = self._acquire_sample()

            if sample is not None:
                self._sample_to_buffer(sample, ts)

            self._stop_capture()

        with self._acquisition_condition:
            if self.acquisition_type == AcquisitionType.CONTINUOUS and timeout:
                self._acquisition_condition.wait_for(
                    lambda: self._buffer_idx != self._buffer_needle, timeout=timeout
                )

            if self._buffer_idx == self._buffer_needle:
                return None

            idx = (self._buffer_idx - 1) % self.buffer_size
            sample = self._buffer[idx]
            self._buffer_needle = self._buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        self._ensure_open()
        deadline = monotonic_ns() + int(timeout * 1e9) if timeout >= 0 else None

        if self.acquisition_type == AcquisitionType.FINITE:
            self._start_capture()

            if n < 0 and timeout < 0:
                raise ValueError("For finite acquisition, either n or timeout must be specified.")

            acquired = 0
            while (
                (n < 0 or acquired < n)
                and (deadline is None or monotonic_ns() < deadline)
                and self._acquire_sample() is not None
            ):
                acquired += 1

            self._stop_capture()

        elif self.acquisition_type == AcquisitionType.CONTINUOUS:
            need = min(n if n > 0 else (1 if timeout >= 0 else 0), self.buffer_size - 1)
            with self._acquisition_condition:
                self._wake_at = max(need, 1)
                self._acquisition_condition.wait_for(
                    lambda: not self._is_open or self._buffer_idx - self._buffer_needle >= need,
                    timeout=timeout if timeout >= 0 else None,
                )

        with self._acquisition_condition:
            avail = self._buffer_idx - self._buffer_needle
            if avail <= 0:
                return 0

            count = avail if n < 0 else min(avail, n)

            size = self.buffer_size
            buffer = self._buffer
            needle = self._buffer_needle

            if count > size:
                needle = self._buffer_idx - size
                count = size

            start = needle % size
            end = start + count

            if end <= size:
                data[:count] = buffer[start:end]
            else:
                first = size - start
                data[:first] = buffer[start:]
                data[first:count] = buffer[: count - first]

            self._buffer_needle = needle + count

            return count

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"{type(self).__name__} connection is not open. Call open() first.")

    def _set_buffer(self) -> None:
        spec = self._get_sample_spec()

        self._buffer = zeros(
            self.buffer_size,
            dtype=[("timestamp", "uint64"), *spec],
        )
        self._buffer_idx = 0
        self._buffer_needle = 0

    def _release_buffer(self) -> None:
        del self._buffer
        self._buffer_idx = 0

    def _sample_to_buffer(self, sample: ndarray, timestamp: int) -> None:
        idx = self._buffer_idx % self.buffer_size
        if self._buffer[idx - 1]["timestamp"] == timestamp:
            return

        rec = self._buffer[idx]
        rec["timestamp"] = timestamp
        rec["data"] = sample

        with self._acquisition_condition:
            self._buffer_idx += 1
            if self._buffer_idx - self._buffer_needle >= self._wake_at:
                self._acquisition_condition.notify()

    def _start_acquisition_thread(self) -> None:
        try:
            self._acquisition_thread = Thread(
                target=self._acquisition_thread_func,
                name=f"{type(self).__name__} Acquisition Thread",
                daemon=True,
            )
            self._acquisition_thread.start()
        except Exception as e:
            raise RuntimeError("Acquisition thread failed to start") from e

    def _stop_acquisition_thread(self) -> None:
        self._stop_acquisition_event.set()
        self._acquisition_thread.join(timeout=5)

        if self._acquisition_thread.is_alive():
            LOGGER.warning("Streaming thread did not stop gracefully")

        del self._acquisition_thread

        self._stop_acquisition_event.clear()

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
