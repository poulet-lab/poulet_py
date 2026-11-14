try:
    from abc import abstractmethod
    from collections.abc import Callable
    from queue import Full, Queue
    from threading import Thread
    from time import monotonic_ns, time_ns
    from types import TracebackType
    from typing import Any, Literal, Self

    from numpy import ndarray
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, BaseEvent, EventBus, EventHandler, SinkEvent
except ImportError as e:
    raise ImportError("""
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class BaseSink(EventHandler):
    name: str | None = Field(default=None, description="Name of the sink")
    queue_size: int = Field(default=1000, description="Size of the internal queue")
    convert_timestamp: Literal["datetime", "index"] | Callable[[ndarray], ndarray] | None = Field(
        default=None
    )
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )

    _time_ns: int = PrivateAttr(default_factory=time_ns)
    _mono_ns: int = PrivateAttr(default_factory=monotonic_ns)
    _last_ts: int = PrivateAttr(default=0)
    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _sink_queue: Queue[BaseEvent | None] = PrivateAttr()
    _sink_thread: Thread = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _close(self) -> None: ...

    @abstractmethod
    def _on_event(self, event: BaseEvent) -> None: ...

    def _keyboard_controls(self) -> dict[str, tuple[str, Callable]]:
        """
        Return a dict of keyboard shortcuts and their handlers.
        Format: {key_combination: (description, callback)}
        Example: {'ctrl+r': ('Reset device', self.reset_device)}
        """
        ...

    def on_event(self, event: BaseEvent) -> None:
        self._ensure_open()

        try:
            self._sink_queue.put_nowait(event)
        except Full:
            LOGGER.warning("Sink queue full — dropping packet")

    @property
    def bus(self) -> EventBus:
        return self._bus

    @bus.setter
    def bus(self, value: EventBus) -> None:
        if self._is_open:
            raise RuntimeError(f"Cannot change bus while {self.name} is open")

        self._bus = value
        self._external_bus = True

    def open(self) -> None:
        if self._is_open:
            return

        if not self._external_bus:
            self.bus.open()

        self.bus.subscribe(self)

        self._sink_queue = Queue(maxsize=self.queue_size)

        self._open()

        self._sink_thread = Thread(
            target=self._run, name=f"{type(self).__name__}Thread", daemon=True
        )
        self._sink_thread.start()

        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return

        self._sink_queue.put(None)
        self._sink_thread.join()
        self._close()

        if not self._external_bus:
            self.bus.close()

        self._is_open = False

    def _run(self) -> None:
        while True:
            event: BaseEvent | None = self._sink_queue.get()

            if event is None:
                self._sink_queue.task_done()
                break

            if not isinstance(event, SinkEvent):
                continue

            try:
                self._timestamp_converter(event)
                self._on_event(event)
            except Exception as e:
                LOGGER.exception("Error while writing packet: %s", e)
            finally:
                self._sink_queue.task_done()

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"{type(self)} need to be opened first")

    def _timestamp_converter(self, event: SinkEvent) -> SinkEvent:
        if isinstance(event.payload, ndarray):
            if self.convert_timestamp == "datetime":
                event.payload["timestamp"][:] = (
                    event.payload["timestamp"][:] - self._mono_ns + self._time_ns
                )
            elif self.convert_timestamp == "index":
                event.payload["timestamp"][:] = event.payload["timestamp"][:] - self._mono_ns
            elif isinstance(self.convert_timestamp, Callable):
                event.payload["timestamp"][:] = self.convert_timestamp(event.payload["timestamp"])

        return event

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
