try:
    from abc import abstractmethod
    from collections.abc import Callable
    from queue import Full, Queue
    from threading import Thread
    from typing import Any

    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, BaseEvent, EventBus, EventHandler
except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseSink(EventHandler):
    name: str = Field(..., description="Name of the sink")
    queue_size: int = Field(default=1000, description="Size of the internal queue")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )

    _bus: EventBus = PrivateAttr(default_factory=EventBus)
    _external_bus: bool = PrivateAttr(default=False)

    _queue: Queue = PrivateAttr()
    _thread: Thread = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _on_event(self, event: BaseEvent): ...

    def _keyboard_controls(self) -> dict[str, tuple[str, Callable]]:
        """
        Return a dict of keyboard shortcuts and their handlers.
        Format: {key_combination: (description, callback)}
        Example: {'ctrl+r': ('Reset device', self.reset_device)}
        """
        ...

    def on_event(self, event: BaseEvent):
        self._ensure_open()

        try:
            self._queue.put_nowait(event)
        except Full:
            LOGGER.warning("Sink queue full — dropping packet")

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

        self.bus.subscribe(self)

        self._queue = Queue(maxsize=self.queue_size)

        self._open()

        self._thread = Thread(target=self._run, name=f"{type(self).__name__}Thread", daemon=True)
        self._thread.start()

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        self._queue.put(None)
        self._thread.join()
        self._close()

        if not self._external_bus:
            self.bus.close()

        self._is_open = False

    def _run(self):
        while True:
            event = self._queue.get()

            if event is None:  # sentinel
                self._queue.task_done()
                break

            try:
                self._on_event(event)
            except Exception as e:
                LOGGER.exception("Error while writing packet: %s", e)
            finally:
                self._queue.task_done()

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
