try:
    from abc import abstractmethod
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
    queue_size: int = Field(default=1000, description="Size of the internal queue")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )
    bus: EventBus | None = Field(default=None)

    _queue: Queue = PrivateAttr()
    _thread: Thread = PrivateAttr()
    _running: bool = PrivateAttr(default=False)

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _on_event(self, event: BaseEvent): ...

    def _run(self):
        while True:
            event = self._queue.get()

            if event is None:  # sentinel
                break

            try:
                self._on_event(event)
            except Exception as e:
                LOGGER.exception("Error while writing packet: %s", e)

            finally:
                self._queue.task_done()

    def on_event(self, event: BaseEvent):
        if not self._running:
            msg = "Sink not initialized. Call 'open()' first."
            raise RuntimeError(msg)

        try:
            self._queue.put_nowait(event)
        except Full:
            LOGGER.warning("Sink queue full — dropping packet")

    def open(self, bus: EventBus | None = None):
        if self._running:
            LOGGER.warning("Sink already running.")
            return

        if bus and not self.bus:
            self.bus = bus

        if not self.bus:
            msg = "Event bus must be defined"
            raise RuntimeError(msg)

        self.bus.subscribe(self)

        self._queue = Queue(maxsize=self.queue_size)
        self._running = True

        self._open()

        self._thread = Thread(target=self._run, daemon=False)
        self._thread.start()

    def close(self):
        if not self._running:
            return

        self._running = False
        self._queue.put(None)
        self._thread.join()
        self._close()

    def __enter__(self, bus: EventBus | None = None):
        self.open(bus=bus)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
