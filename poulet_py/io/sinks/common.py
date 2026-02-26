try:
    from abc import ABC, abstractmethod
    from queue import Full, Queue
    from threading import Thread
    from typing import Any

    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, Event, EventBus
except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseSink(BaseModel, ABC):
    queue_size: int = Field(default=1000, description="Size of the internal queue")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )
    bus: EventBus | None = Field(default=None)

    _queue: Queue = PrivateAttr()
    _thread: Thread = PrivateAttr()
    _running: bool = PrivateAttr(default=False)

    @abstractmethod
    def _init(self): ...
    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _handle(self, event: Event): ...

    def _run(self):
        while True:
            event = self._queue.get()

            if event is None:  # sentinel
                break

            try:
                self._handle(event)
            except Exception as e:
                LOGGER.exception("Error while writing packet: %s", e)

            finally:
                self._queue.task_done()

    def handle(self, event: Event):
        if not self._running:
            msg = "DataSink not initialized. Call 'open()' first."
            raise RuntimeError(msg)

        try:
            self._queue.put_nowait(event)
        except Full:
            LOGGER.warning("DataSink queue full — dropping packet")

    def open(self, bus: EventBus | None = None):
        if self._running:
            LOGGER.warning("DataSink already running.")
            return

        if bus and not self.bus:
            self.bus = bus

        if not self.bus:
            msg = "Event bus must be defined"
            raise RuntimeError(msg)

        self.bus.subscribe(self)

        self._queue = Queue(maxsize=self.queue_size)
        self._running = True

        self._init()

        self._thread = Thread(target=self._run, daemon=False)
        self._thread.start()

    def close(self):
        if not self._running:
            return

        self._running = False
        self._queue.put(None)
        self._thread.join()
        self._close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
