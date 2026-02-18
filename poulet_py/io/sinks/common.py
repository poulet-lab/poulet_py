try:
    from abc import ABC, abstractmethod
    from queue import Full, Queue
    from threading import Thread
    from typing import Any

    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, BaseDataPacket
except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseDataSink(BaseModel, ABC):
    queue_size: int = Field(default=1000, description="Size of the internal queue")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )

    _queue: Queue | None = PrivateAttr(None)
    _running: bool = PrivateAttr(False)
    _thread: Thread | None = PrivateAttr(None)

    @abstractmethod
    def _init(self): ...
    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _write(self, packet: BaseDataPacket): ...

    def _run(self):
        while True:
            packet = self._queue.get()

            if packet is None:  # sentinel
                break

            try:
                self._write(packet)
            except Exception as e:
                LOGGER.exception("Error while writing packet: %s", e)

            finally:
                self._queue.task_done()

    def push(self, packet: BaseDataPacket):
        if not self._running:
            msg = "DataSink not initialized. Call 'open()' first."
            raise RuntimeError(msg)

        try:
            self._queue.put_nowait(packet)
        except Full:
            LOGGER.warning("DataSink queue full — dropping packet")

    def open(self):
        if self._running:
            LOGGER.warning("DataSink already running.")
            return

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

        self._thread = None
        self._queue = None
        self._close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
