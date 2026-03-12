try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor

    from pydantic import BaseModel, PrivateAttr

    from poulet_py import BaseEvent

except ImportError as e:
    msg = """
Missing 'event' module. Install options:
- Dedicated:    pip install poulet_py[event]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class EventHandler(BaseModel, ABC):
    @abstractmethod
    def on_event(self, event: BaseEvent) -> None: ...


class EventBus(BaseModel):
    _subs: Sequence[EventHandler] = PrivateAttr(default_factory=list)
    _executor: ThreadPoolExecutor | None = PrivateAttr(None)

    def subscribe(self, handler: EventHandler):
        self._subs.append(handler)

    def emit(self, event: BaseEvent):
        if not self._executor:
            msg = "EventBus must be opened first"
            raise RuntimeError(msg)

        for handlers in self._subs:
            self._executor.submit(handlers.on_event, event)

    def open(self, max_workers: int | None = None):
        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        if self._executor:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self, max_workers: int | None = None):
        self.open(max_workers=max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
