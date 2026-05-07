try:
    from abc import ABC, abstractmethod
    from concurrent.futures import ThreadPoolExecutor

    from pydantic import BaseModel, Field, PrivateAttr

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
    max_workers: int | None = Field(
        default=None, description="Maximum number of threads for handling events"
    )
    _subs: list[EventHandler] = PrivateAttr(default_factory=list)
    _executor: ThreadPoolExecutor = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)

    def subscribe(self, handler: EventHandler):
        self._ensure_open()
        self._subs.append(handler)

    def emit(self, event: BaseEvent):
        self._ensure_open()

        for handlers in self._subs:
            self._executor.submit(handlers.on_event, event)

    def open(self):
        if self._is_open:
            return

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        self._executor.shutdown()
        del self._executor
        self._is_open = False

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
