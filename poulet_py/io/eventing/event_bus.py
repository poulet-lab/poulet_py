try:
    from typing import Any, TYPE_CHECKING

    from numpydantic import NDArray
    from pydantic import BaseModel, PrivateAttr

    from poulet_py import Event

    if TYPE_CHECKING:
        from poulet_py import BaseSink
except ImportError as e:
    msg = """
Missing 'writers' module. Install options:
- Dedicated:    pip install poulet_py[writers]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class EventBus(BaseModel):
    _subscribers: list["BaseSink"] = PrivateAttr(default_factory=list)

    def subscribe(self, handler: "BaseSink"):
        self._subscribers.append(handler)

    def emit(
        self, name: str, payload: dict[str, NDArray[Any, Any]], meta: dict[str, Any] | None = None
    ):
        if meta is None:
            meta = {}
        event = Event(name=nameƒ, payload=payload, meta=meta)
        for handler in self._subscribers:
            handler.handle(event)
