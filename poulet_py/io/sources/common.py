try:
    from abc import ABC, abstractmethod
    from typing import Any

    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import BaseDataPacket, BaseSink, BaseStimulus
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseSource(BaseModel, ABC):
    name: str = Field(..., description="Name of the data source")
    _subscribers: list[BaseSink] = PrivateAttr(default_factory=list)

    @abstractmethod
    def _init(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def next(self, stimulus: BaseStimulus) -> Any: ...

    def subscribe(self, sink: BaseSink):
        self._subscribers.append(sink)

    def publish(self, packet: BaseDataPacket):
        for sub in self._subscribers:
            sub.write(packet)

    def open(self):
        self._init()

    def close(self):
        self._close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
