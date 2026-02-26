from enum import Enum

try:
    from abc import ABC, abstractmethod
    from time import sleep
    from typing import Any

    from numpydantic import NDArray
    from pydantic import BaseModel, Field

    from poulet_py import BaseStimulus, EventBus
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class AcquisitionType(str, Enum):
    CONTINUOUS = "continuous"
    FINITE = "finite"


class BaseSource(BaseModel, ABC):
    name: str = Field(..., description="Name of the data source")
    acquisition_type: AcquisitionType = Field(default=AcquisitionType.FINITE, description="")
    bus: EventBus | None = Field(default=None)

    @abstractmethod
    def _init(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _next(self, stimulus: list[BaseStimulus]) -> Any: ...

    def next(self, stimulus: list[BaseStimulus], isi: int | None = None) -> Any:
        ret = self._next(stimulus)
        if isi:
            sleep(isi / 1000)
        return ret

    def publish(self, payload: dict[str, NDArray[Any, Any]], meta: dict[str, Any] | None = None):
        if self.bus is None:
            msg = "No Event bus is attached"
            raise RuntimeError(msg)

        self.bus.emit(self.name, payload=payload, meta=meta)

    def open(self, bus: EventBus | None = None):
        if not bus and not self.bus:
            msg = "Event bus must be defined"
            raise RuntimeError(msg)

        if bus and not self.bus:
            self.bus = bus

        self._init()

    def close(self):
        self._close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
