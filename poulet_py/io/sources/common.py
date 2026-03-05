try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from enum import Enum

    from pydantic import BaseModel, Field

    from poulet_py import BaseEvent, BaseStimulus, EventBus
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class AcquisitionType(str, Enum):
    NONE = "None"
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
    def _supports(self, stimuli: Sequence[BaseStimulus]) -> Sequence[BaseStimulus]: ...

    @abstractmethod
    def _fire(self, stimuli: Sequence[BaseStimulus]) -> bool: ...

    @abstractmethod
    def _publish(self, stimuli: Sequence[BaseStimulus]) -> bool: ...

    def fire(self, stimuli: Sequence[BaseStimulus]) -> bool:
        st = self._supports(stimuli)
        self._fire(st)
        self._publish(st)

        return True

    def publish(self, event: BaseEvent):
        if self.bus is None:
            msg = "No Event bus is attached"
            raise RuntimeError(msg)

        self.bus.emit(event)

    def open(self, bus: EventBus | None = None):
        if not bus and not self.bus:
            msg = "Event bus must be defined"
            raise RuntimeError(msg)

        if bus and not self.bus:
            self.bus = bus

        self._init()

    def close(self):
        self._close()

    def __enter__(self, bus: EventBus | None = None):
        self.open(bus=bus)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
