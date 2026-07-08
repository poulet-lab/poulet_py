try:
    from abc import ABC, abstractmethod
    from typing import Any

    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError("""
Missing 'stim' module. Install options:
- Module:       pip install poulet_py[stim]
- Full:         pip install poulet_py[all]
""") from e


class BaseStimulus(BaseModel, ABC):
    name: str | None = Field(default=None, description="Optional name of the stimulus")
    duration: int = Field(..., ge=1)
    pre_delay: int = Field(default=0, ge=0)
    post_delay: int = Field(default=0, ge=0)

    @abstractmethod
    def build(self, *args, **kwargs) -> Any: ...


class EmptyStimulus(BaseStimulus):
    def build(self, *args, **kwargs) -> Any:
        return
