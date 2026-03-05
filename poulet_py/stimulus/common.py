try:
    from abc import ABC, abstractmethod
    from typing import Any

    from pydantic import BaseModel, Field
except ImportError as e:
    msg = """
Missing 'stimulus' module. Install options:
- Module:       pip install poulet_py[stimulus]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseStimulus(BaseModel, ABC):
    name: str | None = Field(default=None, description="Optional name of the stimulus")
    isi: int | None = Field(default=None)

    @abstractmethod
    def build(self, *args, **kwargs) -> Any: ...


class EmptyStimulus(BaseStimulus):
    def build(self, *args, **kwargs) -> Any:
        return
