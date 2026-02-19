from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class BaseStimulus(BaseModel, ABC):
    name: str | None = Field(default=None, description="Optional name of the stimulus")

    @abstractmethod
    def build(self, *args, **kwargs) -> Any: ...
