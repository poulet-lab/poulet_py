try:
    from time import time_ns
    from typing import Any

    from numpydantic import NDArray
    from pydantic import BaseModel, Field

except ImportError as e:
    msg = """
Missing 'writers' module. Install options:
- Dedicated:    pip install poulet_py[writers]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseEvent(BaseModel):
    name: str = Field(..., description="Name of the data source")
    timestamp: int = Field(default_factory=time_ns, description="Timestamp of the event")


class SinkEvent(BaseEvent):
    payload: dict[str, NDArray[Any, Any]] = Field(
        ..., description="Data fields as a dictionary of numpy arrays"
    )
    meta: dict[str, Any] | None = Field(
        default=None, description="Additional metadata for the data packet"
    )
