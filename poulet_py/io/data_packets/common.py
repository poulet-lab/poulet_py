try:
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


class BaseDataPacket(BaseModel):
    name: str = Field(..., description="Name of the data source")
    data: dict[str, NDArray[Any, Any]] = Field(
        ..., description="Data fields as a dictionary of numpy arrays"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the data packet"
    )
