try:
    from abc import ABC, abstractmethod
    from typing import Any

    from numpydantic import NDArray
    from pydantic import BaseModel
except ImportError as e:
    msg = """
Missing 'writers' module. Install options:
- Dedicated:    pip install poulet_py[writers]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DataPacket(BaseModel):
    source: str
    data: NDArray[Any, Any]


class Writer(ABC):
    @abstractmethod
    def write(self, packet: DataPacket): ...
