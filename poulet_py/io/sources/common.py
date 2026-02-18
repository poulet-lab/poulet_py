try:
    from abc import ABC, abstractmethod
    from typing import Any

    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import BaseDataSink
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseDataSource(BaseModel, ABC):
    name: str = Field(..., description="Name of the data source")
    _sink: BaseDataSink | None = PrivateAttr(default=None)

    def to(self, sink: BaseDataSink):
        self._sink = sink

    @abstractmethod
    def next(self) -> Any: ...
