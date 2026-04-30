try:
    from abc import ABC, abstractmethod
    from datetime import datetime
    from pathlib import Path

    from pydantic import BaseModel, Field, PrivateAttr, model_validator
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseData(BaseModel, ABC):
    path: Path = Field(..., description="Path to trial folder")
    start: datetime | int | None = Field(default=None)
    end: datetime | int | None = Field(default=None)

    _is_open: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def validate_path(self):
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        msg = ""
        if not self.path.exists():
            msg = f"Data Path {self.path} does not exist"
        if not self.path.is_dir():
            msg = f"Data Path must be a directory: {self.path}"
        if msg:
            raise ValueError(msg)

        return self

    @abstractmethod
    def _open(self): ...

    @abstractmethod
    def _close(self): ...

    @abstractmethod
    def _should_open(self) -> bool: ...

    def open(self) -> None:
        self._open()

    def close(self) -> None:
        self._close()

    def _ensure_open(self):
        if not self._is_open:
            msg = f"{type(self)} need to be opened first"
            raise RuntimeError(msg)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
