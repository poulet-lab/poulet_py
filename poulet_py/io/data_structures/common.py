try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from enum import Enum
    from pathlib import Path
    from typing import Any, ClassVar

    from pydantic import BaseModel, Field, PrivateAttr
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DataStructure(int, Enum):
    NONE = 0
    SINGLE_FILE = 1
    FOLDER_PER_TRIAL = 2


class DataSignature(BaseModel):
    data_structure: DataStructure = Field(
        ...,
        description="Expected structure of trial data (e.g. folder per trial, single file, etc.)",
    )
    data_type: type["BaseData"] = Field(
        ...,
        description="Expected type of data contained in trials (e.g. electrophysiology, behavior, etc.)",
    )
    files: Sequence[str] = Field(default_factory=list, description="List of file patterns to match")

    def matches(self, path: Path) -> bool:
        """Check if path matches this signature."""

        if not path.exists():
            return False

        if not self.files:
            return True

        return all(any(path.glob(file_pattern)) for file_pattern in self.files)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DataSignature):
            return NotImplemented

        return (
            self.data_structure == value.data_structure
            and self.data_type == value.data_type
            and set(self.files) == set(value.files)
        )

    def __hash__(self) -> int:
        return hash((self.data_structure, self.data_type, frozenset(self.files)))


class BaseData(BaseModel, ABC):
    DATA_SIGNATURE: ClassVar[DataSignature]

    path: Path = Field(..., description="Path to trial folder")

    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    _paths: Sequence[Path] = PrivateAttr(default_factory=list)

    @property
    def signature(self) -> DataSignature:
        return self.DATA_SIGNATURE

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @abstractmethod
    def summary(self) -> str: ...

    def __str__(self) -> str:
        return self.summary()
