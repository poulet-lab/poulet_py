try:
    from abc import ABC, abstractmethod

    from pydantic import BaseModel, Field, PrivateAttr
except ImportError as e:
    raise ImportError("""
Missing 'triggers' module. Install options:
- Dedicated:    pip install poulet_py[triggers]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class BaseTrigger(BaseModel, ABC):
    """Abstract base class for trigger devices."""

    name: str = Field(default="", description="Name of the trigger device")
    timeout: float | None = Field(default=None, description="Timeout in seconds for waiting")

    _is_open: bool = PrivateAttr(default=False)

    @abstractmethod
    def _init(self) -> None: ...

    @abstractmethod
    def _close(self) -> None: ...

    @abstractmethod
    def _wait(self) -> bool: ...

    def model_post_init(self, __context):
        self._init()

    def wait(self) -> bool:
        return self._wait()

    def close(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
