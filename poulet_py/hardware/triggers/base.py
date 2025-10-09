try:
    from abc import ABC, abstractmethod

    from pydantic import BaseModel, Field
except ImportError as e:
    msg = """
Missing 'triggers' module. Install options:
- Dedicated:    pip install poulet_py[triggers]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaseTrigger(BaseModel, ABC):
    """Abstract base class for trigger devices."""

    name: str = Field("", description="Name of the trigger device")
    timeout: float | None = Field(None, description="Timeout in seconds for waiting")

    @abstractmethod
    def wait(self) -> bool:
        """Wait for trigger event."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup trigger resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
