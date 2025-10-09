try:
    from time import sleep, time
    from typing import Literal

    from gpiozero import Button
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, BaseTrigger
except ImportError as e:
    msg = """
Missing 'gpio' module. Install options:
- Dedicated:    pip install poulet_py[gpio]
- Sub-Module:   pip install poulet_py[triggers]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class GPIOTrigger(BaseTrigger):
    """GPIO-based trigger using gpiozero."""

    pin: int = Field(..., description="GPIO pin number")
    pull_up: bool = Field(False, description="Use pull-up resistor")
    edge: Literal["rising", "falling", "both"] = Field("rising", description="Edge to detect")

    _triggered: bool = PrivateAttr(False)
    _device: Button | None = PrivateAttr(None)

    def __init__(self, **data):
        super().__init__(**data)
        self._setup()

    def _setup(self) -> None:
        """Setup GPIO device."""
        try:
            self._device = Button(self.pin, pull_up=self.pull_up)
            self._device.when_activated = self._on_rising
            self._device.when_deactivated = self._on_falling
        except Exception as e:
            msg = f"Failed to initialize GPIO pin {self.pin}: {e}"
            raise RuntimeError(msg) from e

    def _on_rising(self):
        if self.edge in ("rising", "both"):
            self._triggered = True

    def _on_falling(self):
        if self.edge in ("falling", "both"):
            self._triggered = True

    def wait(self) -> bool:
        """Wait for GPIO event."""
        try:
            start = time()
            self._triggered = False

            while not self._triggered:
                if self.timeout and time() - start > self.timeout:
                    return False
                sleep(0.001)

            return True
        except Exception as e:
            LOGGER.error(f"Error waiting for GPIO event: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup GPIO resources."""
        if self._device is not None:
            self._device.close()
            self._device = None
