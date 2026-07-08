try:
    from time import time
    from typing import Literal

    from gpiozero import Button
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, BaseTrigger, precise_sleep
except ImportError as e:
    raise ImportError("""
Missing 'gpio' module. Install options:
- Dedicated:    pip install poulet_py[gpio]
- Sub-Module:   pip install poulet_py[triggers]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class GPIOTrigger(BaseTrigger):
    """GPIO-based trigger using gpiozero."""

    pin: int = Field(..., description="GPIO pin number")
    pull_up: bool = Field(default=False, description="Use pull-up resistor")
    edge: Literal["rising", "falling", "both"] = Field("rising", description="Edge to detect")

    _triggered: bool = PrivateAttr(default=False)
    _device: Button = PrivateAttr()

    def _init(self) -> None:
        try:
            self._device = Button(self.pin, pull_up=self.pull_up)
            self._device.when_activated = self._on_rising
            self._device.when_deactivated = self._on_falling
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPIO pin {self.pin}: {e}") from e

    def _close(self) -> None:
        self._device.close()
        del self._device

    def _wait(self) -> bool:
        try:
            start = time()
            self._triggered = False

            while not self._triggered:
                if self.timeout and time() - start > self.timeout:
                    return False
                precise_sleep(0.001)

            return True
        except Exception as e:
            LOGGER.error(f"Error waiting for GPIO event: {e}")
            return False

    def _on_rising(self):
        if self.edge in ("rising", "both"):
            self._triggered = True

    def _on_falling(self):
        if self.edge in ("falling", "both"):
            self._triggered = True
