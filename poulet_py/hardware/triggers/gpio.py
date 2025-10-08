try:
    from time import sleep, time
    from typing import Literal

    from gpiozero import DigitalInputDevice
    from pydantic import Field

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
    pull_up: bool = Field(True, description="Use pull-up resistor")
    active_state: bool = Field(True, description="Active state (True=high, False=low)")
    edge: Literal["rising", "falling", "both"] = Field("rising", description="Edge to detect")

    def __init__(self, **data):
        super().__init__(**data)
        self._device: DigitalInputDevice | None = None
        self._setup()

    def _setup(self) -> None:
        """Setup GPIO device."""
        try:
            self._device = DigitalInputDevice(
                self.pin, pull_up=self.pull_up, active_state=self.active_state
            )
        except Exception as e:
            msg = f"Failed to initialize GPIO pin {self.pin}: {e}"
            raise RuntimeError(msg) from e

    def wait(self) -> bool:
        """Wait for GPIO event."""
        if self._device is None or not self.active:
            return True

        try:
            timeout = self.timeout
            start_time = time()

            while True:
                if self._device.value == self.active_state:
                    return True

                if timeout is not None and time() - start_time < timeout:
                    break

                sleep(0.000001)
            return False
        except Exception as e:
            LOGGER.error(f"Error waiting for GPIO event: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup GPIO resources."""
        if self._device is not None:
            self._device.close()
            self._device = None
