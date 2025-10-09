from unittest.mock import MagicMock, patch

import pytest
from gpiozero import DigitalInputDevice

from poulet_py import GPIOTrigger, KeyboardTrigger


# Test for GPIOTrigger class
@pytest.fixture
def gpio_trigger():
    return GPIOTrigger(pin=17, timeout=2.0)


def test_gpio_trigger_wait_success(gpio_trigger):
    # Mock DigitalInputDevice to simulate pin value
    with patch.object(DigitalInputDevice, "value", new_callable=MagicMock) as mock_value:
        mock_value.return_value = True  # Simulate the pin being active
        result = gpio_trigger.wait()
        assert result is True, "GPIO trigger should return True when pin is active"


def test_gpio_trigger_wait_timeout(gpio_trigger):
    # Mock DigitalInputDevice to simulate pin value
    with patch.object(DigitalInputDevice, "value", new_callable=MagicMock) as mock_value:
        mock_value.return_value = False  # Simulate the pin being inactive
        result = gpio_trigger.wait()
        assert result is False, "GPIO trigger should return False when timeout occurs"


# Test for KeyboardTrigger class
@pytest.fixture
def keyboard_trigger():
    return KeyboardTrigger(key="a", timeout=1)


def test_keyboard_trigger_wait_success(keyboard_trigger):
    # Mock input function to simulate user pressing the "enter" key
    with patch("builtins.input", return_value="a"):
        result = keyboard_trigger.wait()
        assert result is True, "Keyboard trigger should return True when the correct key is pressed"


@pytest.mark.timeout(1)
def test_keyboard_trigger_wait_failure(keyboard_trigger):
    # Mock input function to simulate user pressing a wrong key
    with patch("builtins.input", return_value="b"):
        result = keyboard_trigger.wait()
        assert result is False, "Keyboard trigger should return False when a wrong key is pressed"
