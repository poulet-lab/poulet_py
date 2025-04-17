from time import time_ns
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from serial import Serial

from poulet_py.hardware.sensors.qst import TCS, TCSCommand, TCSStimulus


# Fixtures
@pytest.fixture
def mock_serial():
    return MagicMock(spec=Serial)


@pytest.fixture
def basic_stimulus():
    return TCSStimulus(
        surface=1,
        baseline=30.0,
        target=35.0,
        rise_rate=1.0,
        return_speed=1.0,
        duration=100,
    )


@pytest.fixture
def tcs_instance(mock_serial):
    with patch("serial.Serial", return_value=mock_serial):
        tcs = TCS(port="/dev/ttyUSB0", maximum_temperature=40.0)
    return tcs


# Test TCSCommand Enum
class TestTCSCommand:
    def test_command_formatting_success(self):
        assert TCSCommand.BASELINE_TEMPERATURE.format(300) == b"N300"
        assert TCSCommand.TARGET_TEMPERATURE.format(1, 350) == b"C1350"
        assert TCSCommand.SET_MAX_TEMPERATURE.format(400) == b"Om400"

    def test_command_formatting_failure(self):
        with pytest.raises(ValueError):
            TCSCommand.BASELINE_TEMPERATURE.format()  # Missing argument

        with pytest.raises(ValueError):
            TCSCommand.TARGET_TEMPERATURE.format("a", "b")  # Wrong type

    def test_all_commands_have_format_method(self):
        for cmd in TCSCommand:
            assert hasattr(cmd, "format")
            if "%" in cmd.value.decode():
                # Commands with format specifiers should raise if not given args
                with pytest.raises((ValueError, TypeError)):
                    cmd.format()


# Test TCSStimulus Model
class TestTCSStimulus:
    def test_valid_stimulus(self, basic_stimulus):
        assert basic_stimulus.surface == 1
        assert basic_stimulus.baseline == 30.0
        assert basic_stimulus.target == 35.0

    def test_invalid_surface(self):
        with pytest.raises(ValidationError):
            TCSStimulus(surface=6)  # Surface must be <=5

    def test_invalid_temperatures(self):
        with pytest.raises(ValidationError):
            TCSStimulus(baseline=10)  # Below minimum

        with pytest.raises(ValidationError):
            TCSStimulus(target=70)  # Above maximum

    def test_commands_method(self, basic_stimulus):
        commands = basic_stimulus.commands()
        assert len(commands) == 5
        assert commands[0] == TCSCommand.BASELINE_TEMPERATURE.format(300)
        assert commands[1] == TCSCommand.TARGET_TEMPERATURE.format(1, 350)


# Test TCS Model
class TestTCS:
    def test_valid_port(self):
        valid_ports = ["COM3", "/dev/ttyUSB0", "/dev/tty.usbmodem123"]
        for port in valid_ports:
            tcs = TCS(port=port, maximum_temperature=40.0)
            assert tcs.port == port

    def test_invalid_port(self):
        with pytest.raises(ValidationError):
            TCS(port="invalid_port", maximum_temperature=40.0)

    def test_max_temperature_validation(self):
        with pytest.raises(ValidationError):
            TCS(port="COM1", maximum_temperature=70)  # Above max

    def test_serial_property_caching(self, tcs_instance, mock_serial):
        # First access creates and caches the serial property
        serial1 = tcs_instance.serial
        # Second access should return the same instance
        serial2 = tcs_instance.serial
        assert serial1 is serial2
        mock_serial.assert_called_once()

    def test_stimulus_property(self, tcs_instance, basic_stimulus):
        # Test default stimulus
        assert isinstance(tcs_instance.stimulus, TCSStimulus)

        # Test setting valid stimulus
        tcs_instance.stimulus = basic_stimulus
        assert tcs_instance.stimulus == basic_stimulus

        # Test setting invalid stimulus type
        with pytest.raises(ValueError):
            tcs_instance.stimulus = "not a stimulus"

        # Test stimulus with too high target temperature
        with pytest.raises(ValueError):
            tcs_instance.stimulus = TCSStimulus(target=50)  # Above max of 40

    def test_write_method(self, tcs_instance, mock_serial):
        test_command = b"test"
        tcs_instance.write(test_command)
        mock_serial.flush.assert_called_once()
        mock_serial.write.assert_called_with(test_command)

    def test_read_method(self, tcs_instance, mock_serial):
        mock_serial.read_until.return_value = b"test response\r\n"
        timestamp, response = tcs_instance.read()
        assert isinstance(timestamp, int)
        assert response == "test response\n"

    def test_init_method(self, tcs_instance, mock_serial):
        mock_serial.read_until.return_value = b"Firmware: v1.0\nProbe ID: 123\n"
        tcs_instance.init()
        mock_serial.write.assert_any_call(
            TCSCommand.AUTOMATIC_CALIBRATION.format()
        )
        mock_serial.write.assert_any_call(
            TCSCommand.SET_MAX_TEMPERATURE.format(400)
        )

    def test_trigger_method(self, tcs_instance, basic_stimulus, mock_serial):
        tcs_instance.stimulus = basic_stimulus
        tcs_instance.trigger()
        # Should write all stimulus commands plus trigger
        assert mock_serial.write.call_count == 6  # 5 commands + trigger

    def test_trigger_with_beep(self, tcs_instance, basic_stimulus, mock_serial):
        tcs_instance.beep = True
        tcs_instance.stimulus = basic_stimulus
        tcs_instance.trigger()
        # Should write all stimulus commands plus beep plus trigger
        assert mock_serial.write.call_count == 7  # 5 commands + beep + trigger

    def test_close_method(self, tcs_instance, mock_serial):
        tcs_instance.close()
        mock_serial.reset_input_buffer.assert_called_once()
        mock_serial.reset_output_buffer.assert_called_once()
        mock_serial.close.assert_called_once()

    def test_reset_method(self, tcs_instance, mock_serial):
        tcs_instance.reset()
        mock_serial.write.assert_called_with(TCSCommand.RESET.format())

    def test_get_readings_method(self, tcs_instance, mock_serial):
        mock_response = b"\n30.0 31.0 32.0 33.0 34.0 35.0\n"
        mock_serial.read_until.return_value = mock_response
        readings = tcs_instance.get_readings()
        assert readings == {
            "neutral": 3.0,
            "s1": 3.1,
            "s2": 3.2,
            "s3": 3.3,
            "s4": 3.4,
            "s5": 3.5,
            "time": pytest.approx(time_ns(), rel=1e6),  # Allow 1ms difference
        }

    def test_get_readings_invalid_data(self, tcs_instance, mock_serial):
        mock_serial.read_until.return_value = b"invalid data\n"
        readings = tcs_instance.get_readings()
        assert readings == {}
