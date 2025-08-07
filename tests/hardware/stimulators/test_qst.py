import re
from time import time_ns
from unittest.mock import Mock, call, patch

import pytest

from poulet_py import TCS, TCSCommand, TCSStimulus


# Fixtures
@pytest.fixture
def mock_serial():
    with patch("serial.Serial") as mock:
        yield mock


@pytest.fixture
def tcs(mock_serial):
    return TCS(port="/dev/ttyUSB0")


@pytest.fixture
def stimulus():
    return TCSStimulus(
        surface=1,
        baseline=30.0,
        target=35.0,
        rise_rate=1.0,
        return_speed=1.0,
        duration=100,
    )


# Test cases
class TestTCSCommand:
    def test_command_formatting(self):
        assert TCSCommand.TARGET_TEMPERATURE.format(1, 350) == b"C1350"
        assert TCSCommand.BASELINE_TEMPERATURE.format(300) == b"N300"
        assert TCSCommand.STIMULATION_RATE.format(1, 10) == b"V00110"

    def test_invalid_formatting(self):
        with pytest.raises(ValueError):
            TCSCommand.TARGET_TEMPERATURE.format(1)  # Missing argument


class TestTCSStimulus:
    def test_stimulus_validation(self):
        with pytest.raises(ValueError):
            TCSStimulus(surface=6)  # Invalid surface

        with pytest.raises(ValueError):
            TCSStimulus(target=70)  # Too hot

    def test_commands_generation(self, stimulus):
        commands = stimulus.commands()
        assert len(commands) == 6
        assert commands[0] == b"S10000"  # Surface selection
        assert commands[1] == b"N300"  # Baseline
        assert commands[2] == b"C1350"  # Target
        assert commands[3] == b"V10010"  # Rise rate
        assert commands[4] == b"D100100"  # Duration
        assert commands[5] == b"R10010"  # Return speed


class TestTCSInitialization:
    def test_init_creates_serial_connection(self, mock_serial, tcs):
        tcs.init()
        mock_serial.assert_called_once_with(
            port="/dev/ttyUSB0",
            baudrate=115200,
            bytesize=8,
            parity="N",
            timeout=2,
        )

    def test_init_sets_max_temperature(self, tcs, mock_serial):
        tcs.maximum_temperature = 45.0
        tcs.init()
        mock_serial.return_value.write.assert_any_call(b"Om450")

    def test_init_starts_reader_thread(self, tcs):
        tcs.init()
        assert hasattr(tcs, "_stop_event")
        assert tcs.thread.is_alive()

    def test_close_stops_thread_and_closes_serial(self, tcs, mock_serial):
        tcs.init()
        tcs.close()
        assert tcs.stop_event is True
        mock_serial.return_value.close.assert_called_once()


class TestTCSCommandExecution:
    def test_write_command(self, tcs, mock_serial):
        tcs.init()
        tcs.write(b"TEST")
        mock_serial.return_value.write.assert_called_with(b"TEST")

    def test_execute_command(self, tcs):
        tcs.init()
        with (
            patch.object(tcs, "write") as mock_write,
            patch.object(tcs, "_expect_response") as mock_expect,
        ):
            tcs.execute_command(TCSCommand.READ_INFO, expected_pattern=re.compile(".*"))
            mock_write.assert_called_once_with(b"H")
            mock_expect.assert_called_once()

    def test_expect_response(self, tcs):
        tcs.init()
        pattern = re.compile(r"TEST(\d+)")
        test_data = "TEST123\n"

        # Mock the serial read
        mock_serial = tcs.serial
        mock_serial.read_until.return_value = test_data.encode()

        # Mock current_search mechanism
        tcs.current_search = (pattern, Mock(), None)

        # This would normally be done by the reader thread
        result = tcs._expect_response(pattern)

        assert result is not None
        timestamp, match = result
        assert match.group(1) == "123"


class TestTCSFunctionality:
    def test_trigger_stimulation(self, tcs, stimulus):
        tcs.init()
        tcs.stimulus = stimulus

        with patch.object(tcs, "write") as mock_write:
            tcs.trigger()
            calls = [
                call(b"S10000"),
                call(b"N300"),
                call(b"C1350"),
                call(b"V10010"),
                call(b"D100100"),
                call(b"R10010"),
                call(b"L"),
            ]
            mock_write.assert_has_calls(calls)

    def test_get_readings(self, tcs):
        tcs.init()
        test_response = "300 310 320 330 340 350\n"

        with (
            patch.object(tcs.serial, "read_until") as mock_read,
            patch.object(tcs, "_expect_response") as mock_expect,
        ):
            mock_read.return_value = test_response.encode()
            mock_expect.return_value = (time_ns(), re.match(r"(\d{3})", "300"))

            readings = tcs.get_readings()
            assert readings["neutral"] == 30.0
            assert readings["s1"] == 31.0

    def test_context_manager(self, mock_serial):
        with TCS(port="/dev/ttyUSB0") as tcs:
            tcs.init = Mock()
            tcs.close = Mock()
        tcs.init.assert_called_once()
        tcs.close.assert_called_once()


class TestThreadSafety:
    def test_concurrent_access(self, tcs):
        tcs.init()

        # Simulate concurrent access to current_search
        from threading import Thread

        def worker():
            tcs.current_search = (re.compile(".*"), Mock(), None)

        threads = [Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any threading-related exceptions


class TestErrorHandling:
    def test_serial_error_on_write(self, tcs, mock_serial):
        mock_serial.return_value.write.side_effect = OSError("Serial error")
        tcs.init()
        with pytest.raises(RuntimeError):
            tcs.write(b"TEST")

    def test_timeout_on_response(self, tcs):
        tcs.init()
        with patch.object(tcs, "_expect_response", return_value=None):
            with pytest.raises(RuntimeError):
                tcs.get_readings()
