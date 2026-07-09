"""
QST TCS (Temperature Control System) module for controlling Peltier-based thermal stimulation devices.

This module provides classes for communicating with and controlling TCS devices
via serial interface, including temperature monitoring, stimulus presentation,
and calibration functionality.
"""

try:
    from collections import deque
    from re import Match, Pattern, compile, search
    from threading import Condition, Event, Thread
    from time import monotonic_ns

    from numpy import ndarray, zeros
    from pydantic import BaseModel, Field, PrivateAttr
    from serial import Serial

    from poulet_py import LOGGER, AcquisitionType, TCSCommand, TCSStimulus, precise_sleep
except ImportError as e:
    raise ImportError("""
Missing 'qst' module. Install options:
- Dedicated:    pip install poulet_py[qst]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class TCSSerialSearchRequest(BaseModel):
    """
    Request object for searching patterns in serial data stream.

    This class encapsulates a request to search for a specific regex pattern
    in incoming serial data, providing synchronization primitives for
    asynchronous pattern matching.

    Parameters
    ----------
    pattern : Pattern[bytes] | None, default=None
        Regex pattern to search for in incoming serial data

    Attributes
    ----------
    _result : tuple[int, Match[bytes] | None] | None
        Private attribute storing the search result (timestamp, match)
    _event : Event
        Private attribute for thread synchronization
    """

    pattern: Pattern[bytes] | None = Field(
        default=None, description="Regex pattern to search for in incoming serial data"
    )
    _result: tuple[int, Match[bytes] | None] | None = PrivateAttr(default=None)
    _event: Event = PrivateAttr(default_factory=Event)

    @property
    def event(self) -> Event:
        """Get the synchronization event."""
        return self._event

    @event.setter
    def event(self, value: Event) -> None:
        """Set the synchronization event."""
        self._event = value

    @property
    def result(self) -> tuple[int, Match[bytes] | None] | None:
        """Get the search result (timestamp, match) or None if not found."""
        return self._result

    @result.setter
    def result(self, value: tuple[int, Match[bytes] | None] | None) -> None:
        """Set the search result."""
        self._result = value


class TCS(BaseModel, validate_assignment=True):
    r"""
    Temperature Control System (TCS) main interface class.

    This class provides comprehensive control over TCS devices including
    temperature monitoring, stimulus presentation, calibration, and
    data acquisition.

    Parameters
    ----------
    port : str
        Serial port to which the TCS device is connected.
        Must match pattern: ^(COM\d+|(/dev/)tty(USB\d+|\.usb[a-zA-Z0-9]+))$
    buffer_size : int, default=1000
        Size of the internal sampling queue. Must be >= 1.
    maximum_temperature : float, default=40.0
        Maximum allowed temperature in °C.
    beep : bool, default=False
        Whether to enable audible beeps during stimulus presentation.
    trigger_out_channel : int, default=255
        Output channel for trigger signals (0-255).
    read_timeout : float, default=2.0
        Timeout for read operations in seconds.
    response_timeout : float, default=2.0
        Timeout for device responses in seconds.

    Attributes
    ----------
    _serial : Serial
        PySerial object for serial communication.
    _is_open : bool
        Flag indicating if the serial connection is open.
    _tcs_buffer_idx : int
        Current index in the circular buffer.
    _tcs_buffer : ndarray
        Circular buffer for temperature samples.
    _acquisition_thread : Thread
        Background thread for continuous data acquisition.
    _stop_acquisition_thread : Event
        Event to signal acquisition thread termination.
    _acquisition_cond : Condition
        Condition variable for coordinating buffer access.
    _stimulus_running : bool
        Flag indicating if a stimulus is currently running.
    _temperature_line_pattern : Pattern[bytes]
        Regex pattern for parsing temperature data lines.
    _serial_search_queue : deque[TCSSerialSearchRequest]
        Queue of pending pattern search requests.

    Examples
    --------
    >>> with TCS(port="COM3", maximum_temperature=42.0) as device:
    ...     info = device.info()
    ...     battery = device.battery_info()
    ...     calibration_offset = device.calibration()
    ...     device.trigger(stimulus)
    ...     samples = device.read_many_sample(data_array, n=100)
    """

    port: str = Field(
        ...,
        description="Serial port to which the TCS device is connected",
        pattern=r"^(COM\d+|(/dev/)tty(USB\d+|\.usb[a-zA-Z0-9]+))$",
    )
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="Type of data acquisition, continuous or finite"
    )  # TODO
    buffer_size: int = Field(default=100, description="Size of the internal sampling queue", ge=1)
    maximum_temperature: float = Field(
        default=40.0, description="Maximum allowed temperature in °C"
    )
    beep: bool = Field(
        default=False, description="Whether to enable audible beeps during stimulus presentation"
    )
    trigger_out_channel: int = Field(
        default=255, description="Output channel for trigger signals (0-255)"
    )
    read_timeout: float = Field(default=2.0, description="Timeout for read operations in seconds")
    response_timeout: float = Field(
        default=2.0, description="Timeout for device responses in seconds"
    )

    _serial: Serial = PrivateAttr(default_factory=Serial)
    _is_open: bool = PrivateAttr(default=False)

    _tcs_buffer_idx: int = PrivateAttr(0)
    _tcs_buffer_needle: int = PrivateAttr(0)
    _tcs_buffer: ndarray = PrivateAttr()

    _stimulus: TCSStimulus = PrivateAttr()
    _done_trigger: Event = PrivateAttr(default_factory=Event)
    _start_trigger: Event = PrivateAttr(default_factory=Event)
    _stop_trigger_thread: Event = PrivateAttr(default_factory=Event)
    _trigger_thread: Thread = PrivateAttr()

    _stop_acquisition_thread: Event = PrivateAttr(default_factory=Event)
    _acquisition_thread: Thread = PrivateAttr()
    _acquisition_cond: Condition = PrivateAttr(default_factory=Condition)

    _temperature_line_pattern: Pattern[bytes] = PrivateAttr(
        default=compile(
            rb"[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?"
        )
    )
    _serial_search_queue: deque[TCSSerialSearchRequest] = PrivateAttr(default_factory=deque)

    @property
    def stimulus_running(self) -> bool:
        return not self._done_trigger.is_set()

    def open(self):
        """
        Open the serial connection and initialize the TCS device.

        This method establishes serial communication, initializes the internal
        buffer, starts the acquisition thread, configures maximum temperature,
        and logs hardware and battery information.

        Raises
        ------
        RuntimeError
            If serial initialization fails or device communication errors occur.
        """
        if self._is_open:
            return

        self._tcs_open_serial()
        self._tcs_set_buffer()
        self._tcs_start_trigger_thread()
        self._tcs_start_acquisition_thread()

        self.execute_command(TCSCommand.SET_MAX_TEMPERATURE, int(self.maximum_temperature * 10))
        self.execute_command(TCSCommand.DISPLAY_TEMPERATURES_DURING_STIMULATION)

        info = self.info()
        match = search(compile(r"Firmware:(.*)\nProbe ID:(.*)\nProbe TYPE:(.*)\n"), info)
        battery_info = self.battery_info()

        LOGGER.info(
            "Hardware info:\n"
            f"\tFirmware: {match.group(1).strip() if match else 'Unknown'}\n"
            f"\tProbe ID: {match.group(2).strip() if match else 'Unknown'}\n"
            f"\tProbe TYPE: {match.group(3).strip() if match else 'Unknown'}\n\n"
            "Battery:\n"
            f"\tVoltage: {battery_info['voltage']}\n"
            f"\tPercentage: {battery_info['percentage']}\n\n"
            "Initialized successfully\n"
        )

        self._is_open = True

    def close(self):
        """Close the serial connection and clean up resources."""
        self._tcs_stop_acquisition_thread()
        self._tcs_stop_trigger_thread()
        self._tcs_close_serial()
        self._tcs_delete_buffer()
        self._is_open = False

    def info(self) -> str:
        """
        Retrieve device information.

        Returns
        -------
        str
            Device information string containing firmware version, probe ID,
            and probe type.

        Raises
        ------
        RuntimeError
            If info request times out or response format is invalid.
        """
        result = self.execute_command(
            TCSCommand.READ_INFO, expected_pattern=compile(rb"(Firmware:\s+.*)\n")
        )

        if result:
            _, match = result
            if not match:
                raise RuntimeError("Info response did not match expected format")

            return match.group(1).decode().replace("\r", "\n")

        raise RuntimeError("Device info request timed out")

    def battery_info(self):
        """
        Retrieve battery status information.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'voltage' (float): Battery voltage in volts
            - 'percentage' (int): Battery charge percentage

        Raises
        ------
        RuntimeError
            If no response from device or response format is invalid.
        """
        result = self.execute_command(
            TCSCommand.READ_BATTERY,
            expected_pattern=compile(rb"(?P<voltage>\d+\.\d+)v\s+(?P<percent>\d+)%"),
        )
        if not result:
            raise RuntimeError("No response from device")

        _, match = result

        if not match:
            raise RuntimeError("Battery response did not match expected format")

        return {
            "voltage": float(match.group("voltage")),
            "percentage": int(match.group("percent")),
        }

    def write(self, command: bytes) -> int | None:
        """
        Write raw bytes to the serial port.

        Parameters
        ----------
        command : bytes
            Bytes to write to the serial port.

        Returns
        -------
        int | None
            Number of bytes written, or None if write fails.

        Raises
        ------
        RuntimeError
            If serial connection is not open.

        Notes
        -----
        This method flushes the serial buffer before writing.
        """
        self._tcs_ensure_open()

        self._serial.flush()
        LOGGER.debug(f"Sending command: {command}")

        bytes_written = self._serial.write(command)
        if bytes_written != len(command):
            LOGGER.warning(f"Partial write: {bytes_written}/{len(command)} bytes")

        return bytes_written

    def execute_command(
        self,
        command: TCSCommand,
        *args,
        expected_pattern: Pattern | None = None,
        timeout: float | None = None,
    ) -> tuple[int, Match[bytes] | None] | None:
        """
        Execute a TCS command and optionally wait for a response.

        Parameters
        ----------
        command : TCSCommand
            Command to execute.
        *args : tuple
            Arguments to format into the command.
        expected_pattern : Pattern | None, default=None
            Regex pattern expected in the response. If provided, blocks until
            pattern is found or timeout occurs.
        timeout : float | None, default=None
            Timeout in seconds for waiting for expected pattern. If None,
            uses response_timeout attribute.

        Returns
        -------
        tuple[int, Match[bytes] | None] | None
            Tuple containing (timestamp, match) if pattern was found and matched,
            None if timeout occurred or no pattern was expected.

        Raises
        ------
        RuntimeError
            If serial connection is not open.
        """
        self._tcs_ensure_open()

        request = TCSSerialSearchRequest(pattern=expected_pattern)
        event = request.event

        if expected_pattern is not None:
            with self._acquisition_cond:
                self._serial_search_queue.append(request)

        self.write(command.format(*args))

        if expected_pattern is not None:
            if not event.wait(timeout=timeout or self.response_timeout):
                with self._acquisition_cond:
                    if request in self._serial_search_queue:
                        self._serial_search_queue.remove(request)
                LOGGER.warning("Device response timed out")
                return None

        return request.result

    def trigger(self, stimulus: TCSStimulus, *, wait: bool = False):
        """
        Trigger a thermal stimulation protocol.

        Parameters
        ----------
        stimulus : TCSStimulus
            Stimulus configuration defining temperature profile and duration.

        Raises
        ------
        RuntimeError
            If serial connection is not open.
        ValueError
            If stimulus is invalid or temperatures exceed maximum_temperature.

        Notes
        -----
        This method starts a timer thread for stimulus duration and optionally
        activates buzzer and trigger output.
        """
        self._tcs_ensure_open()
        self._tcs_validate_stimulus(stimulus)

        self._stimulus = stimulus

        self._done_trigger.clear()
        self._start_trigger.set()

        if wait:
            self.trigger_wait()

    def trigger_wait(self) -> bool:
        self._done_trigger.wait()
        return True

    def calibration(self, timeout: float = 30.0) -> float:
        """
        Perform automatic calibration of the TCS device.

        Parameters
        ----------
        timeout : float, default=30.0
            Maximum time in seconds to wait for calibration to complete.

        Returns
        -------
        float
            Calibration offset value in °C.

        Raises
        ------
        RuntimeError
            If serial connection is not open, calibration fails, or
            response format is invalid.
        """
        self._tcs_ensure_open()

        match = self.execute_command(
            command=TCSCommand.AUTOMATIC_CALIBRATION,
            expected_pattern=compile(rb"N(\d{3})"),
            timeout=timeout,
        )

        if match is None:
            raise RuntimeError("Calibration failed or timed out without response")

        _, match = match
        if not match:
            raise RuntimeError("Calibration response did not match expected format")

        neutral_raw = int(match.group(1))
        return neutral_raw / 10.0

    def reset(self):
        """Reset the TCS device to its initial state."""
        self._tcs_ensure_open()

        self.execute_command(TCSCommand.RESET)
        LOGGER.info("Reset successfully")

    def read_sample(self, timeout: float = 0.01) -> ndarray | None:
        """
        Read the most recent temperature sample.

        Returns
        -------
        ArrayLike
            Structured array containing timestamp and temperature values
            for all 5 channels.

        Raises
        ------
        RuntimeError
            If no samples have been collected yet or serial connection is not open.

        Notes
        -----
        The returned array has dtype with fields: 'timestamp' (uint64) and
        's0' through 's4' (float64) for the 5 temperature channels.
        """
        self._tcs_ensure_open()
        sample = None

        with self._acquisition_cond:
            if self._tcs_buffer_needle == self._tcs_buffer_idx:
                self._acquisition_cond.wait(timeout)

            idx = (self._tcs_buffer_idx - 1) % self.buffer_size
            sample = self._tcs_buffer[idx]
            self._tcs_buffer_needle = self._tcs_buffer_idx

        return sample

    def read_many_sample(self, data: ndarray, n: int = -1, timeout: float = -1) -> int:
        """
        Read multiple temperature samples into a pre-allocated array.

        Parameters
        ----------
        data : ndarray
            Pre-allocated array to store samples. Must have compatible dtype
            and at least `n` rows.
        n : int
            Number of samples to read.
        timeout : float, default=-1
            Maximum time in seconds to wait for samples to become available.

        Returns
        -------
        int
            Number of samples actually read (may be less than n if insufficient
            samples available).

        Raises
        ------
        ValueError
            If `data` array has fewer than `n` rows.
        RuntimeError
            If serial connection is not open.

        Notes
        -----
        This method reads the most recent samples from the circular buffer,
        returning them in chronological order (oldest to newest).
        """
        self._tcs_ensure_open()

        if data.shape[0] < n:
            raise ValueError(f"Provided array has {data.shape[0]} rows, need at least {n}")

        deadline = monotonic_ns() + int(timeout * 1e9) if timeout >= 0 else None

        count = 0
        with self._acquisition_cond:
            if n == -1 and deadline is None:
                pass
            elif n == -1 and deadline is not None:
                remaining = (deadline - monotonic_ns()) / 1e9
                self._acquisition_cond.wait(remaining)
            elif n != -1 and deadline is None:
                while self._tcs_buffer_idx - self._tcs_buffer_needle < n:
                    self._acquisition_cond.wait()
            elif n != -1 and deadline is not None:
                remaining = (deadline - monotonic_ns()) / 1e9
                while self._tcs_buffer_idx - self._tcs_buffer_needle < n and remaining > 0:
                    self._acquisition_cond.wait(remaining)
                    remaining = (deadline - monotonic_ns()) / 1e9

            avail = self._tcs_buffer_idx - self._tcs_buffer_needle
            if avail <= 0:
                return 0

            count = avail if n < 0 else min(avail, n)

            size = self.buffer_size
            buffer = self._tcs_buffer
            needle = self._tcs_buffer_needle

            if count > size:
                needle = self._tcs_buffer_idx - size
                count = size

            start = needle % size
            end = start + count

            if end <= size:
                data[:count] = buffer[start:end]
            else:
                first = size - start
                data[:first] = buffer[start:]
                data[first:count] = buffer[: count - first]

            self._tcs_buffer_needle = needle + count

        return count

    def _tcs_validate_stimulus(self, stimulus: TCSStimulus) -> None:
        if not isinstance(stimulus, TCSStimulus):
            raise ValueError(f"Stimulus must be a TCSStimulus instance, got {type(stimulus)}")

        if stimulus.target > self.maximum_temperature:
            raise ValueError(
                f"Target temperature {stimulus.target} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )

        if stimulus.baseline > self.maximum_temperature:
            raise ValueError(
                f"Baseline temperature {stimulus.baseline} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )

    def _tcs_ensure_open(self):
        if not self._serial.is_open:
            raise RuntimeError("TCS serial connection is not open. Call open() first.")

    def _tcs_open_serial(self):
        try:
            self._serial = Serial(
                port=self.port,
                baudrate=115200,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.read_timeout,
                write_timeout=2,
            )
        except Exception as e:
            raise RuntimeError("Serial initialization failed") from e

    def _tcs_close_serial(self):
        if self._serial.is_open:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._serial.close()

    def _tcs_set_buffer(self):
        try:
            self._tcs_buffer = zeros(
                self.buffer_size,
                dtype=[("timestamp", "uint64"), *((f"s{i}", "float64") for i in range(5))],
            )
            self._tcs_buffer_idx = 0
            self._tcs_buffer_needle = 0
        except Exception as e:
            raise RuntimeError("Buffer initialization failed") from e

    def _tcs_delete_buffer(self):
        del self._tcs_buffer

    def _tcs_trigger(self):
        for command in self._stimulus.build():
            self.write(command)

        precise_sleep(self._stimulus.pre_delay / 1000.0)
        self.execute_command(TCSCommand.TRIGGER_STIMULATION)
        now = monotonic_ns()

        if self.beep:
            self.execute_command(TCSCommand.BUZZER, min(999, self._stimulus.duration // 10), 44)

        self.execute_command(
            TCSCommand.TRIGGER_CHANNEL_DURATION,
            self.trigger_out_channel,
            max(1, min(999, self._stimulus.duration // 10)),
        )

        precise_sleep(
            (
                self._stimulus.duration
                + self._stimulus.post_delay
                - ((monotonic_ns() - now) / 1_000_000)
            )
            / 1000.0
        )

        self._done_trigger.set()

    def _tcs_start_trigger_thread(self):
        self._trigger_thread = Thread(
            target=self._tcs_trigger_thread_func, daemon=True, name="TCS Trigger Thread"
        )
        self._trigger_thread.start()

    def _tcs_stop_trigger_thread(self):
        self._stop_trigger_thread.set()
        self._start_trigger.set()

        if self._trigger_thread.is_alive():
            self._trigger_thread.join()

        del self._trigger_thread

        self._stop_trigger_thread.clear()

    def _tcs_trigger_thread_func(self):
        while not self._stop_trigger_thread.is_set():
            self._start_trigger.wait()
            self._start_trigger.clear()

            if self._stop_trigger_thread.is_set():
                break

            self._tcs_trigger()

    def _tcs_start_acquisition_thread(self):
        self._acquisition_thread = Thread(
            target=self._tcs_acquisition_thread_func, name="TCS Acquisition Thread", daemon=True
        )
        self._acquisition_thread.start()

    def _tcs_stop_acquisition_thread(self):
        self._stop_acquisition_thread.set()

        if self._acquisition_thread.is_alive():
            self._acquisition_thread.join()

        del self._acquisition_thread

        self._stop_acquisition_thread.clear()

    def _tcs_acquisition_thread_func(self):
        try:
            while not self._stop_acquisition_thread.is_set():
                line = self._serial.read_until(b"\n")
                LOGGER.debug(f"Read line: {line}")

                with self._acquisition_cond:
                    if self._serial_search_queue:
                        request = self._serial_search_queue[0]

                        if request.pattern is not None:
                            if match := search(request.pattern, line):
                                request.result = (monotonic_ns(), match)
                                request.event.set()
                                self._serial_search_queue.popleft()

                if match := search(self._temperature_line_pattern, line):
                    idx = self._tcs_buffer_idx % self.buffer_size
                    timestamp = monotonic_ns()
                    values = tuple(map(float, match.groups()))
                    self._tcs_buffer[idx] = (timestamp, *values)

                    with self._acquisition_cond:
                        self._tcs_buffer_idx += 1
                        self._acquisition_cond.notify_all()

        except Exception as e:
            LOGGER.exception(f"Read loop failed: {e}")
            self._stop_acquisition_thread.set()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
