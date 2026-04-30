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
    from time import time_ns

    from numpy import ndarray, zeros
    from numpy.typing import ArrayLike
    from pydantic import BaseModel, Field, PrivateAttr
    from serial import Serial

    from poulet_py import LOGGER, TCSCommand, TCSStimulus, precise_sleep
except ImportError as e:
    msg = """
Missing 'qst' module. Install options:
- Dedicated:    pip install poulet_py[qst]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


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
    _buffer_idx : int
        Current index in the circular buffer.
    _buffer : ndarray
        Circular buffer for temperature samples.
    _acquisition_thread : Thread
        Background thread for continuous data acquisition.
    _stop_acquisition_event : Event
        Event to signal acquisition thread termination.
    _sampling_cond : Condition
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
    buffer_size: int = Field(default=1000, description="Size of the internal sampling queue", ge=1)
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

    _buffer_idx: int = PrivateAttr(0)
    _buffer: ndarray = PrivateAttr()
    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _sampling_cond: Condition = PrivateAttr(default_factory=Condition)
    _stimulus_running: bool = PrivateAttr(default=False)

    _temperature_line_pattern: Pattern[bytes] = PrivateAttr(
        default=compile(
            rb"[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?"
        )
    )
    _serial_search_queue: deque[TCSSerialSearchRequest] = PrivateAttr(default_factory=deque)

    @property
    def stimulus_running(self) -> bool:
        """Check if a stimulus is currently running."""
        return self._stimulus_running

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

        self._open_serial()
        self._set_buffer()
        self._start_acquisition_thread()

        self.execute_command(TCSCommand.SET_MAX_TEMPERATURE, int(self.maximum_temperature * 10))

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
        self._stop_acquisition_thread()
        self._close_serial()
        self._delete_buffer()
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
                msg = "Info response did not match expected format"
                raise RuntimeError(msg)

            return match.group(1).decode().replace("\r", "\n")

        msg = "Device info request timed out"
        raise RuntimeError(msg)

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
            msg = "No response from device"
            raise RuntimeError(msg)

        _, match = result

        if not match:
            msg = "Battery response did not match expected format"
            raise RuntimeError(msg)

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
        self._ensure_open()

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
        self._ensure_open()

        request = TCSSerialSearchRequest(pattern=expected_pattern)
        event = request.event

        if expected_pattern is not None:
            with self._sampling_cond:
                self._serial_search_queue.append(request)

        self.write(command.format(*args))

        if expected_pattern is not None:
            if not event.wait(timeout=timeout or self.response_timeout):
                with self._sampling_cond:
                    if request in self._serial_search_queue:
                        self._serial_search_queue.remove(request)
                LOGGER.warning("Device response timed out")
                return None

        return request.result

    def trigger(self, stimulus: TCSStimulus):
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
        self._ensure_open()

        self._validate_stimulus(stimulus)

        for command in stimulus.build():
            self.write(command)

        self.execute_command(TCSCommand.TRIGGER_STIMULATION)

        Thread(
            target=self._stimulus_timer, args=(stimulus.duration,), name="TCS-stimulus-timer"
        ).start()

        if self.beep:
            self.execute_command(TCSCommand.BUZZER, min(999, stimulus.duration // 10), 44)

        self.execute_command(
            TCSCommand.TRIGGER_CHANNEL_DURATION,
            self.trigger_out_channel,
            max(1, min(999, stimulus.duration // 10)),
        )

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
        self._ensure_open()

        match = self.execute_command(
            command=TCSCommand.AUTOMATIC_CALIBRATION,
            expected_pattern=compile(rb"N(\d{3})"),
            timeout=timeout,
        )

        if match is None:
            msg = "Calibration failed or timed out without response"
            raise RuntimeError(msg)

        _, match = match
        if not match:
            msg = "Calibration response did not match expected format"
            raise RuntimeError(msg)

        neutral_raw = int(match.group(1))
        return neutral_raw / 10.0

    def reset(self):
        """Reset the TCS device to its initial state."""
        self._ensure_open()

        self.execute_command(TCSCommand.RESET)
        LOGGER.info("Reset successfully")

    def read_last_sample(self) -> ArrayLike:
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
        self._ensure_open()

        with self._sampling_cond:
            if self._buffer_idx == 0:
                msg = "No samples collected yet"
                raise RuntimeError(msg)

            idx = (self._buffer_idx - 1) % self.buffer_size
            return self._buffer[idx]

    def read_many_sample(self, data: ndarray, n: int, timeout: float = 10.0) -> int:
        """
        Read multiple temperature samples into a pre-allocated array.

        Parameters
        ----------
        data : ndarray
            Pre-allocated array to store samples. Must have compatible dtype
            and at least `n` rows.
        n : int
            Number of samples to read.
        timeout : float, default=10.0
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
        self._ensure_open()

        if data.shape[0] < n:
            msg = f"Provided array has {data.shape[0]} rows, need at least {n}"
            raise ValueError(msg)

        deadline = time_ns() + timeout

        with self._sampling_cond:
            while self._buffer_idx == 0:
                remaining = deadline - time_ns()
                if remaining <= 0:
                    return 0
                self._sampling_cond.wait(timeout=remaining / 1e9)

            total_samples = self._buffer_idx
            available = min(total_samples, self.buffer_size)
            count = min(n, available)

            start_idx = (total_samples - count) % self.buffer_size
            first_chunk = min(self.buffer_size - start_idx, count)
            second_chunk = count - first_chunk

            data[0:first_chunk] = self._buffer[start_idx : start_idx + first_chunk]

            if second_chunk > 0:
                data[first_chunk:count] = self._buffer[0:second_chunk]

            return count

    def _validate_stimulus(self, stimulus: TCSStimulus) -> None:
        """
        Validate stimulus parameters.

        Parameters
        ----------
        stimulus : TCSStimulus
            Stimulus to validate.

        Raises
        ------
        ValueError
            If stimulus is not a TCSStimulus instance or if temperatures
            exceed maximum_temperature.
        """
        if not isinstance(stimulus, TCSStimulus):
            msg = f"Stimulus must be a TCSStimulus instance, got {type(stimulus)}"
            raise ValueError(msg)

        if stimulus.target > self.maximum_temperature:
            msg = (
                f"Target temperature {stimulus.target} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )
            raise ValueError(msg)

        if stimulus.baseline > self.maximum_temperature:
            msg = (
                f"Baseline temperature {stimulus.baseline} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )
            raise ValueError(msg)

    def _stimulus_timer(self, duration_ms: int):
        """
        Timer function for stimulus duration.

        Parameters
        ----------
        duration_ms : int
            Stimulus duration in milliseconds.
        """
        try:
            self._stimulus_running = True
            precise_sleep(duration_ms / 1000.0)
        finally:
            self._stimulus_running = False

    def _ensure_open(self):
        """
        Check if serial connection is open.

        Raises
        ------
        RuntimeError
            If serial connection is not open.
        """
        if not self._serial.is_open:
            msg = "TCS serial connection is not open. Call open() first."
            raise RuntimeError(msg)

    def _open_serial(self):
        """
        Open serial connection with TCS-specific parameters.

        Raises
        ------
        RuntimeError
            If serial initialization fails.
        """
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
            self.close()
            msg = "Serial initialization failed"
            raise RuntimeError(msg) from e

    def _close_serial(self):
        """
        Close serial connection safely.

        Raises
        ------
        RuntimeError
            If error occurs while closing serial connection.
        """

        if self._serial.is_open:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._serial.close()

    def _set_buffer(self):
        """Initialize the circular buffer for temperature samples."""
        try:
            self._buffer = zeros(
                self.buffer_size,
                dtype=[("timestamp", "uint64"), *((f"s{i}", "float64") for i in range(5))],
            )
            self._buffer_idx = 0
        except Exception as e:
            self.close()
            msg = "Buffer initialization failed"
            raise RuntimeError(msg) from e

    def _delete_buffer(self):
        """Delete the circular buffer and reset index."""
        del self._buffer
        self._buffer_idx = 0

    def _start_acquisition_thread(self):
        """Start the background acquisition thread."""
        try:
            self.execute_command(TCSCommand.DISPLAY_TEMPERATURES_DURING_STIMULATION)

            self._acquisition_thread = Thread(
                target=self._acquisition_thread_func, name="TCS Acquisition Thread", daemon=True
            )
            self._acquisition_thread.start()
        except Exception as e:
            self.close()
            msg = "Acquisition thread failed to start"
            raise RuntimeError(msg) from e

    def _stop_acquisition_thread(self):
        """Stop the background acquisition thread."""
        self._stop_acquisition_event.set()
        self._acquisition_thread.join(timeout=5)

        if self._acquisition_thread.is_alive():
            LOGGER.warning("Streaming thread did not stop gracefully")

        del self._acquisition_thread

        self._stop_acquisition_event.clear()

    def _acquisition_thread_func(self):
        """
        Background thread function for continuous data acquisition.

        This function continuously reads temperature data from the serial port,
        parses temperature values, stores them in the circular buffer, and
        processes pattern search requests.
        """
        try:
            while not self._stop_acquisition_event.is_set():
                line = self._serial.read_until(b"\n")
                LOGGER.debug(f"Read line: {line}")

                with self._sampling_cond:
                    if self._serial_search_queue:
                        request = self._serial_search_queue[0]

                        if request.pattern is not None:
                            if match := search(request.pattern, line):
                                request.result = (time_ns(), match)
                                request.event.set()
                                self._serial_search_queue.popleft()

                if match := search(self._temperature_line_pattern, line):
                    idx = self._buffer_idx % self.buffer_size
                    timestamp = time_ns()
                    values = tuple(map(float, match.groups()))
                    self._buffer[idx] = (timestamp, *values)

                    with self._sampling_cond:
                        self._buffer_idx += 1
                        self._sampling_cond.notify_all()

        except Exception as e:
            msg = f"Read loop failed: {e}"
            LOGGER.exception(msg)
            self._stop_acquisition_event.set()

    def __enter__(self):
        """Context manager entry: open the connection."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close the connection."""
        self.close()
