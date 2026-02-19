"""
Thermal Control System (TCS) interface module.

This module provides a Python interface for communicating with a TCS thermal stimulator
via serial connection. It includes command definitions, stimulus configuration,
and data reading capabilities.

Examples
--------
>>> with TCS(port="/dev/ttyUSB0") as tcs:
...     tcs.init()
...     stimulus = TCSStimulus(surface=1, target=35.0)
...     tcs.stimulus = stimulus
...     tcs.trigger()
...     readings = tcs.read_sample()
...     print(readings)
"""

try:
    from collections import deque
    from enum import Enum
    from re import Match, Pattern, compile, search
    from threading import Event, Lock, Thread
    from time import monotonic, perf_counter_ns, sleep

    from deprecated import deprecated
    from numpy import dtype, empty, ndarray
    from numpy.typing import ArrayLike
    from pydantic import BaseModel, Field, PrivateAttr
    from serial import Serial

    from poulet_py import LOGGER, BaseTrigger, TCSStimulus
except ImportError as e:
    msg = """
Missing 'qst' module. Install options:
- Dedicated:    pip install poulet_py[qst]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TCSSerialSearchRequest(BaseModel):
    pattern: Pattern[str] | None = Field(
        default=None, description="Regex pattern to search for in incoming serial data"
    )
    event: Event = Field(default_factory=Event, description="Event to signal when pattern is found")
    result: tuple[int, Match[str] | None] | None = Field(
        default=None, description="The matched result once the pattern is found"
    )


class TCSCommand(bytes, Enum):
    """
    Enumeration of all available TCS commands with their byte representations.

    Each command includes formatting capability for parameterized commands.

    Examples
    --------
    >>> TCSCommand.READ_TEMPERATURES
    <TCSCommand.READ_TEMPERATURES: b'E'>
    >>> TCSCommand.BASELINE_TEMPERATURE.format(300)
    b'N300'
    """

    READ_INFO = b"H"
    # Neutral temperature then each surface
    READ_TEMPERATURES = b"E"
    # Display the current values of the stimulator parameters
    READ_STIMULATION_VALUES = b"P"
    # Return the status of buttons 1 and 2.
    # 10 button 1 pressed; 01 button 2 pressed; 11 both pressed
    READ_BUTTON_STATUS = b"K"
    # Display voltage and % battery charge
    READ_BATTERY = b"B"
    # Return error codes for probe diagnosis
    # Returns “xxxxxx” for each zone and the basic temperature;
    # x = 0 : OK / x > 1 : ERROR
    READ_ERRORS = b"Q"

    # Allow regular display of current temperatures, 1Hz
    DISPLAY_TEMPERATURES_BETWEEN_STIMULATION = b"Oa"
    # Allow the display of temperatures during stimulation, 100 Hz
    DISPLAY_TEMPERATURES_DURING_STIMULATION = b"Ob"
    # Reset the TCS (same action as switching OFF and ON again)
    RESET = b"Oc"

    # Define a maximum stimulation temperature, xxx' 1/10 °C
    SET_MAX_TEMPERATURE = b"Om%03d"

    # Automatic calibration of the reference temperature,
    # Displays Nxxx with neutral t° in case of success
    AUTOMATIC_CALIBRATION = b"G"
    # Deactivate the display of current temperatures
    DEACTIVATE_DISPLAY = b"F"
    # Trigger stimulation with the current settings
    TRIGGER_STIMULATION = b"L"
    # Force a halt to the current stimulation A
    HALT_STIMULATION = b"A"

    # xxx=200-450, unit=0.1°C, default: 300
    BASELINE_TEMPERATURE = b"N%03d"
    # xxxxx=0 or 1 per surface , default: 00000
    SURFACE_SELECTION = b"S%05d"

    # s=0-5 (surface number), xxx=000-600, unit=0.1°C, default: 100
    TARGET_TEMPERATURE = b"C%d%03d"
    # s=0-5 (surface number), xxxx=0001-9999, unit=0.1°C/s,
    # default: Depends on the type of sensor
    STIMULATION_RATE = b"V%d%04d"
    # s=0-5 (surface number). xxxx=0001-9999, unit=0.1°C/s,
    # default: Depends on the type of sensor
    RETURN_SPEED = b"R%d%04d"
    # s=0-5 (surface number). xxxxx=00010-99999, unit=ms, default: 00100
    STIMULATION_DURATION = b"D%d%05d"
    # xxx=001-255 (trigger_channel), yyy=010-999 (duration), unit=ms, default: 255300
    TRIGGER_CHANNEL_DURATION = b"T%03d%03d"
    # Buzzer ddd: duration in 10X ms, fff: frequency in 10× Hz
    BUZZER = b"Z%03d%03d"

    def format(self, *args: int | float) -> bytes:
        """
        Format the command with the given arguments.

        Parameters
        ----------
        *args : int, float
            Arguments to format into the command string

        Returns
        -------
        bytes
            Formatted command string

        Raises
        ------
        ValueError
            If arguments don't match the command's format requirements

        Examples
        --------
        >>> TCSCommand.TARGET_TEMPERATURE.format(1, 350)
        b'C1350'
        """
        try:
            LOGGER.debug(f"Formatting command {self.name} with args {args}")
            return self.value % args
        except TypeError as e:
            msg = f"Wrong number/type of arguments for {self.name}: {e}"
            raise ValueError(msg) from e


class TCS(BaseModel, validate_assignment=True):
    """
    Interface for communicating with a TCS thermal stimulator.

    Parameters
    ----------
    port : str
        Serial port to which the device is connected.
    maximum_temperature : float, optional
        Maximum allowed temperature in °C (default: 40).
    beep : bool, optional
        Whether to enable audible beeps (default: False).
    trigger_out_channel : int, optional
        Output channel for trigger signals (default: 255).
    read_timeout : float, optional
        Timeout for read operations in seconds (default: 2).
    response_timeout : float, optional
        Timeout for device responses in seconds (default: 2).
    stimulus_trigger : BaseTrigger, optional
        A Trigger found in poulet_py/hardware/triggers to trigger the next stimulus.

    Methods
    -------
    init() -> None
        Initialize the TCS connection and verify communication.
    close() -> None
        Close the connection and clean up resources.
    info() -> str
        Get device information including firmware version and probe details.
    reset() -> None
        Reset the TCS device to its default state.
    trigger() -> None
        Execute the configured stimulation.
    read_sample() -> dict
        Get current temperature readings from all sensors.

    Examples
    --------
    >>> with TCS(port="/dev/ttyUSB0") as tcs:
    >>>     tcs.init()
    >>>     stimulus = TCSStimulus(surface=1, target=35.0)
    >>>     tcs.stimulus = stimulus
    >>>     tcs.trigger()
    >>>     readings = tcs.read_sample()
    >>>     print(readings)
    """

    port: str = Field(
        ...,
        description="Serial port to which the TCS device is connected",
        pattern=r"^(COM\d+|/dev/ttyUSB\d+|/dev/tty\.usb\w+)$",
    )
    buffer_size: int = Field(
        default=1000,
        description="Size of the internal sampling queue",
        ge=1,
        le=10000,
    )
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
    stimulus_trigger: BaseTrigger | None = Field(
        default=None,
        description="A Trigger found in poulet_py/hardware/triggers to trigger the next stimulus",
    )

    _serial: Serial = PrivateAttr()

    _samples_buffer: ndarray = PrivateAttr()
    _sampling_thread: Thread | None = PrivateAttr(None)
    _sampling_stop_event: Event = PrivateAttr(default_factory=Event)
    _sampling_lock: Lock = PrivateAttr(default_factory=Lock)
    _sampling_event: Event = PrivateAttr(default_factory=Event)
    _stimulus_running: Event = PrivateAttr(default_factory=Event)

    _temperature_line_pattern: Pattern = PrivateAttr(
        default=compile(
            r"(\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})"
        )
    )
    _serial_search_queue: deque[TCSSerialSearchRequest] = PrivateAttr(default_factory=deque)

    @property
    def is_stimulus_running(self) -> bool:
        """Return True while a stimulus is active."""
        return self._stimulus_running.is_set()

    def _validate_stimulus(self, stimulus: TCSStimulus) -> None:
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
        """Internal timer that clears running flag after duration."""
        try:
            sleep(duration_ms / 1000.0)
        finally:
            self._stimulus_running.clear()

    def _start_streaming(self):
        """Start the background serial reader thread if not already running."""
        if self._sampling_thread is None or not self._sampling_thread.is_alive():
            self._samples_buffer = empty(
                (self.buffer_size, 7),
                dtype=dtype(
                    [("timestamp", "uint64"), ("neutral", "float32")]
                    + [(f"s{i}", "float32") for i in range(1, 6)]
                ),
            )
            self._sampling_idx = 0
            self._sampling_stop_event.clear()

            self.execute_command(TCSCommand.DISPLAY_TEMPERATURES_DURING_STIMULATION)

            self._sampling_thread = Thread(
                target=self._streaming_loop, name="TCS Temperature Streamer"
            )
            self._sampling_thread.start()
            LOGGER.info("Started temperature streaming")

    def _stop_streaming(self):
        """Stop temperature streaming thread"""
        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_stop_event.set()
            self._sampling_thread.join(timeout=5)
            if self._sampling_thread.is_alive():
                LOGGER.warning("Streaming thread did not stop gracefully")
            self._sampling_thread = None
        self._sampling_stop_event.clear()
        self._sampling_idx = 0
        del self._samples_buffer

    def _streaming_loop(self):
        """Continuous reading loop running in background thread"""
        LOGGER.debug("Serial reader thread started")
        try:
            while not self._sampling_stop_event.is_set():
                if self._serial.in_waiting > 0:
                    line = self._serial.read_until(b"\n").decode().strip()
                    LOGGER.debug(f"Read line: {line}")

                    if self._serial_search_queue:
                        request = self._serial_search_queue[0]
                        if request.pattern is not None:
                            if match := search(request.pattern, line):
                                request.result = (perf_counter_ns(), match)
                                request.event.set()
                                self._serial_search_queue.popleft()
                                continue

                    if match := search(self._temperature_line_pattern, line):
                        idx = self._sampling_idx % self.buffer_size
                        self._samples_buffer[idx] = (
                            perf_counter_ns(),
                            float(match.group(1)) / 10.0,
                            float(match.group(2)) / 10.0,
                            float(match.group(3)) / 10.0,
                            float(match.group(4)) / 10.0,
                            float(match.group(5)) / 10.0,
                            float(match.group(6)) / 10.0,
                        )

                    with self._sampling_lock:
                        self._sampling_idx += 1
                        self._sampling_event.set()

        except Exception as e:
            msg = f"Read loop failed: {e}"
            LOGGER.exception(msg)
            self._sampling_stop_event.set()

    @deprecated(
        reason="init() will be removed in future releases use open() method instead",
        version="0.1.4",
    )
    def init(self):
        self.open()

    def open(self):
        """
        Initialize the TCS connection and verify communication.

        Raises
        ------
        RuntimeError
            If initialization fails
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

            self._start_streaming()
            self.execute_command(TCSCommand.SET_MAX_TEMPERATURE, int(self.maximum_temperature * 10))

            info = self.info()
            match = search(
                compile(r"Firmware:(.*)\nProbe ID:(.*)\nProbe TYPE:(.*)\n"),
                info,
            )

            LOGGER.info(
                "Initialized successfully\n"
                f"Firmware: {match.group(1).strip() if match else 'Unknown'}\n"
                f"Probe ID: {match.group(2).strip() if match else 'Unknown'}\n"
                f"Probe TYPE: {match.group(3).strip() if match else 'Unknown'}"
            )
        except Exception as e:
            msg = "TCS initialization failed"
            raise RuntimeError(msg) from e

    def close(self):
        """Close the connection and clean up resources."""
        try:
            self._stop_streaming()
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._serial.close()
        except Exception as e:
            msg = "Error closing TCS connection"
            raise RuntimeError(msg) from e

    def info(self) -> str:
        """
        Get device information including firmware version and probe details.

        Returns
        -------
        str
            Device information string

        Raises
        ------
        RuntimeError
            If the info command fails or times out
        """
        result = self.execute_command(
            TCSCommand.READ_INFO,
            expected_pattern=compile(r"(Firmware:\s+.*)\n"),
        )

        if result:
            _, match = result
            if not match:
                msg = "Info response did not match expected format"
                raise RuntimeError(msg)

            return match.group(1).replace("\r", "\n")

        msg = "Device info request timed out"
        raise RuntimeError(msg)

    def write(self, command: bytes) -> int | None:
        """
        Write a command to the TCS device.

        Parameters
        ----------
        command : bytes
            The command to send

        Returns
        -------
        int
            Number of bytes written

        Raises
        ------
        RuntimeError
            If the write operation fails
        """
        try:
            if not self._serial.is_open:
                msg = "Serial port is not open"
                raise RuntimeError(msg)

            self._serial.flush()
            LOGGER.debug(f"Sending command: {command}")
            bytes_written = self._serial.write(command)
            if bytes_written != len(command):
                LOGGER.warning(f"Partial write: {bytes_written}/{len(command)} bytes")
            return bytes_written
        except Exception as e:
            msg = f"Write operation failed: {e}"
            raise RuntimeError(msg) from e

    def execute_command(
        self,
        command: TCSCommand,
        *args,
        expected_pattern: Pattern | None = None,
        timeout: float | None = None,
    ) -> tuple[int, Match[str] | None] | None:
        """
        Execute a command and optionally wait for a response.

        Parameters
        ----------
        command : TCSCommand
            The command to execute
        *args
            Arguments to format into the command
        expected_pattern : Pattern | None
            Regex pattern to match against the response

        Returns
        -------
        tuple[int, Match[str]]] | None
            If expected_pattern provided, returns (timestamp, match) tuple

        Examples
        --------
        >>> tcs.execute_command(TCSCommand.READ_INFO, expected_pattern=compile(r"Firmware:(.*)"))
        """
        event = Event()
        request = TCSSerialSearchRequest(pattern=expected_pattern, event=event)

        if expected_pattern is not None:
            with self._sampling_lock:
                self._serial_search_queue.append(request)

        self.write(command.format(*args))

        if expected_pattern is not None:
            if not event.wait(timeout=timeout or self.response_timeout):
                with self._sampling_lock:
                    if request in self._serial_search_queue:
                        self._serial_search_queue.remove(request)
                LOGGER.warning("Device response timed out")
                return None

        return request.result

    def trigger(self, stimulus: TCSStimulus):
        """
        Execute the configured stimulation.

        Raises
        ------
        RuntimeError
            If stimulation fails to trigger
        """
        try:
            self._validate_stimulus(stimulus)

            for command in stimulus.build():
                self.write(command)

            if self.stimulus_trigger is not None:
                if not self.stimulus_trigger.wait():
                    msg = "Trigger Failed, canceling stimulation"
                    raise RuntimeError(msg)

            self.execute_command(TCSCommand.TRIGGER_STIMULATION)

            self._stimulus_running.set()
            self._stimulus_thread = Thread(
                target=self._stimulus_timer, args=(stimulus.duration,), name="TCS Stimulus Timer"
            )
            self._stimulus_thread.start()

            if self.beep:
                self.execute_command(TCSCommand.BUZZER, min(999, stimulus.duration // 10), 44)

            self.execute_command(
                TCSCommand.TRIGGER_CHANNEL_DURATION,
                self.trigger_out_channel,
                max(1, min(999, stimulus.duration // 10)),
            )

        except Exception as e:
            self._stimulus_running.clear()
            msg = "Stimulation failed"
            raise RuntimeError(msg) from e

    def calibration(self, timeout: float = 30.0) -> float:
        """
        Perform automatic calibration and wait until it completes.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for calibration to complete.

        Returns
        -------
        float
            Calibrated neutral temperature in °C.
        """
        match = self.execute_command(
            command=TCSCommand.AUTOMATIC_CALIBRATION,
            expected_pattern=compile(r"N(\d{3})"),
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
        """Reset the TCS device to its default state."""
        try:
            self.execute_command(TCSCommand.RESET)
            LOGGER.info("Reset successfully")
        except Exception as e:
            msg = "Reset operation failed"
            raise RuntimeError(msg) from e

    @deprecated(
        reason="get_readings() is deprecated, use read_sample() instead",
        version="0.1.4",
    )
    def get_readings(self) -> dict[str, float]:
        """
        Get current temperature readings from all sensors.

        Returns
        -------
        Dict[str, float]
            Dictionary containing temperatures for neutral and all surfaces,
            plus a timestamp key.

        Raises
        ------
        RuntimeError
            If reading temperatures fails
        """
        result = self.execute_command(
            TCSCommand.READ_TEMPERATURES,
            expected_pattern=compile(
                r"(\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})"
            ),
        )

        if result:
            timestamp, match = result
            if not match:
                msg = "Temperature response did not match expected format"
                raise RuntimeError(msg)

            readings = {
                "timestamp": timestamp,
                "neutral": float(match.group(1)) / 10,
                "s1": float(match.group(2)) / 10,
                "s2": float(match.group(3)) / 10,
                "s3": float(match.group(4)) / 10,
                "s4": float(match.group(5)) / 10,
                "s5": float(match.group(6)) / 10,
            }
            return readings

        msg = "Temperature readings request timed out"
        raise RuntimeError(msg)

    def read_last_sample(self) -> ArrayLike:
        """Return the most recent sample"""
        with self._sampling_lock:
            if self._sampling_idx == 0:
                msg = "No samples collected yet"
                raise RuntimeError(msg)

            idx = (self._sampling_idx - 1) % self.buffer_size
            return self._samples_buffer[idx]

    def read_many_sample(self, data: ndarray, n: int, timeout: float = 10.0) -> int:
        """
        Copy up to `n` most recent samples into the provided array `data`.

        Parameters
        ----------
        data : ndarray
            Preallocated array with at least `n` rows, same dtype as `_samples_buffer`
        n : int
            Number of samples to retrieve
        timeout : float
            Max time in seconds to wait for samples

        Returns
        -------
        int
            Number of samples actually copied
        """
        if data.shape[0] < n:
            msg = f"Provided array has {data.shape[0]} rows, need at least {n}"
            raise ValueError(msg)

        deadline = monotonic() + timeout
        copied = 0

        while copied < n:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break

            if not self._sampling_event.wait(timeout=remaining):
                break

            with self._sampling_lock:
                total_samples = self._sampling_idx
                available = min(total_samples, self.buffer_size)

                if available == 0:
                    self._sampling_event.clear()
                    continue

                count = min(n, available)
                start_idx = (total_samples - count) % self.buffer_size
                first_chunk = min(self.buffer_size - start_idx, count)
                second_chunk = count - first_chunk

                data[0:first_chunk] = self._samples_buffer[start_idx : start_idx + first_chunk]

                if second_chunk > 0:
                    data[first_chunk:count] = self._samples_buffer[0:second_chunk]

                copied = count

                self._sampling_event.clear()

        return copied

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
