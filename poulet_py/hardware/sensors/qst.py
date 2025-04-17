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
...     readings = tcs.get_readings()
...     print(readings)
"""

try:
    from atexit import register
    from enum import Enum
    from functools import cached_property
    from re import Match, Pattern, compile, search
    from threading import Event, Lock, Thread
    from time import sleep, time_ns

    from pydantic import BaseModel, Field
    from serial import Serial

    from poulet_py.config.logging import LOGGER
except ImportError as e:
    msg = "Missing 'qst' module. To install it use: pip install poulet_py[qst]"
    raise ImportError(msg) from e


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
    # Buzzer ddd: duration in 10× ms, fff: frequency in 10× Hz
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


class TCSStimulus(BaseModel):
    """
    Configuration for thermal stimulation parameters.
    """

    surface: int = Field(
        0,
        description="Target surface (0-5, where 0 means all surfaces)",
        ge=0,
        le=5,
    )
    baseline: float = Field(
        30,
        description="Baseline temperature in °C (20-45)",
        ge=20,
        le=45,
    )
    target: float = Field(
        10,
        description="Target temperature in °C (0-60)",
        ge=0,
        le=60,
    )
    rise_rate: float = Field(
        1,
        description="Temperature rise rate in °C/s (0.1-999.9",
        ge=0.1,
        le=999.9,
    )
    return_speed: float = Field(
        1,
        description="Temperature return speed in °C/s (0.1-999.9)",
        ge=0.1,
        le=999.9,
    )
    duration: int = Field(
        100,
        description="Stimulation duration in ms (10-99999)",
        ge=10,
        le=99999,
    )

    def commands(self) -> tuple[TCSCommand, tuple[int | float]]:
        """
        Generate the sequence of commands needed to configure this stimulus.

        Returns
        -------
        Tuple[bytes, ...]
            Sequence of formatted command strings

        Examples
        --------
        >>> stimulus = TCSStimulus(surface=1)
        >>> stimulus.commands()
        [b'S10000', b'N300', b'C1000', b'V10010', b'D100100', b'R10010']
        """
        surface_map = {0: 11111, 1: 10000, 2: 1000, 3: 100, 4: 10, 5: 1}
        LOGGER.info(
            f"Generating commands for stimulus on surface {self.surface}"
        )
        return [
            TCSCommand.SURFACE_SELECTION.format(surface_map[self.surface]),
            TCSCommand.BASELINE_TEMPERATURE.format(int(self.baseline * 10)),
            TCSCommand.TARGET_TEMPERATURE.format(
                self.surface, int(self.target * 10)
            ),
            TCSCommand.STIMULATION_RATE.format(
                self.surface, int(self.rise_rate * 10)
            ),
            TCSCommand.STIMULATION_DURATION.format(self.surface, self.duration),
            TCSCommand.RETURN_SPEED.format(
                self.surface, int(self.return_speed * 10)
            ),
        ]


class TCS(BaseModel):
    """
    Interface for communicating with a TCS thermal stimulator.
    """

    port: str = Field(
        ...,
        description="Serial port device path (e.g. '/dev/ttyUSB0' or 'COM3')",
        pattern=r"^(COM\d+|/dev/ttyUSB\d+|/dev/tty\.usb\w+)$",
    )
    maximum_temperature: float = Field(
        40,
        description="Safety limit for maximum allowed temperature (0-60°C)",
        ge=0,
        le=60,
    )
    beep: bool = Field(
        False, description="Whether to enable audible beep during stimulation"
    )
    trigger_out_channel: int = Field(
        255,
        description="Output channel of trigger signal (1-255)",
        ge=1,
        le=255,
    )
    read_timeout: float = Field(2, description="Serial read timeout in seconds")
    response_timeout: float = Field(
        2, description="Timeout for command responses in seconds"
    )

    @cached_property
    def serial(self):
        """Initialize and return the serial connection."""
        LOGGER.info(f"Initializing serial connection on port {self.port}")
        return Serial(
            port=self.port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            timeout=self.read_timeout,
        )

    @cached_property
    def lock(self):
        return Lock()

    @cached_property
    def current_search(
        self,
    ) -> tuple[Pattern, Event, tuple[int, str] | None] | None:
        return None

    @current_search.setter
    def current_search(self, value):
        with self.lock:
            LOGGER.debug("Updating current_search pattern")
            self._current_search = value

    @current_search.deleter
    def current_search(self):
        with self.lock:
            LOGGER.debug("Clearing current_search pattern")
            self._current_search = None

    @cached_property
    def thread(self):
        return Thread(
            target=self._read_loop, daemon=True, name="TCS Serial Reader"
        )

    @property
    def stop_event(self) -> bool:
        if not hasattr(self, "_stop_event"):
            self._stop_event = False
        return self._stop_event

    @stop_event.setter
    def stop_event(self, value: bool):
        if not isinstance(value, bool):
            msg = "stop_event must be a boolean value"
            raise ValueError(msg)
        LOGGER.debug(f"Setting stop_event to {value}")
        self._stop_event = value

    @stop_event.deleter
    def stop_event(self):
        LOGGER.info("Resetting stop_event")
        del self._stop_event

    @property
    def stimulus(self) -> TCSStimulus:
        if not hasattr(self, "_stimulus"):
            LOGGER.warning("No stimulus configured, using defaults")
            self._stimulus = TCSStimulus()
        return self._stimulus

    @stimulus.setter
    def stimulus(self, value: TCSStimulus):
        msg = ""
        if not isinstance(value, TCSStimulus):
            msg = "Stimulus must be of type TCSStimulus"
        if self.maximum_temperature < value.target:
            msg = (
                f"Target temperature {value.target} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )
        if value.baseline > self.maximum_temperature:
            msg = (
                f"Baseline temperature {value.baseline} exceeds "
                f"maximum temperature {self.maximum_temperature}"
            )
        if msg:
            raise ValueError(msg)

        LOGGER.info(f"Setting new stimulus configuration: {value}")
        self._stimulus = value

    @stimulus.deleter
    def stimulus(self):
        LOGGER.info("Resetting stimulus configuration")
        del self._stimulus

    def _start_reader(self):
        """Start the background serial reader thread if not already running."""
        if not self.thread.is_alive():
            LOGGER.info("Starting serial reader thread")
            self.stop_event = False
            self.thread.start()
            register(self._stop_reader)

    def _stop_reader(self):
        """Stop the background serial reader thread."""
        if self.thread.is_alive():
            LOGGER.info("Stopping serial reader thread")
            self.stop_event = True
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                LOGGER.warning("Reader thread did not stop gracefully")
            del self.thread

    def _read_loop(self):
        """Continuous reading loop running in background thread"""
        LOGGER.debug("Serial reader thread started")
        while not self.stop_event:
            try:
                if self.serial.in_waiting > 0:
                    data = self.serial.read_until(b"\n").decode()
                    timestamp = time_ns()

                    LOGGER.debug(f"Read data: {data}")

                    if self.current_search:
                        pattern, event, _ = self.current_search
                        if match := search(pattern, data):
                            LOGGER.debug(f"Matched pattern {pattern.pattern}")
                            self.current_search = (
                                pattern,
                                event,
                                (timestamp, match),
                            )
                            event.set()

            except Exception as e:
                LOGGER.error(f"Error in read loop: {e}")
                sleep(0.001)

    def write(self, command: bytes) -> int:
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
            # Start reader thread if not already running
            self._start_reader()
            self.serial.flush()
            LOGGER.debug(f"Sending command: {command}")
            bytes_written = self.serial.write(command)
            if bytes_written != len(command):
                LOGGER.warning(
                    f"Partial write: {bytes_written}/{len(command)} bytes"
                )
            return bytes_written
        except Exception as e:
            LOGGER.error(
                f"Write failed for command {command}: {e}", exc_info=True
            )
            msg = f"Write operation failed: {e}"
            raise RuntimeError(msg) from e

    def _expect_response(
        self, pattern: Pattern
    ) -> tuple[int, Match[str]] | None:
        """
        Wait for a response matching the given pattern.

        Parameters
        ----------
        pattern : Pattern
            The regex pattern to match against incoming data

        Returns
        -------
        tuple[int, Match[str]]] | None
            Tuple of (timestamp, match object) if pattern matched, None otherwise
        """
        event = Event()
        self.current_search = (pattern, event, None)
        LOGGER.debug(f"Waiting for pattern: {pattern.pattern}")

        try:
            if event.wait(timeout=self.response_timeout):
                LOGGER.debug("Pattern matched successfully")
                return self.current_search[2]
            LOGGER.warning(f"Timeout waiting for pattern: {pattern.pattern}")
            return None
        finally:
            self.current_search = None

    def execute_command(
        self,
        command: TCSCommand,
        *args,
        expected_pattern: Pattern | None = None,
    ) -> tuple[int, Match[str]] | None:
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
        >>> tcs.execute_command(
        ...     TCSCommand.READ_INFO, expected_pattern=compile(r"Firmware:(.*)")
        ... )
        """
        LOGGER.info(f"Executing command: {command.name} with args {args}")
        self.write(command.format(*args))

        if expected_pattern:
            return self._expect_response(expected_pattern)
        return None

    def init(self):
        """
        Initialize the TCS connection and verify communication.

        Raises
        ------
        RuntimeError
            If initialization fails
        """
        try:
            LOGGER.info("Initializing TCS connection")
            self._start_reader()
            self.write(
                TCSCommand.SET_MAX_TEMPERATURE.format(
                    int(self.maximum_temperature * 10)
                )
            )

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
            LOGGER.error(f"Initialization failed: {e}", exc_info=True)
            msg = "TCS initialization failed"
            raise RuntimeError(msg) from e

    def close(self):
        """Close the connection and clean up resources."""
        try:
            LOGGER.info("Closing TCS connection")
            self._stop_reader()
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.serial.close()
            del self.serial
            LOGGER.info("Connection closed successfully")
        except Exception as e:
            LOGGER.error(f"Error closing connection: {e}")
            raise e

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
        LOGGER.info("Requesting device info")
        result = self.execute_command(
            TCSCommand.READ_INFO,
            expected_pattern=compile(r"(Firmware:\s+.*)\n"),
        )

        if result:
            _, match = result
            return match.group(1).replace("\r", "\n")

        LOGGER.error("Failed to get device info")
        msg = "Device info request timed out"
        raise RuntimeError(msg)

    def reset(self):
        """Reset the TCS device to its default state."""
        try:
            LOGGER.info("Resetting TCS device")
            self.write(TCSCommand.RESET.format())
            LOGGER.info("Reset command sent successfully")
        except Exception as e:
            LOGGER.error(f"Reset failed: {e!s}", exc_info=True)
            msg = "Reset operation failed"
            raise RuntimeError(msg) from e

    def trigger(self):
        """
        Execute the configured stimulation.

        Raises
        ------
        RuntimeError
            If stimulation fails to trigger
        """
        try:
            LOGGER.info("Starting stimulation with current configuration")
            for command in self.stimulus.commands():
                self.write(command)

            self.write(TCSCommand.TRIGGER_STIMULATION.format())

            if self.beep:
                self.write(
                    TCSCommand.BUZZER.format(
                        min(999, self.stimulus.duration // 10), 44
                    )
                )
            self.write(
                TCSCommand.TRIGGER_CHANNEL_DURATION.format(
                    self.trigger_out_channel,
                    min(999, self.stimulus.duration // 10),
                )
            )
            LOGGER.info("Stimulation triggered successfully")
        except Exception as e:
            LOGGER.error(f"Stimulation failed: {e}")
            msg = "Stimulation failed"
            raise RuntimeError(msg) from e

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
        LOGGER.debug("Requesting temperature readings")
        result = self.execute_command(
            TCSCommand.READ_TEMPERATURES,
            expected_pattern=compile(
                r"(\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})[+\-\s](\d{3})"
            ),
        )

        if result:
            timestamp, match = result
            readings = {
                "timestamp": timestamp,
                "neutral": float(match.group(1)) / 10,
                "s1": float(match.group(2)) / 10,
                "s2": float(match.group(3)) / 10,
                "s3": float(match.group(4)) / 10,
                "s4": float(match.group(5)) / 10,
                "s5": float(match.group(6)) / 10,
            }
            LOGGER.debug(f"Readings obtained: {readings}")
            return readings

        LOGGER.error("Failed to get temperature readings")
        msg = "Temperature readings request timed out"
        raise RuntimeError(msg)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
