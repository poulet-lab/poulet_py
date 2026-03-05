try:
    from collections import deque
    from re import Match, Pattern, compile, search
    from threading import Condition, Event, Thread
    from time import monotonic_ns, sleep

    from numpy import empty, ndarray
    from numpy.typing import ArrayLike
    from pydantic import BaseModel, Field, PrivateAttr
    from serial import Serial

    from poulet_py import LOGGER, BaseTrigger, TCSCommand, TCSStimulus


except ImportError as e:
    msg = """
Missing 'qst' module. Install options:
- Dedicated:    pip install poulet_py[qst]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class TCSSerialSearchRequest(BaseModel):
    pattern: Pattern[bytes] | None = Field(
        default=None, description="Regex pattern to search for in incoming serial data"
    )
    _result: tuple[int, Match[bytes] | None] | None = PrivateAttr(default=None)
    _event: Event = PrivateAttr(default_factory=Event)

    @property
    def event(self) -> Event:
        return self._event

    @event.setter
    def event(self, value: Event) -> None:
        self._event = value

    @property
    def result(self) -> tuple[int, Match[bytes] | None] | None:
        return self._result

    @result.setter
    def result(self, value: tuple[int, Match[bytes] | None] | None) -> None:
        self._result = value


class TCS(BaseModel, validate_assignment=True):
    port: str = Field(
        ...,
        description="Serial port to which the TCS device is connected",
        pattern=r"^(COM\d+|(/dev/)tty(USB\d+|\.usb[a-zA-Z0-9]+))$",
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
    _is_open: bool = PrivateAttr(default=False)

    _sampling_idx: int = PrivateAttr(0)
    _samples_buffer: ndarray | None = PrivateAttr(None)
    _sampling_thread: Thread | None = PrivateAttr(None)
    _sampling_stop_event: Event = PrivateAttr(default_factory=Event)
    _sampling_cond: Condition = PrivateAttr(default_factory=Condition)

    _stimulus_thread: Thread | None = PrivateAttr(None)
    _stimulus_done: Event = PrivateAttr(default_factory=Event)

    _temperature_line_pattern: Pattern[bytes] = PrivateAttr(
        default=compile(
            rb"[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?(\d+\.\d+)[+\-\s,]+?"
        )
    )
    _serial_search_queue: deque[TCSSerialSearchRequest] = PrivateAttr(default_factory=deque)

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
        try:
            sleep(duration_ms / 1000.0)
        finally:
            self._stimulus_done.set()

    def _start_streaming(self):
        if self._samples_buffer is None:
            self._samples_buffer = empty(
                self.buffer_size,
                dtype=[
                    ("timestamp", "uint64"),
                    ("temperature", "float32", (5,)),
                ],
            )
            self._sampling_idx = 0

        if self._sampling_thread is None or not self._sampling_thread.is_alive():
            self.execute_command(TCSCommand.DISPLAY_TEMPERATURES_DURING_STIMULATION)

            self._sampling_thread = Thread(
                target=self._streaming_loop, name="TCS Temperature Streamer"
            )
            self._sampling_thread.start()

    def _stop_streaming(self):
        if self._sampling_thread and self._sampling_thread.is_alive():
            self._sampling_stop_event.set()
            self._sampling_thread.join(timeout=5)
            if self._sampling_thread.is_alive():
                LOGGER.warning("Streaming thread did not stop gracefully")
            self._sampling_thread = None
        self._sampling_stop_event.clear()

    def _streaming_loop(self):
        try:
            if self._samples_buffer is None:
                msg = "Samples buffer is not initialized"
                raise RuntimeError(msg)

            while not self._sampling_stop_event.is_set():
                line = self._serial.read_until(b"\n")
                LOGGER.debug(f"Read line: {line}")

                with self._sampling_cond:
                    if self._serial_search_queue:
                        request = self._serial_search_queue[0]
                        if request.pattern is not None:
                            if match := search(request.pattern, line):
                                request.result = (monotonic_ns(), match)
                                request.event.set()
                                self._serial_search_queue.popleft()

                if match := search(self._temperature_line_pattern, line):
                    idx = self._sampling_idx % self.buffer_size
                    timestamp = monotonic_ns()
                    values = tuple(map(float, match.groups()))
                    self._samples_buffer[idx] = (timestamp, values)

                    with self._sampling_cond:
                        self._sampling_idx += 1
                        self._sampling_cond.notify_all()

        except Exception as e:
            msg = f"Read loop failed: {e}"
            LOGGER.exception(msg)
            self._sampling_stop_event.set()

    def open(self):
        try:
            if self._is_open:
                self.close()
                msg = "Device already open"
                raise RuntimeError(msg)

            self._serial = Serial(
                port=self.port,
                baudrate=115200,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self.read_timeout,
                write_timeout=2,
            )
            self._is_open = True

            self._start_streaming()
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
        except Exception as e:
            self.close()
            msg = "TCS initialization failed"
            raise RuntimeError(msg) from e

    def close(self):
        try:
            self._stop_streaming()
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._serial.close()
            self._is_open = False
        except Exception as e:
            msg = "Error closing TCS connection"
            raise RuntimeError(msg) from e

    def info(self) -> str:
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
        if not self._is_open:
            msg = "use open() first"
            raise RuntimeError(msg)

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
        if not self._is_open:
            msg = "use open() first"
            raise RuntimeError(msg)

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
        if not self._is_open:
            msg = "use open() first"
            raise RuntimeError(msg)

        self._validate_stimulus(stimulus)

        for command in stimulus.build():
            self.write(command)

        if self.stimulus_trigger is not None:
            if not self.stimulus_trigger.wait():
                msg = "Trigger Failed, canceling stimulation"
                raise RuntimeError(msg)

        self.execute_command(TCSCommand.TRIGGER_STIMULATION)

        self._stimulus_done.clear()
        Thread(
            target=self._stimulus_timer, args=(stimulus.duration,), name="TCS Stimulus Timer"
        ).start()

        if self.beep:
            self.execute_command(TCSCommand.BUZZER, min(999, stimulus.duration // 10), 44)

        self.execute_command(
            TCSCommand.TRIGGER_CHANNEL_DURATION,
            self.trigger_out_channel,
            max(1, min(999, stimulus.duration // 10)),
        )

    def calibration(self, timeout: float = 30.0) -> float:
        if not self._is_open:
            msg = "use open() first"
            raise RuntimeError(msg)

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
        if not self._is_open:
            msg = "use open() first"
            raise RuntimeError(msg)

        self.execute_command(TCSCommand.RESET)
        LOGGER.info("Reset successfully")

    def read_last_sample(self) -> ArrayLike:
        if not self._is_open or self._samples_buffer is None:
            msg = "use open() first"
            raise RuntimeError(msg)

        with self._sampling_cond:
            if self._sampling_idx == 0:
                msg = "No samples collected yet"
                raise RuntimeError(msg)

            idx = (self._sampling_idx - 1) % self.buffer_size
            return self._samples_buffer[idx]

    def read_many_sample(self, data: ndarray, n: int, timeout: float = 10.0) -> int:
        if not self._is_open or self._samples_buffer is None:
            msg = "use open() first"
            raise RuntimeError(msg)

        if data.shape[0] < n:
            msg = f"Provided array has {data.shape[0]} rows, need at least {n}"
            raise ValueError(msg)

        deadline = monotonic_ns() + timeout

        with self._sampling_cond:
            while self._sampling_idx == 0:
                remaining = deadline - monotonic_ns()
                if remaining <= 0:
                    return 0
                self._sampling_cond.wait(timeout=remaining)

            total_samples = self._sampling_idx
            available = min(total_samples, self.buffer_size)
            count = min(n, available)

            start_idx = (total_samples - count) % self.buffer_size
            first_chunk = min(self.buffer_size - start_idx, count)
            second_chunk = count - first_chunk

            data[0:first_chunk] = self._samples_buffer[start_idx : start_idx + first_chunk]

            if second_chunk > 0:
                data[first_chunk:count] = self._samples_buffer[0:second_chunk]

            return count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
