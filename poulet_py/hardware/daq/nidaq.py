"""
NI-DAQmx Task Management System.

This module provides a structured, object-oriented interface for creating and managing
NI-DAQmx tasks with proper synchronization and resource management.

Classes:
    NIClockHandle: Represents a clock configuration for task synchronization.
    NIBaseChannel: Abstract base class for all channel types.
    NIAnalogInputChannel: Configuration for analog input channels.
    NIAnalogOutputChannel: Configuration for analog output channels.
    NIDigitalInputChannel: Configuration for digital input channels.
    NIDigitalOutputChannel: Configuration for digital output channels.
    NIBaseTask: Abstract base class for all task types.
    NIClockTask: Task for generating a clock signal.
    NIAnalogInputTask: Task for reading analog input data.
    NIAnalogOutputTask: Task for writing analog output data.
    NIDigitalInputTask: Task for reading digital input data.
    NIDigitalOutputTask: Task for writing digital output data.
    NIDaQ: Main class for managing multiple synchronized tasks.
"""

try:
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from time import monotonic_ns
    from typing import Self

    from nidaqmx import Task
    from nidaqmx.constants import AcquisitionType as NIAcquisitionType
    from nidaqmx.constants import (
        Edge,
        LineGrouping,
        OverwriteMode,
        TaskMode,
        TerminalConfiguration,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalMultiChannelReader
    from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalMultiChannelWriter
    from nidaqmx.system import System
    from nidaqmx.utils import flatten_channel_string
    from numpy import arange, ndarray, uint64, zeros
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

    from poulet_py import (
        AcquisitionType,
        NIAnalogBaseStimulus,
        NIAnalogCompositeStimulus,
        NIDigitalBaseStimulus,
        NIDigitalCompositeStimulus,
    )
except ImportError as e:
    raise ImportError("""
Missing 'nidaq' module. Install options:
- Dedicated:    pip install poulet_py[nidaq]
- Submodule:    pip install poulet_py[daq]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
""") from e


class NIClockHandle(BaseModel):
    """
    A handle to a clock configuration used for task synchronization.
    """

    model_config = ConfigDict(frozen=True)

    terminal: str = Field(
        ..., description="The terminal path of the clock signal (e.g., '/Dev1/Ctr0InternalOutput')."
    )
    rate: float = Field(..., description="The sampling rate in Hz.")
    samps_per_chan: int = Field(
        ...,
        description="Number of samples per channel. On continuous sampling defines the buffer size",
    )
    acquisition_type: AcquisitionType = Field(AcquisitionType.FINITE)

    def to_ni_acquisition_type(self) -> NIAcquisitionType:
        if self.acquisition_type == AcquisitionType.FINITE:
            return NIAcquisitionType.FINITE
        if self.acquisition_type == AcquisitionType.CONTINUOUS:
            return NIAcquisitionType.CONTINUOUS

        raise RuntimeError("wrong acquisition type")


class NIBaseChannel(BaseModel):
    """
    Abstract base class for all channel configurations.
    """

    name: str = Field(..., description="Unique name to assign to the channel for identification.")


class NIAnalogInputChannel(NIBaseChannel):
    """
    Configuration for an analog input channel.
    """

    number: int = Field(..., description="The AI channel number (e.g., 0 for ai0).")
    min_val: float = Field(..., description="Minimum expected voltage value in volts.")
    max_val: float = Field(..., description="Maximum expected voltage value in volts.")
    terminal_config: TerminalConfiguration = Field(
        default=TerminalConfiguration.RSE,
        description="Terminal configuration (RSE, NRSE, DIFF, PSEUDO_DIFF). Defaults to RSE.",
    )


class NIAnalogOutputChannel(NIBaseChannel):
    """
    Configuration for an analog output channel.
    """

    number: int = Field(..., description="The AO channel number (e.g., 0 for ao0).")
    min_val: float = Field(..., description="Minimum output voltage value in volts.")
    max_val: float = Field(..., description="Maximum output voltage value in volts.")


class NIDigitalInputChannel(NIBaseChannel):
    """
    Configuration for a digital input channel.
    """

    port: int = Field(..., description="The digital port number.")
    line: int = Field(..., description="The line number within the port.")


class NIDigitalOutputChannel(NIBaseChannel):
    """
    Configuration for a digital output channel.
    """

    port: int = Field(..., description="The digital port number.")
    line: int = Field(..., description="The line number within the port.")


class NIBaseTask(BaseModel, ABC):
    """
    Abstract base class for all DAQ tasks.

    This class provides common functionality for task management including
    opening, starting, stopping, and cleaning up tasks.
    """

    device: str | None = Field(
        default=None,
        description='The device name (e.g., "Dev1"). If None should be defined on NIDaQ',
    )
    name: str = Field(..., description="A descriptive name for the task.")
    clock: NIClockHandle | None = Field(default=None)

    _requires_clock: bool = PrivateAttr(default=True)
    _clock: NIClockHandle = PrivateAttr()
    _task: Task = PrivateAttr(default_factory=Task)
    _is_open: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if self._requires_clock and not self.clock:
            raise RuntimeError(f"{type(self)} requires a clock")

        if self.clock:
            self._clock = self.clock

        return self

    @property
    def requires_clock(self) -> bool:
        return self._requires_clock

    @property
    def is_running(self) -> bool:
        return not self._task.is_task_done()

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def task(self) -> Task | None:
        return self._task

    @abstractmethod
    def _open(self) -> None: ...

    def open(self) -> None:
        """
        Open the task with the given clock configuration.

        Parameters
        ----------
        clock : NIClockHandle | None, optional
            Clock configuration for task synchronization. Required for tasks
            that require a clock.

        """
        if self._is_open:
            return

        self._open()
        self._is_open = True

    def close(self) -> None:
        """
        Close the task and release all resources.

        This method should be called when the task is no longer needed to
        free system resources.
        """
        if not self._is_open:
            return

        self._task.close()
        self._task = Task()
        self._is_open = False

    def start(self) -> None:
        self._ensure_open()
        self._task.control(TaskMode.TASK_VERIFY)
        self._task.start()

    def stop(self) -> None:
        self._ensure_open()
        self._task.stop()

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("{type(self).__name__} needs to be opened first")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class NIClockTask(NIBaseTask):
    """
    Task for generating a clock signal.

    This task creates a counter output pulse train that can be used to
    synchronize other tasks.

    """

    line: int = Field(..., description="The counter line number to use (e.g., 0 for ctr0).")
    rate: float = Field(..., description="The frequency of the clock signal in Hz.")
    samps_per_chan: int = Field(
        ...,
        description="Number of samples per channel. On continuous sampling defines the buffer size.",
    )
    acquisition_type: AcquisitionType = Field(
        AcquisitionType.FINITE, description="The acquisition type (FINITE or CONTINUOUS)."
    )

    _requires_clock: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.clock:
            self.clock = NIClockHandle(
                terminal=f"/{self.device}/Ctr{self.line}InternalOutput",
                rate=self.rate,
                samps_per_chan=self.samps_per_chan,
                acquisition_type=self.acquisition_type,
            )
            self._clock = self.clock

        return self

    def _open(self) -> None:
        """
        open the clock task.

        Creates a counter output channel configured as a pulse train.
        """
        self._task.co_channels.add_co_pulse_chan_freq(
            f"{self.device}/ctr{self.line}", freq=self.rate
        )

        self._task.timing.cfg_implicit_timing(
            sample_mode=self._clock.to_ni_acquisition_type(),
            samps_per_chan=self.samps_per_chan,
        )

    def wait_until_done(self, timeout: float = 10.0):
        self._ensure_open()
        self._task.wait_until_done(timeout=timeout)


class NIAnalogInputTask(NIBaseTask):
    """
    Task for reading analog input data.

    This task configures one or more analog input channels and reads data.
    Currently only voltage channels are supported
    """

    channels: Sequence[NIAnalogInputChannel] = Field(
        ..., description="List of analog input channel configurations."
    )
    active_edge: Edge = Field(
        default=Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )

    _reader: AnalogMultiChannelReader = PrivateAttr()
    _ai_buffer: ndarray = PrivateAttr()
    _buffer: ndarray = PrivateAttr()

    def _open(self) -> None:
        for ch in self.channels:
            self._task.ai_channels.add_ai_voltage_chan(
                f"{self.device}/ai{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
                terminal_config=ch.terminal_config,
            )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock.rate,
            source=self._clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock.to_ni_acquisition_type(),
            samps_per_chan=self._clock.samps_per_chan,
        )
        self._task.in_stream.overwrite = OverwriteMode.OVERWRITE_UNREAD_SAMPLES

        self._ai_buffer = zeros((len(self.channels), self._clock.samps_per_chan), dtype="float64")
        self._buffer = zeros(
            self._clock.samps_per_chan,
            dtype=[("timestamp", "uint64"), *((ch.name, "float64") for ch in self.channels)],
        )
        self._reader = AnalogMultiChannelReader(self._task.in_stream)

    def read(self, samples: int = -1, timeout: float = 0.0) -> ndarray:
        """
        Read analog input samples as a timestamped structured array.

        Parameters
        ----------
        samples : int, optional
            Number of samples to read per channel. Use -1 to read all available samples.
        timeout : float, optional
            Timeout in seconds to wait for data. Use -1 to wait indefinitely.
        """
        self._ensure_open()

        n = self._reader.read_many_sample(
            self._ai_buffer,
            number_of_samples_per_channel=samples,
            timeout=timeout,
        )

        if n > 0:
            for idx, col in enumerate(self._buffer.dtype.names[1:]):
                self._buffer[col][:n] = self._ai_buffer[idx, :n]

            t_read = monotonic_ns()
            dt = uint64(1e9 / self._clock.rate)
            t0 = t_read - (n - 1) * dt

            timestamp = self._buffer["timestamp"][:n]
            timestamp[:] = t0
            timestamp += dt * arange(n, dtype="uint64")

        return self._buffer[:n]


class NIAnalogOutputTask(NIBaseTask):
    """
    Task for writing analog output data.

    This task configures one or more analog output channels and writes data.
    Currently only voltage channels are supported
    """

    channels: Sequence[NIAnalogOutputChannel] = Field(
        ..., description="List of analog output channel configurations."
    )
    active_edge: Edge = Field(
        default=Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )

    _writer: AnalogMultiChannelWriter = PrivateAttr()

    def _open(self) -> None:
        for ch in self.channels:
            self._task.ao_channels.add_ao_voltage_chan(
                f"{self.device}/ao{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
            )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock.rate,
            source=self._clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock.to_ni_acquisition_type(),
            samps_per_chan=self._clock.samps_per_chan,
        )

        self._writer = AnalogMultiChannelWriter(self._task.out_stream)
        self._writer.auto_start = False

    def write(self, data: ndarray, timeout: float = 10.0):
        """
        Write analog output data.

        Parameters
        ----------
        data : ndarray
            Numpy array containing the data to write.
            Shape should be (num_channels, num_samples).
        timeout : float, optional
            Timeout in seconds to wait for write completion. Defaults to 10.0.
        """
        self._ensure_open()
        self._writer.write_many_sample(data, timeout=timeout)


class NIDigitalInputTask(NIBaseTask):
    """
    Task for reading digital input data.

    This task configures one or more digital input lines and reads digital data.
    """

    channels: Sequence[NIDigitalInputChannel] = Field(
        ..., description="List of digital input channel configurations."
    )
    active_edge: Edge = Field(
        default=Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    line_grouping: LineGrouping = Field(
        default=LineGrouping.CHAN_PER_LINE,
        description="How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).",
    )

    _reader: DigitalMultiChannelReader = PrivateAttr()
    _di_buffer: ndarray = PrivateAttr()
    _buffer: ndarray = PrivateAttr()

    def _open(self) -> None:
        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        self._task.di_channels.add_di_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock.rate,
            source=self._clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock.to_ni_acquisition_type(),
            samps_per_chan=self._clock.samps_per_chan,
        )
        self._task.in_stream.overwrite = OverwriteMode.OVERWRITE_UNREAD_SAMPLES

        self._di_buffer = zeros((len(self.channels), self._clock.samps_per_chan), dtype="uint32")
        self._buffer = zeros(
            self._clock.samps_per_chan,
            dtype=[("timestamp", "uint64"), *((ch.name, "uint32") for ch in self.channels)],
        )

        self._reader = DigitalMultiChannelReader(self._task.in_stream)

    def read(self, samples: int = -1, timeout: float = 0) -> ndarray:
        """
        Read digital input data.

        Parameters
        ----------
        samples : int, optional
            Number of samples to read per channel. Use -1 to read all available samples.
        timeout : float, optional
            Timeout in seconds to wait for data. Use -1 to wait indefinitely.
        """
        self._ensure_open()

        n = self._reader.read_many_sample_port_uint32(self._di_buffer, samples, timeout)

        if n > 0:
            for idx, col in enumerate(self._buffer.dtype.names[1:]):
                self._buffer[col][:n] = self._di_buffer[idx, :n]

            t_read = monotonic_ns()
            dt = uint64(1e9 / self._clock.rate)
            t0 = t_read - (n - 1) * dt

            timestamp = self._buffer["timestamp"][:n]
            timestamp[:] = t0
            timestamp += dt * arange(n, dtype="uint64")

        return self._buffer[:n]


class NIDigitalOutputTask(NIBaseTask):
    """
    Task for writing digital output data.

    This task configures one or more digital output lines and writes digital data.
    """

    channels: Sequence[NIDigitalOutputChannel] = Field(
        ..., description="List of digital output channel configurations."
    )
    active_edge: Edge = Field(
        default=Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    line_grouping: LineGrouping = Field(
        default=LineGrouping.CHAN_PER_LINE,
        description="How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).",
    )

    _writer: DigitalMultiChannelWriter = PrivateAttr()

    def _open(self) -> None:
        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        self._task.do_channels.add_do_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock.rate,
            source=self._clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock.to_ni_acquisition_type(),
            samps_per_chan=self._clock.samps_per_chan,
        )

        self._writer = DigitalMultiChannelWriter(self._task.out_stream)
        self._writer.auto_start = False

    def write(self, data: ndarray, timeout: float = 10.0):
        """
        Write digital output data.

        Parameters
        ----------
        data : ndarray
            Numpy array containing the data to write.
            Shape depends on line_grouping configuration.
        timeout : float, optional
            Timeout in seconds to wait for write completion. Defaults to 10.0.

        Raises
        ------
        RuntimeError
            If the task has not been opened.
        """
        self._ensure_open()
        self._writer.write_many_sample_port_uint32(data, timeout=timeout)


class NIDaQ(BaseModel):
    """
    Main class for managing multiple synchronized NI-DAQmx tasks.

    This class provides a centralized way to create, open, start, and stop
    multiple tasks that are synchronized to a common clock.

    """

    device: str = Field(..., description='The device name (e.g., "Dev1") used by all tasks.')
    tasks: Sequence[NIBaseTask] = Field(...)
    acquisition_type: AcquisitionType = Field(
        default=AcquisitionType.FINITE, description="The acquisition type (FINITE or CONTINUOUS)."
    )

    _clock_task: NIClockTask = PrivateAttr()
    _read_tasks: Sequence[NIAnalogInputTask | NIDigitalInputTask] = PrivateAttr(
        default_factory=list
    )
    _write_tasks: Sequence[NIAnalogOutputTask | NIDigitalOutputTask] = PrivateAttr(
        default_factory=list
    )
    _is_open: bool = PrivateAttr(default=False)
    _executor: ThreadPoolExecutor = PrivateAttr()

    @staticmethod
    def get_available_devices() -> Sequence:
        return System.local().devices

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        names = []
        for t in self.tasks:
            if not t.device:
                t.device = self.device
            elif t.device != self.device:
                raise ValueError("Task device mismatch")

            if isinstance(t, NIClockTask):
                if self._clock_task:
                    raise ValueError("Only one NIClockTask allowed")

                t.acquisition_type = self.acquisition_type
                self._clock_task = t

            if t.name in names:
                raise RuntimeError("A task with the same name already exist")

            if isinstance(t, (NIAnalogInputTask, NIDigitalInputTask)):
                self._read_tasks = [*self._read_tasks, t]

            if isinstance(t, (NIAnalogOutputTask, NIDigitalOutputTask)):
                self._write_tasks = [*self._write_tasks, t]

            names.append(t.name)

        if not self._clock_task:
            raise RuntimeError("No NIClockTask registered")

        for task in self.tasks:
            if task != self._clock_task:
                task.clock = self._clock_task.clock

        return self

    def open(self) -> None:
        """
        open all registered tasks.

        This method opens the clock task first, then opens all other tasks
        using the exported clock handle.

        Raises
        ------
        RuntimeError
            If no NIClockTask has been registered.
        """
        if self._is_open:
            return

        self._clock_task.open()

        for task in self.tasks:
            if task != self._clock_task:
                task.open()

        if self._read_tasks:
            self._executor = ThreadPoolExecutor(len(self._read_tasks))

        self._is_open = True

    def close(self) -> None:
        """
        Close all registered tasks and release resources.

        This should be called when the NIDaQ instance is no longer needed.
        """
        if not self._is_open:
            return

        for task in self.tasks:
            task.close()

        self._executor.shutdown(wait=True)
        del self._executor

        self._is_open = False

    def start(self) -> None:
        """
        Start all registered tasks.

        Starts the clock task first, then starts all other tasks.

        Raises
        ------
        RuntimeError
            If the system has not been opened.
        """
        self._ensure_open()

        for task in self.tasks:
            if task is not self._clock_task:
                task.start()

        self._clock_task.start()

    def stop(self) -> None:
        """
        Stop all registered tasks.

        Stops all tasks except the clock task first, then stops the clock task.
        """
        self._ensure_open()

        for task in self.tasks:
            if task is not self._clock_task:
                task.stop()

        if self._clock_task:
            self._clock_task.stop()

    def write(
        self,
        stimulus: NIDigitalCompositeStimulus
        | NIAnalogCompositeStimulus
        | NIAnalogBaseStimulus
        | NIDigitalBaseStimulus,
    ):
        self._ensure_open()

        for task in self._write_tasks:
            if isinstance(task, NIDigitalOutputTask) and isinstance(
                stimulus, (NIDigitalBaseStimulus, NIDigitalCompositeStimulus)
            ):
                task.write(stimulus.build(task._clock.rate))
            elif isinstance(task, NIAnalogOutputTask) and isinstance(
                stimulus, (NIAnalogBaseStimulus, NIAnalogCompositeStimulus)
            ):
                task.write(stimulus.build(task._clock.rate))

    def read(self, samples: int = -1, timeout: float = 10.0) -> dict[str, ndarray]:
        self._ensure_open()

        ret: dict[str, ndarray] = {}

        if self._read_tasks:
            future_map = {
                self._executor.submit(task.read, samples=samples, timeout=timeout): task
                for task in self._read_tasks
            }

            for future in as_completed(future_map):
                task = future_map[future]
                ret[task.name] = future.result()
        else:
            self._clock_task._task.wait_until_done(-1)
        return ret

    def wait_until_done(self, timeout: float = 10.0):
        self._ensure_open()

        self._clock_task.wait_until_done(timeout)

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("{type(self).__name__} needs to be opened first")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
