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
    from time import perf_counter_ns
    from typing import Self

    from nidaqmx import Task
    from nidaqmx.constants import (
        AcquisitionType as NIAcquisitionType,
    )
    from nidaqmx.constants import (
        Edge,
        LineGrouping,
        TaskMode,
        TerminalConfiguration,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalMultiChannelReader
    from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalMultiChannelWriter
    from nidaqmx.system import System
    from nidaqmx.utils import flatten_channel_string
    from numpy import arange, empty, float64, ndarray, uint32, uint64
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

    from poulet_py import (
        AcquisitionType,
        NIAnalogBaseStimulus,
        NIAnalogCompositeStimulus,
        NIDigitalBaseStimulus,
        NIDigitalCompositeStimulus,
    )
except ImportError as e:
    msg = """
Missing 'nidaq' module. Install options:
- Dedicated:    pip install poulet_py[nidaq]
- Submodule:    pip install poulet_py[daq]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


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

        msg = "wrong acquisition type"
        raise RuntimeError(msg)


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

    _requires_clock: bool = PrivateAttr(default=True)
    _clock_handle: NIClockHandle | None = PrivateAttr(None)
    _task: Task | None = PrivateAttr(default=None)
    _is_open: bool = PrivateAttr(default=False)

    @abstractmethod
    def _open(self) -> None: ...

    def open(self, clock: NIClockHandle | None = None) -> None:
        """
        Open the task with the given clock configuration.

        Parameters
        ----------
        clock : NIClockHandle | None, optional
            Clock configuration for task synchronization. Required for tasks
            that require a clock.

        Raises
        ------
        RuntimeError
            If the task has already been opened.
        RuntimeError
            If a clock is required but not provided.
        """
        if self._requires_clock:
            if clock:
                self._clock_handle = clock
            else:
                msg = f"{type(self)} requires a clock"
                raise RuntimeError(msg)

        if self._is_open:
            msg = "Task already opened"
            raise RuntimeError(msg)

        self._task = Task()

        self._open()

        self._is_open = True

    def start(self) -> None:
        """
        Start the task.

        This method verifies and starts the underlying NI-DAQmx task.

        Raises
        ------
        RuntimeError
            If the task has not been opened.
        """
        if not self._is_open or not self._task:
            msg = f"Task '{self.name}' not opened"
            raise RuntimeError(msg)

        self._task.control(TaskMode.TASK_VERIFY)
        self._task.start()

    def stop(self) -> None:
        """
        Stop the task.

        This stops the task but does not release resources. Use `close()` to
        release resources.
        """
        if self._task:
            self._task.stop()

    def close(self) -> None:
        """
        Close the task and release all resources.

        This method should be called when the task is no longer needed to
        free system resources.
        """
        if self._task:
            self._task.close()
            self._task = None

        self._is_open = False

    @property
    def requires_clock(self) -> bool:
        return self._requires_clock

    @property
    def is_running(self) -> bool:
        """
        Check if the task is currently running.

        Returns
        -------
        bool
            True if the task is running, False otherwise.
        """
        return not self._task.is_task_done() if self._task else False

    @property
    def is_open(self) -> bool:
        """
        Check if the task is currently open.

        Returns
        -------
        bool
            True if the task is open, False otherwise.
        """
        return self._is_open

    @property
    def task(self) -> Task | None:
        """
        Get the underlying NI-DAQmx Task object.

        Returns
        -------
        Task | None
            The NI-DAQmx Task object, or None if not opened.
        """
        return self._task

    def __enter__(self, clock: NIClockHandle | None = None):
        self.open(clock=clock)
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
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

    def _open(self) -> None:
        """
        open the clock task.

        Creates a counter output channel configured as a pulse train.
        """
        if not self._task:
            msg = "Task not created"
            raise RuntimeError(msg)

        self._clock_handle = NIClockHandle(
            terminal=f"/{self.device}/Ctr{self.line}InternalOutput",
            rate=self.rate,
            samps_per_chan=self.samps_per_chan,
            acquisition_type=self.acquisition_type,
        )

        self._task.co_channels.add_co_pulse_chan_freq(
            f"{self.device}/ctr{self.line}", freq=self.rate
        )

        self._task.timing.cfg_implicit_timing(
            sample_mode=self._clock_handle.to_ni_acquisition_type(),
            samps_per_chan=self.samps_per_chan,
        )

    @property
    def clock_handle(self) -> NIClockHandle | None:
        return self._clock_handle

    def wait(self, timeout: float = 10.0):
        if not self._task:
            msg = "NIClockTask not opened"
            raise RuntimeError(msg)

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

    _reader: AnalogMultiChannelReader | None = PrivateAttr(None)
    _ai_buffer: ndarray | None = PrivateAttr(None)
    _buffer: ndarray | None = PrivateAttr(None)

    def _ensure_buffer(self, samples: int):
        if not self._reader or self._ai_buffer is None:
            msg = "Task not opened"
            raise RuntimeError(msg)

        capacity = self._ai_buffer.shape[1]

        if samples > capacity:
            new_capacity = max(samples, capacity * 2)

            self._ai_buffer = empty((len(self.channels), new_capacity), dtype=float64)

            self._buffer = empty(
                new_capacity,
                dtype=[("timestamp", uint64), *((ch.name, float64) for ch in self.channels)],
            )

    def _open(self) -> None:
        if not self._task or not self._clock_handle:
            msg = "Task not created"
            raise RuntimeError(msg)

        for ch in self.channels:
            self._task.ai_channels.add_ai_voltage_chan(
                f"{self.device}/ai{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
                terminal_config=ch.terminal_config,
            )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock_handle.rate,
            source=self._clock_handle.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock_handle.to_ni_acquisition_type(),
            samps_per_chan=self._clock_handle.samps_per_chan,
        )

        self._ai_buffer = empty(
            (len(self.channels), self._clock_handle.samps_per_chan), dtype=float64
        )
        self._buffer = empty(
            self._clock_handle.samps_per_chan,
            dtype=[("timestamp", uint64), *((ch.name, float64) for ch in self.channels)],
        )
        self._reader = AnalogMultiChannelReader(self._task.in_stream)

    def read(self, samples: int = -1, timeout: float = 10.0) -> ndarray:
        """
        Read analog input data.

        Parameters
        ----------
        data : ndarray
            Pre-allocated numpy array to store the read data.
            Shape should be (num_channels, num_samples).
        samples : int, optional
            Number of samples to read per channel. Use -1 to read all available samples.
            Defaults to -1.
        timeout : float, optional
            Timeout in seconds to wait for data. Defaults to 10.0.

        Raises
        ------
        RuntimeError
            If the task has not been opened.
        """
        if (
            not self._reader
            or not self._clock_handle
            or self._ai_buffer is None
            or self._buffer is None
        ):
            msg = "Task not opened"
            raise RuntimeError(msg)

        self._ensure_buffer(samples)
        n = self._reader.read_many_sample(self._ai_buffer, samples, timeout)

        if n > 0:
            data = self._ai_buffer[:, :n]
            for idx, col in enumerate(self._buffer.dtype.names[1:]):
                self._buffer[col][:n] = data[idx]

            t_read = perf_counter_ns()
            dt = uint64(1e9 / self._clock_handle.rate)
            t0 = t_read - (n - 1) * dt

            timestamps = self._buffer["timestamp"][:n]
            timestamps[:] = t0
            timestamps += dt * arange(n, dtype=uint64)

        return self._buffer[:n].copy()


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

    _writer: AnalogMultiChannelWriter | None = PrivateAttr(None)

    def _open(self) -> None:
        if not self._task or not self._clock_handle:
            msg = "Task not created"
            raise RuntimeError(msg)

        for ch in self.channels:
            self._task.ao_channels.add_ao_voltage_chan(
                f"{self.device}/ao{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
            )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock_handle.rate,
            source=self._clock_handle.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock_handle.to_ni_acquisition_type(),
            samps_per_chan=self._clock_handle.samps_per_chan,
        )

        self._writer = AnalogMultiChannelWriter(self._task.out_stream, auto_start=False)

        # self.write(zeros((len(self.channels),1)))

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

        Raises
        ------
        RuntimeError
            If the task has not been opened.
        """
        if not self._writer:
            msg = "Task not opened"
            raise RuntimeError(msg)

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

    _reader: DigitalMultiChannelReader | None = PrivateAttr(None)
    _di_buffer: ndarray | None = PrivateAttr(None)
    _buffer: ndarray | None = PrivateAttr(None)

    def _ensure_buffer(self, samples: int):
        if not self._reader or self._di_buffer is None:
            msg = "Task not opened"
            raise RuntimeError(msg)

        capacity = self._di_buffer.shape[1]

        if samples > capacity:
            new_capacity = max(samples, capacity * 2)

            self._di_buffer = empty((len(self.channels), new_capacity), dtype=uint32)

            self._buffer = empty(
                new_capacity,
                dtype=[("timestamp", uint64), *((ch.name, float64) for ch in self.channels)],
            )

    def _open(self) -> None:
        if not self._task or not self._clock_handle:
            msg = "Task not created"
            raise RuntimeError(msg)

        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        self._task.di_channels.add_di_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock_handle.rate,
            source=self._clock_handle.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock_handle.to_ni_acquisition_type(),
            samps_per_chan=self._clock_handle.samps_per_chan,
        )

        self._di_buffer = empty(
            (len(self.channels), self._clock_handle.samps_per_chan), dtype=uint32
        )
        self._buffer = empty(
            self._clock_handle.samps_per_chan,
            dtype=[("timestamp", uint64), *((ch.name, float64) for ch in self.channels)],
        )

        self._reader = DigitalMultiChannelReader(self._task.in_stream)

    def read(self, samples: int = -1, timeout: float = 10.0) -> ndarray:
        """
        Read digital input data.

        Parameters
        ----------
        data : ndarray
            Pre-allocated numpy array to store the read data.
            Shape depends on line_grouping configuration.
        samples : int, optional
            Number of samples to read per channel. Use -1 to read all available samples.
            Defaults to -1.
        timeout : float, optional
            Timeout in seconds to wait for data. Defaults to 10.0.

        Raises
        ------
        RuntimeError
            If the task has not been opened.
        """
        if (
            not self._reader
            or not self._clock_handle
            or self._di_buffer is None
            or self._buffer is None
        ):
            msg = "Task not opened"
            raise RuntimeError(msg)

        self._ensure_buffer(samples)

        n = self._reader.read_many_sample_port_uint32(self._di_buffer, samples, timeout)

        if n > 0:
            data = self._di_buffer[:, :n]
            for idx, col in enumerate(self._buffer.dtype.names[1:]):
                self._buffer[col][:n] = data[idx]

            t_read = perf_counter_ns()
            dt = uint64(1e9 / self._clock_handle.rate)
            t0 = t_read - (n - 1) * dt

            timestamps = self._buffer["timestamp"][:n]
            timestamps[:] = t0
            timestamps += dt * arange(n, dtype=uint64)

        return self._buffer[:n].copy()


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

    _writer: DigitalMultiChannelWriter | None = PrivateAttr(None)

    def _open(self) -> None:
        if not self._task or not self._clock_handle:
            msg = "Task not created"
            raise RuntimeError(msg)

        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        self._task.do_channels.add_do_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        self._task.timing.cfg_samp_clk_timing(
            rate=self._clock_handle.rate,
            source=self._clock_handle.terminal,
            active_edge=self.active_edge,
            sample_mode=self._clock_handle.to_ni_acquisition_type(),
            samps_per_chan=self._clock_handle.samps_per_chan,
        )

        self._writer = DigitalMultiChannelWriter(self._task.out_stream, auto_start=False)
        self._task = self._task
        self._is_open = True

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
        if not self._writer:
            msg = "Task not opened"
            raise RuntimeError(msg)

        self._writer.write_many_sample_port_uint32(data, timeout=timeout)


class NIDaQ(BaseModel):
    """
    Main class for managing multiple synchronized NI-DAQmx tasks.

    This class provides a centralized way to create, open, start, and stop
    multiple tasks that are synchronized to a common clock.

    """

    device: str = Field(..., description='The device name (e.g., "Dev1") used by all tasks.')
    tasks: Sequence[NIBaseTask] = Field(...)

    _clock_task: NIClockTask | None = PrivateAttr(None)
    _read_tasks: Sequence[NIAnalogInputTask | NIDigitalInputTask] = PrivateAttr(
        default_factory=list
    )
    _write_tasks: Sequence[NIAnalogOutputTask | NIDigitalOutputTask] = PrivateAttr(
        default_factory=list
    )
    _is_open: bool = PrivateAttr(default=False)
    _executor: ThreadPoolExecutor | None = PrivateAttr(None)

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
                msg = "Task device mismatch"
                raise ValueError(msg)

            if isinstance(t, NIClockTask):
                if self._clock_task:
                    msg = "Only one NIClockTask allowed"
                    raise ValueError(msg)
                self._clock_task = t

            if t.name in names:
                msg = "A task with the same name already exist"
                raise RuntimeError(msg)
            if isinstance(t, (NIAnalogInputTask, NIDigitalInputTask)):
                self._read_tasks = [*self._read_tasks, t]
            if isinstance(t, (NIAnalogOutputTask, NIDigitalOutputTask)):
                self._write_tasks = [*self._write_tasks, t]
            names.append(t.name)

        return self

    def write(
        self,
        stimulus: NIDigitalCompositeStimulus
        | NIAnalogCompositeStimulus
        | NIAnalogBaseStimulus
        | NIDigitalBaseStimulus,
    ):
        if not self._is_open or not self._clock_task:
            msg = "NIDaQ must be first opened"
            raise RuntimeError(msg)

        for task in self._write_tasks:
            if isinstance(task, NIDigitalOutputTask) and isinstance(
                stimulus, (NIDigitalBaseStimulus, NIDigitalCompositeStimulus)
            ):
                task.write(stimulus.build(task._clock_handle.rate))
            elif isinstance(task, NIAnalogOutputTask) and isinstance(
                stimulus, (NIAnalogBaseStimulus, NIAnalogCompositeStimulus)
            ):
                task.write(stimulus.build(task._clock_handle.rate))

    def read(self, samples: int = -1, timeout: float = 10.0) -> dict[str, ndarray]:
        if not self._is_open or not self._executor:
            msg = "NIDaQ must be first opened"
            raise RuntimeError(msg)

        ret: dict[str, ndarray] = {}

        future_map = {
            self._executor.submit(task.read, samples=samples, timeout=timeout): task
            for task in self._read_tasks
        }

        for future in as_completed(future_map):
            task = future_map[future]
            ret[task.name] = future.result()

        return ret

    def wait(self, timeout: float = 10.0):
        if not self._is_open or not self._clock_task:
            msg = "NIDaQ must be first opened"
            raise RuntimeError(msg)

        self._clock_task.wait(timeout)

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

        if not self._clock_task:
            msg = "No NIClockTask registered"
            raise RuntimeError(msg)

        self._clock_task.open()

        for task in self.tasks:
            if task != self._clock_task:
                task.open(self._clock_task.clock_handle)

        self._executor = ThreadPoolExecutor(len(self._read_tasks)) if self._read_tasks else None

        self._is_open = True

    def start(self) -> None:
        """
        Start all registered tasks.

        Starts the clock task first, then starts all other tasks.

        Raises
        ------
        RuntimeError
            If the system has not been opened.
        """
        if not self._is_open or not self._clock_task:
            msg = "NIDaQ must be first opened"
            raise RuntimeError(msg)

        for task in self.tasks:
            if task is not self._clock_task:
                task.start()

        self._clock_task.start()

    def stop(self) -> None:
        """
        Stop all registered tasks.

        Stops all tasks except the clock task first, then stops the clock task.
        """
        if not self._is_open:
            return

        for task in self.tasks:
            if task is not self._clock_task:
                task.stop()

        if self._clock_task:
            self._clock_task.stop()

    def close(self) -> None:
        """
        Close all registered tasks and release resources.

        This should be called when the NIDaQ instance is no longer needed.
        """
        if not self._is_open:
            return

        for task in self.tasks:
            task.close()

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._is_open = False

    def __enter__(self):
        """
        Context manager entry.

        opens and starts all tasks when entering a context.

        Returns
        -------
        NIDaQ
            The NIDaQ instance.
        """
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Context manager exit.

        Stops and closes all tasks when exiting a context.

        Parameters
        ----------
        exc_type : type | None
            Exception type if an exception occurred.
        exc : Exception | None
            Exception instance if an exception occurred.
        tb : TracebackType | None
            Traceback if an exception occurred.
        """
        self.stop()
        self.close()
