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
    from dataclasses import dataclass

    from nidaqmx import Task
    from nidaqmx.constants import (
        AcquisitionType,
        Edge,
        LineGrouping,
        TaskMode,
        TerminalConfiguration,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalMultiChannelReader
    from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalMultiChannelWriter
    from nidaqmx.utils import flatten_channel_string
    from numpy import ndarray
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError as e:
    msg = """
Missing 'nidaq' module. Install options:
- Dedicated:    pip install poulet_py[nidaq]
- Submodule:    pip install poulet_py[daq]
- Module:       pip install poulet_py[daq]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


@dataclass(frozen=True)
class NIClockHandle:
    """
    A handle to a clock configuration used for task synchronization.

    Attributes
    ----------
    terminal : str
        The terminal path of the clock signal (e.g., "/Dev1/Ctr0InternalOutput").
    rate : float
        The sampling rate in Hz.
    samps_per_chan : int
        Number of samples per channel. Use -1 for continuous sampling.
    """

    terminal: str = Field(
        ..., description="The terminal path of the clock signal (e.g., '/Dev1/Ctr0InternalOutput')."
    )
    rate: float = Field(..., description="The sampling rate in Hz.")
    samps_per_chan: int = Field(
        ..., description="Number of samples per channel. Use -1 for continuous sampling."
    )


class NIBaseChannel(BaseModel):
    """
    Abstract base class for all channel configurations.

    Attributes
    ----------
    name : str
        Unique name to assign to the channel for identification.
    """

    name: str = Field(..., description="Unique name to assign to the channel for identification.")


class NIAnalogInputChannel(NIBaseChannel):
    """
    Configuration for an analog input channel.

    Attributes
    ----------
    name : str
        Unique name to assign to the channel for identification.
    number : int
        The AI channel number (e.g., 0 for ai0).
    min_val : float
        Minimum expected voltage value in volts.
    max_val : float
        Maximum expected voltage value in volts.
    terminal_config : TerminalConfiguration
        Terminal configuration (RSE, NRSE, DIFF, PSEUDO_DIFF).
        Defaults to TerminalConfiguration.RSE.
    """

    number: int = Field(..., description="The AI channel number (e.g., 0 for ai0).")
    min_val: float = Field(..., description="Minimum expected voltage value in volts.")
    max_val: float = Field(..., description="Maximum expected voltage value in volts.")
    terminal_config: TerminalConfiguration = Field(
        TerminalConfiguration.RSE,
        description="Terminal configuration (RSE, NRSE, DIFF, PSEUDO_DIFF). Defaults to RSE.",
    )


class NIAnalogOutputChannel(NIBaseChannel):
    """
    Configuration for an analog output channel.

    Attributes
    ----------
    name : str
        Unique name to assign to the channel for identification.
    number : int
        The AO channel number (e.g., 0 for ao0).
    min_val : float
        Minimum output voltage value in volts.
    max_val : float
        Maximum output voltage value in volts.
    """

    number: int = Field(..., description="The AO channel number (e.g., 0 for ao0).")
    min_val: float = Field(..., description="Minimum output voltage value in volts.")
    max_val: float = Field(..., description="Maximum output voltage value in volts.")


class NIDigitalInputChannel(NIBaseChannel):
    """
    Configuration for a digital input channel.

    Attributes
    ----------
    name : str
        Unique name to assign to the channel for identification.
    port : int
        The digital port number.
    line : int
        The line number within the port.
    """

    port: int = Field(..., description="The digital port number.")
    line: int = Field(..., description="The line number within the port.")


class NIDigitalOutputChannel(NIBaseChannel):
    """
    Configuration for a digital output channel.

    Attributes
    ----------
    name : str
        Unique name to assign to the channel for identification.
    port : int
        The digital port number.
    line : int
        The line number within the port.
    """

    port: int = Field(..., description="The digital port number.")
    line: int = Field(..., description="The line number within the port.")


class NIBaseTask(BaseModel, ABC):
    """
    Abstract base class for all DAQ tasks.

    This class provides common functionality for task management including
    building, starting, stopping, and cleaning up tasks.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.

    Private Attributes
    ------------------
    _requires_clock : bool
        Whether the task requires an external clock. Defaults to True.
    _task : Task | None
        The underlying NI-DAQmx Task object.
    _built : bool
        Flag indicating whether the task has been built.

    Methods
    -------
    build(clock: NIClockHandle | None = None) -> None
        Abstract method to build the task.
    start() -> None
        Starts the task.
    stop() -> None
        Stops the task.
    close() -> None
        Closes the task and releases resources.
    is_running() -> bool
        Checks if the task is currently running.
    """

    device: str = Field(..., description='The device name (e.g., "Dev1").')
    name: str = Field(..., description="A descriptive name for the task.")

    _requires_clock: bool = PrivateAttr(True)
    _task: Task | None = PrivateAttr(None)
    _built: bool = PrivateAttr(False)

    @abstractmethod
    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the task with the given clock configuration.

        Parameters
        ----------
        clock : NIClockHandle | None, optional
            Clock configuration for task synchronization. Required for tasks
            that require a clock.

        Raises
        ------
        RuntimeError
            If the task has already been built.
        RuntimeError
            If a clock is required but not provided.
        """
        ...

    def start(self) -> None:
        """
        Start the task.

        This method verifies and starts the underlying NI-DAQmx task.

        Raises
        ------
        RuntimeError
            If the task has not been built.
        """
        if not self._built or not self._task:
            raise RuntimeError(f"Task '{self.name}' not built")

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
            self._built = False

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
    def task(self) -> Task | None:
        """
        Get the underlying NI-DAQmx Task object.

        Returns
        -------
        Task | None
            The NI-DAQmx Task object, or None if not built.
        """
        return self._task


class NIClockTask(NIBaseTask):
    """
    Task for generating a clock signal.

    This task creates a counter output pulse train that can be used to
    synchronize other tasks.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.
    line : int
        The counter line number to use (e.g., 0 for ctr0).
    rate : float
        The frequency of the clock signal in Hz.
    samps_per_chan : int
        Number of samples per channel. Use -1 for continuous sampling.
    sample_mode : AcquisitionType
        The acquisition type (FINITE or CONTINUOUS).
        Defaults to AcquisitionType.CONTINUOUS.
    """

    line: int = Field(..., description="The counter line number to use (e.g., 0 for ctr0).")
    rate: float = Field(..., description="The frequency of the clock signal in Hz.")
    samps_per_chan: int = Field(
        ..., description="Number of samples per channel. Use -1 for continuous sampling."
    )
    sample_mode: AcquisitionType = Field(
        AcquisitionType.CONTINUOUS, description="The acquisition type (FINITE or CONTINUOUS)."
    )

    _requires_clock = False

    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the clock task.

        Creates a counter output channel configured as a pulse train.

        Parameters
        ----------
        clock : NIClockHandle | None, optional
            Not used for NIClockTask (ignored).

        Raises
        ------
        RuntimeError
            If the task has already been built.
        """
        if self._built:
            msg = "NIClockTask already built"
            raise RuntimeError(msg)

        task = Task()

        task.co_channels.add_co_pulse_chan_freq(
            f"{self.device}/ctr{self.line}",
            freq=self.rate,
        )
        task.timing.cfg_implicit_timing(
            sample_mode=self.sample_mode,
            samps_per_chan=self.samps_per_chan,
        )

        self._task = task
        self._built = True

    def export(self) -> NIClockHandle:
        """
        Export the clock configuration for use by other tasks.

        Returns
        -------
        NIClockHandle
            A handle containing the clock terminal path, rate, and samples per channel.

        Raises
        ------
        RuntimeError
            If the task has not been built.
        """
        if not self._built:
            msg = "NIClockTask not built"
            raise RuntimeError(msg)

        return NIClockHandle(
            terminal=f"/{self.device}/Ctr{self.line}InternalOutput",
            rate=self.rate,
            samps_per_chan=self.samps_per_chan,
        )


class NIAnalogInputTask(NIBaseTask):
    """
    Task for reading analog input data.

    This task configures one or more analog input channels and reads voltage data.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.
    channels : list[NIAnalogInputChannel]
        List of analog input channel configurations.
    active_edge : Edge
        The active edge for sampling (RISING or FALLING).
        Defaults to Edge.RISING.
    sample_mode : AcquisitionType
        The acquisition type (FINITE or CONTINUOUS).
        Defaults to AcquisitionType.CONTINUOUS.

    Private Attributes
    ------------------
    _reader : AnalogMultiChannelReader | None
        The stream reader for analog input data.
    """

    channels: list[NIAnalogInputChannel] = Field(
        ..., description="List of analog input channel configurations."
    )
    active_edge: Edge = Field(
        Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    sample_mode: AcquisitionType = Field(
        AcquisitionType.CONTINUOUS, description="The acquisition type (FINITE or CONTINUOUS)."
    )

    _reader: AnalogMultiChannelReader | None = PrivateAttr(None)

    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the analog input task.

        Configures analog input channels and sets up timing based on the provided clock.

        Parameters
        ----------
        clock : NIClockHandle | None
            Clock configuration for synchronization. Required.

        Raises
        ------
        RuntimeError
            If no clock is provided.
        RuntimeError
            If the task has already been built.
        """
        if not clock:
            msg = "NIAnalogInputTask requires a clock"
            raise RuntimeError(msg)
        if self._built:
            msg = "Task already built"
            raise RuntimeError(msg)

        task = Task()

        for ch in self.channels:
            task.ai_channels.add_ai_voltage_chan(
                f"{self.device}/ai{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
                terminal_config=ch.terminal_config,
            )

        task.timing.cfg_samp_clk_timing(
            rate=clock.rate,
            source=clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self.sample_mode,
            samps_per_chan=clock.samps_per_chan,
        )

        self._reader = AnalogMultiChannelReader(task.in_stream)
        self._task = task
        self._built = True

    def read(self, data: ndarray, samples: int = -1, timeout: float = 10.0):
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
            If the task has not been built.
        """
        if not self._reader:
            msg = "Task not built"
            raise RuntimeError(msg)
        self._reader.read_many_sample(data, samples, timeout)


class NIAnalogOutputTask(NIBaseTask):
    """
    Task for writing analog output data.

    This task configures one or more analog output channels and writes voltage data.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.
    channels : list[NIAnalogOutputChannel]
        List of analog output channel configurations.
    active_edge : Edge
        The active edge for sampling (RISING or FALLING).
        Defaults to Edge.RISING.
    sample_mode : AcquisitionType
        The acquisition type (FINITE or CONTINUOUS).
        Defaults to AcquisitionType.CONTINUOUS.

    Private Attributes
    ------------------
    _writer : AnalogMultiChannelWriter | None
        The stream writer for analog output data.
    """

    channels: list[NIAnalogOutputChannel] = Field(
        ..., description="List of analog output channel configurations."
    )
    active_edge: Edge = Field(
        Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    sample_mode: AcquisitionType = Field(
        AcquisitionType.CONTINUOUS, description="The acquisition type (FINITE or CONTINUOUS)."
    )

    _writer: AnalogMultiChannelWriter | None = PrivateAttr(None)

    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the analog output task.

        Configures analog output channels and sets up timing based on the provided clock.

        Parameters
        ----------
        clock : NIClockHandle | None
            Clock configuration for synchronization. Required.

        Raises
        ------
        RuntimeError
            If no clock is provided.
        RuntimeError
            If the task has already been built.
        """
        if not clock:
            msg = "NIAnalogOutputTask requires a clock"
            raise RuntimeError(msg)
        if self._built:
            msg = "Task already built"
            raise RuntimeError(msg)

        task = Task()

        for ch in self.channels:
            task.ao_channels.add_ao_voltage_chan(
                f"{self.device}/ao{ch.number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
            )

        task.timing.cfg_samp_clk_timing(
            rate=clock.rate,
            source=clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self.sample_mode,
            samps_per_chan=clock.samps_per_chan,
        )

        self._writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)
        self._task = task
        self._built = True

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
            If the task has not been built.
        """
        if not self._writer:
            msg = "Task not built"
            raise RuntimeError(msg)
        self._writer.write_many_sample(data, timeout=timeout)


class NIDigitalInputTask(NIBaseTask):
    """
    Task for reading digital input data.

    This task configures one or more digital input lines and reads digital data.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.
    channels : list[NIDigitalInputChannel]
        List of digital input channel configurations.
    active_edge : Edge
        The active edge for sampling (RISING or FALLING).
        Defaults to Edge.RISING.
    sample_mode : AcquisitionType
        The acquisition type (FINITE or CONTINUOUS).
        Defaults to AcquisitionType.CONTINUOUS.
    line_grouping : LineGrouping
        How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).
        Defaults to LineGrouping.CHAN_PER_LINE.

    Private Attributes
    ------------------
    _reader : DigitalMultiChannelReader | None
        The stream reader for digital input data.
    """

    channels: list[NIDigitalInputChannel] = Field(
        ..., description="List of digital input channel configurations."
    )
    active_edge: Edge = Field(
        Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    sample_mode: AcquisitionType = Field(
        AcquisitionType.CONTINUOUS, description="The acquisition type (FINITE or CONTINUOUS)."
    )
    line_grouping: LineGrouping = Field(
        LineGrouping.CHAN_PER_LINE,
        description="How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).",
    )

    _reader: DigitalMultiChannelReader | None = PrivateAttr(None)

    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the digital input task.

        Configures digital input channels and sets up timing based on the provided clock.

        Parameters
        ----------
        clock : NIClockHandle | None
            Clock configuration for synchronization. Required.

        Raises
        ------
        RuntimeError
            If no clock is provided.
        RuntimeError
            If the task has already been built.
        """
        if not clock:
            msg = "NIDigitalInputTask requires a clock"
            raise RuntimeError(msg)
        if self._built:
            msg = "Task already built"
            raise RuntimeError(msg)

        task = Task()

        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        task.di_channels.add_di_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        task.timing.cfg_samp_clk_timing(
            rate=clock.rate,
            source=clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self.sample_mode,
            samps_per_chan=clock.samps_per_chan,
        )

        self._reader = DigitalMultiChannelReader(task.in_stream)
        self._task = task
        self._built = True

    def read(self, data: ndarray, samples: int = -1, timeout: float = 10.0):
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
            If the task has not been built.
        """
        if not self._reader:
            msg = "Task not built"
            raise RuntimeError(msg)
        self._reader.read_many_sample_port_uint32(data, samples, timeout)


class NIDigitalOutputTask(NIBaseTask):
    """
    Task for writing digital output data.

    This task configures one or more digital output lines and writes digital data.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1").
    name : str
        A descriptive name for the task.
    channels : list[NIDigitalOutputChannel]
        List of digital output channel configurations.
    active_edge : Edge
        The active edge for sampling (RISING or FALLING).
        Defaults to Edge.RISING.
    sample_mode : AcquisitionType
        The acquisition type (FINITE or CONTINUOUS).
        Defaults to AcquisitionType.CONTINUOUS.
    line_grouping : LineGrouping
        How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).
        Defaults to LineGrouping.CHAN_PER_LINE.

    Private Attributes
    ------------------
    _writer : DigitalMultiChannelWriter | None
        The stream writer for digital output data.
    """

    channels: list[NIDigitalOutputChannel] = Field(
        ..., description="List of digital output channel configurations."
    )
    active_edge: Edge = Field(
        Edge.RISING, description="The active edge for sampling (RISING or FALLING)."
    )
    sample_mode: AcquisitionType = Field(
        AcquisitionType.CONTINUOUS, description="The acquisition type (FINITE or CONTINUOUS)."
    )
    line_grouping: LineGrouping = Field(
        LineGrouping.CHAN_PER_LINE,
        description="How to group digital lines (CHAN_PER_LINE or CHAN_FOR_ALL_LINES).",
    )

    _writer: DigitalMultiChannelWriter | None = PrivateAttr(None)

    def build(self, clock: NIClockHandle | None = None) -> None:
        """
        Build the digital output task.

        Configures digital output channels and sets up timing based on the provided clock.

        Parameters
        ----------
        clock : NIClockHandle | None
            Clock configuration for synchronization. Required.

        Raises
        ------
        RuntimeError
            If no clock is provided.
        RuntimeError
            If the task has already been built.
        """
        if not clock:
            msg = "NIDigitalOutputTask requires a clock"
            raise RuntimeError(msg)
        if self._built:
            msg = "Task already built"
            raise RuntimeError(msg)

        task = Task()

        lines = [f"{self.device}/port{c.port}/line{c.line}" for c in self.channels]

        task.do_channels.add_do_chan(
            flatten_channel_string(lines),
            line_grouping=self.line_grouping,
        )

        task.timing.cfg_samp_clk_timing(
            rate=clock.rate,
            source=clock.terminal,
            active_edge=self.active_edge,
            sample_mode=self.sample_mode,
            samps_per_chan=clock.samps_per_chan,
        )

        self._writer = DigitalMultiChannelWriter(task.out_stream, auto_start=False)
        self._task = task
        self._built = True

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
            If the task has not been built.
        """
        if not self._writer:
            msg = "Task not built"
            raise RuntimeError(msg)
        self._writer.write_many_sample_port_uint32(data, timeout=timeout)


class NIDaQ(BaseModel):
    """
    Main class for managing multiple synchronized NI-DAQmx tasks.

    This class provides a centralized way to create, build, start, and stop
    multiple tasks that are synchronized to a common clock.

    Attributes
    ----------
    device : str
        The device name (e.g., "Dev1") used by all tasks.

    Private Attributes
    ------------------
    _tasks : list[NIBaseTask]
        List of all registered tasks.
    _clock_task : NIClockTask | None
        The clock task used for synchronization.
    _clock_handle : NIClockHandle | None
        The clock handle exported by the clock task.

    Methods
    -------
    add_task(task: NIBaseTask) -> None
        Add a task to the manager.
    build_all() -> None
        Build all registered tasks.
    start_all() -> None
        Start all registered tasks.
    stop_all() -> None
        Stop all registered tasks.
    close_all() -> None
        Close all registered tasks.
    __enter__()
        Context manager entry.
    __exit__(exc_type, exc, tb)
        Context manager exit.
    """

    device: str = Field(..., description='The device name (e.g., "Dev1") used by all tasks.')

    _tasks: dict[str, NIBaseTask] = PrivateAttr(default_factory=dict)
    _clock_task: NIClockTask | None = PrivateAttr(None)
    _clock_handle: NIClockHandle | None = PrivateAttr(None)

    def add_task(self, task: NIBaseTask) -> None:
        """
        Add a task to the manager.

        Parameters
        ----------
        task : NIBaseTask
            The task to add.

        Raises
        ------
        ValueError
            If the task device doesn't match the NIDaQ device.
        ValueError
            If trying to add more than one NIClockTask.
        """
        if task.device != self.device:
            msg = "Task device mismatch"
            raise ValueError(msg)

        if isinstance(task, NIClockTask):
            if self._clock_task:
                msg = "Only one NIClockTask allowed"
                raise ValueError(msg)
            self._clock_task = task

        if task.name in self._tasks.keys():
            msg = "A task with the same name already exist"
            raise RuntimeError(msg)

        self._tasks[task.name] = task

    def build_all(self) -> None:
        """
        Build all registered tasks.

        This method builds the clock task first, then builds all other tasks
        using the exported clock handle.

        Raises
        ------
        RuntimeError
            If no NIClockTask has been registered.
        """
        if not self._clock_task:
            msg = "No NIClockTask registered"
            raise RuntimeError(msg)

        self._clock_task.build()
        self._clock_handle = self._clock_task.export()

        for name, task in self._tasks.items():
            task.build(self._clock_handle if task._requires_clock else None)

    def start_all(self) -> None:
        """
        Start all registered tasks.

        Starts the clock task first, then starts all other tasks.

        Raises
        ------
        RuntimeError
            If the system has not been built.
        """
        if not self._clock_task:
            msg = "System not built"
            raise RuntimeError(msg)

        self._clock_task.start()
        for name, task in self._tasks.items():
            if task is not self._clock_task:
                task.start()

    def stop_all(self) -> None:
        """
        Stop all registered tasks.

        Stops all tasks except the clock task first, then stops the clock task.
        """
        for name, task in self._tasks.items():
            if task is not self._clock_task:
                task.stop()
        if self._clock_task:
            self._clock_task.stop()

    def close_all(self) -> None:
        """
        Close all registered tasks and release resources.

        This should be called when the NIDaQ instance is no longer needed.
        """
        for name, task in self._tasks.items():
            task.close()

    def __enter__(self):
        """
        Context manager entry.

        Builds and starts all tasks when entering a context.

        Returns
        -------
        NIDaQ
            The NIDaQ instance.
        """
        self.build_all()
        self.start_all()
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
        self.stop_all()
        self.close_all()
