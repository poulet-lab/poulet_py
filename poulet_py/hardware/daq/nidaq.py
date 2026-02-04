# Ordering software with hardware
from enum import Enum
from nidaqmx import Task
from nidaqmx.system import System
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.utils import flatten_channel_string
from pydantic import BaseModel, Field
from poulet_py import LOGGER


from nidaqmx.constants import TerminalConfiguration


class SampleClockConfig(BaseModel):
    rate: int
    counter: int = 0
    sample_mode: AcquisitionType = AcquisitionType.CONTINUOUS


class NIChannel(BaseModel):
    name: str
    channel_number: int


class AIChannel(NIChannel):
    min_val: float
    max_val: float
    terminal_config: TerminalConfiguration = TerminalConfiguration.RSE


class AOChannel(NIChannel):
    min_val: float
    max_val: float


class DIChannel(NIChannel):
    port: int
    line: int


class DOChannel(NIChannel):
    port: int
    line: int


class CounterChannel(NIChannel):
    pass  # extend later (edge, direction, etc.)


class NIDaQConfig(BaseModel):
    device: str = Field(..., description="NI device name (e.g. Dev1)")
    clock: SampleClockConfig
    channels: list[NIChannel]


class TaskBuilder:
    def __init__(self, config: NIDaQConfig):
        self.config = config

        self.ai_task: Task | None = None
        self.ao_task: Task | None = None
        self.do_task: Task | None = None
        self.clock_task: Task | None = None

        self.reader = None
        self.writer = None

    def build_sample_clock(self) -> str:
        cfg = self.config.clock
        dev = self.config.device

        task = Task()
        task.co_channels.add_co_pulse_chan_freq(
            f"{dev}/ctr{cfg.counter}",
            freq=cfg.rate,
        )

        task.timing.cfg_implicit_timing(sample_mode=cfg.sample_mode)

        self.clock_task = task
        return f"/{dev}/Ctr{cfg.counter}InternalOutput"

    def build_ai_task(self, clk_terminal: str):
        ai_channels = [c for c in self.config.channels if isinstance(c, AIChannel)]
        if not ai_channels:
            return

        task = Task()
        dev = self.config.device

        for ch in ai_channels:
            task.ai_channels.add_ai_voltage_chan(
                f"{dev}/ai{ch.channel_number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
                terminal_config=ch.terminal_config,
            )

        task.timing.cfg_samp_clk_timing(
            rate=self.config.clock.rate,
            source=clk_terminal,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
        )

        self.ai_task = task
        self.reader = AnalogMultiChannelReader(task.in_stream)

    def build_ao_task(self, clk_terminal: str):
        ao_channels = [c for c in self.config.channels if isinstance(c, AOChannel)]
        if not ao_channels:
            return

        task = Task()
        dev = self.config.device

        for ch in ao_channels:
            task.ao_channels.add_ao_voltage_chan(
                f"{dev}/ao{ch.channel_number}",
                name_to_assign_to_channel=ch.name,
                min_val=ch.min_val,
                max_val=ch.max_val,
            )

        task.timing.cfg_samp_clk_timing(
            rate=self.config.clock.rate,
            source=clk_terminal,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
        )

        self.ao_task = task
        self.writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)

    def build_do_task(self, clk_terminal: str):
        do_channels = [c for c in self.config.channels if isinstance(c, DOChannel)]
        if not do_channels:
            return

        task = Task()
        dev = self.config.device

        lines = [f"{dev}/port{c.port}/line{c.line}" for c in do_channels]

        task.do_channels.add_do_chan(
            flatten_channel_string(lines),
            line_grouping=LineGrouping.CHAN_PER_LINE,
        )

        task.timing.cfg_samp_clk_timing(
            rate=self.config.clock.rate,
            source=clk_terminal,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
        )

        self.do_task = task

    def build(self):
        clk_terminal = self.build_sample_clock()
        self.build_ai_task(clk_terminal)
        self.build_ao_task(clk_terminal)
        self.build_do_task(clk_terminal)


class NIDaQ:
    def __init__(self, builder: TaskBuilder):
        self.builder = builder

    def start(self):
        if self.builder.clock_task:
            self.builder.clock_task.start()
        if self.builder.ai_task:
            self.builder.ai_task.start()
        if self.builder.ao_task:
            self.builder.ao_task.start()
        if self.builder.do_task:
            self.builder.do_task.start()

    def stop(self):
        for task in (
            self.builder.ai_task,
            self.builder.ao_task,
            self.builder.do_task,
            self.builder.clock_task,
        ):
            if task:
                task.stop()

    def close(self):
        for task in (
            self.builder.ai_task,
            self.builder.ao_task,
            self.builder.do_task,
            self.builder.clock_task,
        ):
            if task:
                task.close()

    def register_ai_callback(self, sink: DataSink, samples_per_callback: int):
        reader = self.builder.reader
        task = self.builder.ai_task
        channels = [c.name for c in self.ai_channels]

        buffer = np.zeros((len(channels), samples_per_callback))

        def callback(task_handle, event_type, n_samples, cb_data):
            reader.read_many_sample(
                buffer,
                number_of_samples_per_channel=n_samples,
                timeout=0.0,
            )

            packet = DataPacket(
                source="ai",
                timestamp_ns=time.monotonic_ns(),
                data=buffer.copy(),
                channels=channels,
            )

            sink.push(packet)
            return 0

        task.register_every_n_samples_acquired_into_buffer_event(
            samples_per_callback,
            callback,
        )
