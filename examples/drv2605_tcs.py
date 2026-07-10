from adafruit_ina228 import AveragingCount, ConversionTime, Mode
from board import SCL, SDA
from busio import I2C
from poulet_py.hardware.camera.hamamatzu.dcam import DCAMPROP
from poulet_py import (
    AcquisitionType,
    CounterSource,
    DCAMSource,
    DRV2605Source,
    DRV2605Stimulus,
    HDFSink,
    INA228Source,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    TCSSource,
    TCSStimulus
)

# create common i2c bus for all sources
i2c = I2C(SCL, SDA)

sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    DRV2605Source(
        name="drv2605",
        address=0x5A,
        i2c=i2c,
        i2c_retry_attempts=10,
        i2c_retry_backoff_s=0.001,
        continue_on_i2c_error=True,
    ),
    TCSSource(
        name="tcs",
        port="/dev/ttyUSB0",
        maximum_temperature=50,
        buffer_size=10_000,
    ),
    DCAMSource(name="dcam",
                acquisition_type=AcquisitionType.CONTINUOUS,
                resolution = (1024,1024),
                frame_rate=20,
                binning=DCAMPROP.BINNING._2,
                exposure_time=50,
                timing_mode="masterpulse"
                ),
    INA228Source(
        name="ina228_mouse",
        i2c=i2c,
        address=0x41,
        buffer_size=10_000,
        sample_interval_s=0.001,
        mode=Mode.CONT_BUS,
        averaging_count=AveragingCount.COUNT_16,
        bus_voltage_conv_time=ConversionTime.TIME_150_US,
        ftdi_latency_ms=1,
    ),
    INA228Source(
        name="ina228_pad",
        i2c=i2c,
        address=0x40,
        bus_frequency=1_000_000,
        buffer_size=10_000,
        sample_interval_s=0.001,
        mode=Mode.CONT_BUS,
        averaging_count=AveragingCount.COUNT_16,
        bus_voltage_conv_time=ConversionTime.TIME_150_US,
        ftdi_latency_ms=1,
    ),
]


sinks = [
    HDFSink(file="./temp_drv2605_continous.h5",
    queue_size=20_000,
    grow_step=10_000,
    compression = "lzf"),

]

trials = [
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                waveform=16,
                repeat_count=0,
                drive_voltage=4.0,
                duration=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=1000,
                baseline=32,
                target=32,
                rise_rate=100,
                return_speed=100,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                waveform=16,
                repeat_count=1,
                drive_voltage=4.0,
                duration=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=1000,
                baseline=32,
                target=40,
                rise_rate=100,
                return_speed=100,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                waveform=16,
                repeat_count=0,
                drive_voltage=4.0,
                duration=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=1000,
                baseline=32,
                target=40,
                rise_rate=100,
                return_speed=100,
            ),
        ]
    ),
        StimulatorTrial(
        stimuli=[DRV2605Stimulus(
            waveform=16,
            repeat_count=1,
            drive_voltage=4.0,
            duration=1000,
        ),
        TCSStimulus(
            surface=0,
            duration=1000,
            baseline=32,
            target=32,
            rise_rate=100,
            return_speed=100,
        ),
        ]
    ),
]

blocks = [
    StimulatorBlock(trials=trials, trial_repetitions=30),
]

experiment = StimulatorRuntime(
    name="drv2605_tcs_ina_test",
    sources=sources,
    sinks=sinks,
    blocks=blocks,
)

with experiment:
    experiment.run()
