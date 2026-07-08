from adafruit_ina228 import AveragingCount, ConversionTime, Mode
from board import SCL, SDA
from busio import I2C

from poulet_py import (
    CounterSource,
    DRV2605Source,
    DRV2605Stimulus,
    HDFSink,
    INA228Source,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    TCSSource,
    TCSStimulus,
)

# create common i2c bus for all sources
i2c = I2C(SCL, SDA)

sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    DRV2605Source(name="drv2605", address=0x5A, i2c=i2c),  # , bus_frequency=400_000),
    TCSSource(
        name="tcs",
        port="/dev/ttyUSB0",
        maximum_temperature=50,
    ),
    # DCAMSource(name="dcam", acquisition_type=AcquisitionType.CONTINUOUS),
    INA228Source(
        name="ina228_mouse",
        i2c=i2c,
        address=0x41,
        bus_frequency=1_000_000,
        buffer_size=10_000,
        sample_interval_s=0.0,
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
        sample_interval_s=0.0,
        mode=Mode.CONT_BUS,
        averaging_count=AveragingCount.COUNT_16,
        bus_voltage_conv_time=ConversionTime.TIME_150_US,
        ftdi_latency_ms=1,
    ),
]


sinks = [
    HDFSink(file="./temp_drv2605_tcs_ins_10x.h5"),
]

trials = [
    StimulatorTrial(
        stimuli=TCSStimulus(
            surface=0,
            baseline=32,
            target=20,
            rise_rate=10,
            return_speed=10,
            duration=1000,
        )
    ),
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=1,
            drive_voltage=2.0,
            duration=1000,
        )
    ),
    StimulatorTrial(
        stimuli=[
            TCSStimulus(
                surface=0,
                baseline=32,
                target=40,
                rise_rate=10,
                return_speed=10,
                duration=1000,
            ),
            DRV2605Stimulus(
                waveform=16,
                repeat_count=0,
                drive_voltage=2.0,
                duration=1000,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=1,
            drive_voltage=2.0,
            duration=1000,
        )
    ),
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=0,
            drive_voltage=2.0,
            duration=1000,
        )
    ),
]

blocks = [
    StimulatorBlock(trials=trials, trial_repetitions=10),
]

experiment = StimulatorRuntime(
    name="drv2605_tcs_ina_test",
    sources=sources,
    sinks=sinks,
    blocks=blocks,
)

with experiment:
    experiment.run()
