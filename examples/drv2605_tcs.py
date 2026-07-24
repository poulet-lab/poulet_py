from adafruit_ina228 import AveragingCount, ConversionTime, Mode
from board import SCL, SDA
from busio import I2C

from poulet_py import (
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
    TCSStimulus,
)
from poulet_py.hardware.camera.hamamatzu.dcam import DCAMPROP, AcquisitionType

# create common i2c bus for all sources
i2c = I2C(SCL, SDA, frequency=400_000)
# this is a temp fix to test weather pyftdi error handling introduces erroneous values...
# instead move to poulet_py error handling and read discard
# maximum o retries internally through pyftdi, eliminates warning: retry exchange handling and
# liminates erroneous data writing into buffer
i2c._i2c._i2c.set_retry_count(1)
sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    DRV2605Source(
        name="drv2605",
        i2c=i2c,
        motor_type="erm",
        loop_mode="closed_loop",
        calibrate=False,
    ),
    TCSSource(
        name="tcs",
        port="/dev/ttyUSB0",
        maximum_temperature=50,
        buffer_size=500,
    ),
    DCAMSource(
        name="dcam",
        acquisition_type=AcquisitionType.CONTINUOUS,
        resolution=(1024, 1024),
        frame_rate=20,
        binning=DCAMPROP.BINNING._2,
        exposure_time=35,
        buffer_size=10,
        timing_mode="masterpulse",
        output_trigger_kind=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
    ),
    INA228Source(
        name="ina228_mouse",
        i2c=i2c,
        address=0x41,
        buffer_size=50,
        sample_rate_Hz=100,
        mode=Mode.CONT_BUS,
        averaging_count=AveragingCount.COUNT_16,
        bus_voltage_conv_time=ConversionTime.TIME_150_US,
        temperature=True,
    ),
    INA228Source(
        name="ina228_pad",
        i2c=i2c,
        address=0x40,
        buffer_size=50,
        sample_rate_Hz=100,
        mode=Mode.CONT_BUS,
        averaging_count=AveragingCount.COUNT_16,
        bus_voltage_conv_time=ConversionTime.TIME_150_US,
        temperature=True,
    ),
]


sinks = [
    HDFSink(
        file="./temp_whynoframes.h5",
        queue_size=1_000,
        grow_step=100,
        compression="lzf",
    ),
]


# Exact stimulus and trial structure from drv2605_tcs.py.
# Only the DRV2605L usage mode has been changed to RTP.
trials = [
    StimulatorTrial(
        stimuli=[
            TCSStimulus(
                surface=0,
                duration=3000,
                baseline=32,
                target=32,
                rise_rate=100,
                return_speed=100,
                pre_delay=1000,
                post_delay=1000,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                mode="rtp",
                drive_voltage=3,
                duration=3000,
                pre_delay=1000,
                post_delay=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=3000,
                baseline=32,
                target=40,
                rise_rate=100,
                return_speed=100,
                pre_delay=1000,
                post_delay=1000,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                mode="rtp",
                drive_voltage=3,
                duration=3000,
                pre_delay=1000,
                post_delay=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=3000,
                baseline=32,
                target=40,
                rise_rate=100,
                return_speed=100,
                pre_delay=1000,
                post_delay=1000,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                mode="rtp",
                drive_voltage=3,
                duration=3000,
                pre_delay=1000,
                post_delay=1000,
            ),
            TCSStimulus(
                surface=0,
                duration=3000,
                baseline=32,
                target=20,
                rise_rate=100,
                return_speed=100,
                pre_delay=1000,
                post_delay=1000,
            ),
        ]
    ),
]


# Exact block structure from drv2605_tcs.py.
blocks = [
    StimulatorBlock(
        trials=trials,
        trial_repetitions=10,
    ),
]


experiment = StimulatorRuntime(
    name="drv2605_tcs_ina_test",
    sources=sources,
    sinks=sinks,
    blocks=blocks,
)


with experiment:
    experiment.run()
