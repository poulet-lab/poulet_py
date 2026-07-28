from adafruit_ina228 import AveragingCount, ConversionTime, Mode
from board import SCL, SDA
from busio import I2C

from poulet_py import (
    AcquisitionType,
    CounterSource,
    DCAMSource,
    DRV2605Source,
    DRV2605Stimulus,
    HDFSink,
    INA228Source,
    NIAnalogInputChannel,
    NIAnalogInputTask,
    NIClockTask,
    NIDaQSource,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    TCSSource,
    TCSStimulus,
)
from poulet_py.hardware.camera.hamamatzu.dcam import DCAMPROP

# DURATION
ITI = 15000
STIMULUS_DURATION = 3000
STIMULUS_DURATION_TOUCH = 3000
PRE_STIMULUS_DURATION = round(ITI / 2)
POST_STIMULUS_DURATION = round(ITI / 2)
TRIAL_DURATION = PRE_STIMULUS_DURATION + STIMULUS_DURATION + POST_STIMULUS_DURATION
RAMP_UP = 300
RAMP_DOWN = 300
BASELINE = 32
DRIVE_VOLTAGE = 3
STIMULUS_TEMPS = [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]
REPITITIONS = 6
# NI-DAQ configuration: continuously sample the first four analog inputs
# (physical channels Dev1/ai0 through Dev1/ai3) at 1000 samples/s/channel.
NIDAQ_DEVICE = "Dev1"
NIDAQ_RATE_HZ = 1000
NIDAQ_BUFFER_SAMPLES_PER_CHANNEL = TRIAL_DURATION * NIDAQ_RATE_HZ * 1.5


nidaq_clock = NIClockTask(
    name="nidaq_clock",
    device=NIDAQ_DEVICE,
    line=0,
    rate=NIDAQ_RATE_HZ,
    samps_per_chan=NIDAQ_BUFFER_SAMPLES_PER_CHANNEL,
    acquisition_type=AcquisitionType.CONTINUOUS,
)

nidaq_analog_input = NIAnalogInputTask(
    name="analog_input",
    device=NIDAQ_DEVICE,
    clock=nidaq_clock.clock,
    channels=[
        NIAnalogInputChannel(name="Touch_stim", number=0, min_val=-10, max_val=10),
        NIAnalogInputChannel(name="Pad", number=5, min_val=-10, max_val=10),
        NIAnalogInputChannel(name="Mouse", number=6, min_val=-10, max_val=10),
        NIAnalogInputChannel(name="BR_HR_Monitor", number=7, min_val=-10, max_val=10),
    ],
)

# create common i2c bus for all sources
i2c = I2C(SCL, SDA, frequency=400_000)
# this is a temp fix to test weather pyftdi error handling introduces erroneous values...
# instead move to poulet_py error handling and read discard
# maximum of retries internally through pyftdi, eliminates warning: retry exchange handling and
# limits erroneous data writing into buffer, doesnt necesseraliy need to be implemented anymore
i2c._i2c._i2c.set_retry_count(1)
sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    NIDaQSource(
        name="nidaq",
        device=NIDAQ_DEVICE,
        tasks=[nidaq_clock, nidaq_analog_input],
        acquisition_type=AcquisitionType.CONTINUOUS,
        buffer_size=NIDAQ_BUFFER_SAMPLES_PER_CHANNEL,
    ),
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
        address=0x40,
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
        address=0x41,
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
        file="./all_sources_test_short.h5",
        queue_size=1_000,
        grow_step=100,
        compression="lzf",
    ),
]


# Exact stimulus and trial structure from drv2605_tcs.py.
# Only the DRV2605L usage mode has been changed to RTP.
base_trials = [
    StimulatorTrial(
        stimuli=[
            TCSStimulus(
                surface=0,
                duration=STIMULUS_DURATION,
                baseline=BASELINE,
                target=BASELINE,
                rise_rate=RAMP_UP,
                return_speed=RAMP_DOWN,
                pre_delay=PRE_STIMULUS_DURATION,
                post_delay=POST_STIMULUS_DURATION,
            ),
        ]
    ),
    StimulatorTrial(
        stimuli=[
            DRV2605Stimulus(
                mode="rtp",
                drive_voltage=DRIVE_VOLTAGE,
                duration=STIMULUS_DURATION_TOUCH,
                pre_delay=PRE_STIMULUS_DURATION,
                post_delay=POST_STIMULUS_DURATION,
            ),
            TCSStimulus(
                surface=0,
                duration=STIMULUS_DURATION,
                baseline=BASELINE,
                target=BASELINE,
                rise_rate=RAMP_UP,
                return_speed=RAMP_DOWN,
                pre_delay=PRE_STIMULUS_DURATION,
                post_delay=POST_STIMULUS_DURATION,
            ),
        ]
    ),
]
user_trials = []
for temps in STIMULUS_TEMPS:
    single_trial = StimulatorTrial(
        stimuli=TCSStimulus(
            surface=0,
            duration=STIMULUS_DURATION,
            baseline=BASELINE,
            target=BASELINE + temps,
            rise_rate=RAMP_UP,
            return_speed=RAMP_DOWN,
            pre_delay=PRE_STIMULUS_DURATION,
            post_delay=POST_STIMULUS_DURATION,
        ),
    )
    user_trials = user_trials.append(single_trial)

experiment_trials = base_trials.extend(user_trials)
# Exact block structure from drv2605_tcs.py.
blocks = [
    StimulatorBlock(
        trials=experiment_trials,
        trial_repetitions=REPITITIONS,
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
