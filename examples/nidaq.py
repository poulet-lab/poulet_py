from poulet_py import (
    AcquisitionType,
    CounterSource,
    HDFSink,
    NIAnalogOutputChannel,
    NIAnalogOutputTask,
    NIClockTask,
    NIDaQSource,
    NISineAnalogStimulus,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
)

RATE_HZ = 1000
DUR_S = 20

ao_channels = [NIAnalogOutputChannel(name="fp-touch", number=0, min_val=-1, max_val=1)]

tasks = [
    NIClockTask(
        name="clock",
        line=0,
        rate=RATE_HZ,
        samps_per_chan=RATE_HZ * DUR_S,
        acquisition_type=AcquisitionType.FINITE,
    ),
    NIAnalogOutputTask(name="ao", channels=ao_channels),
]

sources = [
    CounterSource(name="trial_source"),
    NIDaQSource(name="nidaq", device="dev1", tasks=tasks),
]
sinks = [HDFSink(file="./temp.h5")]

stimulus = NISineAnalogStimulus(duration=5000, frequency=60, amplitude=1)

trials = [StimulatorTrial(stimuli=stimulus)]
blocks = [StimulatorBlock(trials=trials, trial_repetitions=5)]

exp = StimulatorRuntime(name="Test NI", sources=sources, sinks=sinks, blocks=blocks)

with exp:
    exp.run()
