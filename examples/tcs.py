from poulet_py import (
    CounterSource,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    TCSSource,
    TCSStimulus,
)

sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    TCSSource(name="tcs", port="/dev/ttyUSB0", maximum_temperature=50),
]
sinks = [HDFSink(file="./temp.h5")]

trials = [
    StimulatorTrial(
        stimuli=TCSStimulus(
            surface=0,
            baseline=32,
            target=20,
            rise_rate=10,
            return_speed=10,
            duration=3000,
        )
    ),
    StimulatorTrial(
        stimuli=TCSStimulus(
            surface=0,
            baseline=32,
            target=25,
            rise_rate=10,
            return_speed=10,
            duration=3000,
        )
    ),
]
blocks = [StimulatorBlock(trials=trials, trial_repetitions=2)]

exp = StimulatorRuntime(name="Test TCS", sources=sources, sinks=sinks, blocks=blocks)

with exp:
    exp.run()
