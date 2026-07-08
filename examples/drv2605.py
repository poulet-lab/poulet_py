from poulet_py import (
    CounterSource,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    DRV2605Stimulus,    
)

sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
]
sinks = [HDFSink(file="./temp.h5")]

trials = [
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            address=0x5A,
            bus_frequency=400000,
            repeats=5,
            waveform=16,
            random=False,
            custom=False,
            voltage=4.5
        )
    ),

    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            address=0x5A,
            bus_frequency=400000,
            repeats=5,
            waveform=16,
            random=False,
            custom=False,
            voltage=4.5
        )
    ),
]

blocks = [StimulatorBlock(trials=trials, trial_repetitions=5)]

exp = StimulatorRuntime(name="Test DRV2605", sources=sources, sinks=sinks, blocks=blocks)

with exp:
    exp.run()
