from poulet_py import (
    CounterSource,
    DRV2605Source,
    DRV2605Stimulus,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
)

sources = [
    CounterSource(name="trial"),
    StimuliMetadataSource(name="metadata"),
    DRV2605Source(
        name="drv2605",
        address=0x5A,
        bus_frequency=400_000,
        ftdi_latency_ms=1,
    ),
]

sinks = [
    HDFSink(file="./temp_drv2605.h5"),
]

trials = [
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=1,
            duration=1000,
            drive_voltage=3.0,
        )
    ),
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=3,
            duration=3000,
            drive_voltage=4.0,
        )
    ),
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=7,
            duration=7000,
            drive_voltage=4.5,
        )
    ),
]

blocks = [
    StimulatorBlock(trials=trials),
]

experiment = StimulatorRuntime(
    sources=sources,
    sinks=sinks,
    blocks=blocks,
)

with experiment:
    experiment.run()
