from poulet_py import (
    CounterSource,
    DRV2605Source,
    DRV2605Stimulus,
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
    DRV2605Source(
        name="drv2605",
        address=0x5A,
        bus_frequency=400_000,
        ftdi_latency_ms=1,
    ),
    TCSSource(name="tcs", port="/dev/ttyUSB0", maximum_temperature=50),
]

sinks = [
    HDFSink(file="./temp_drv2605_tcs.h5"),
]

trials = [
    StimulatorTrial(
        stimuli=DRV2605Stimulus(
            waveform=16,
            repeat_count=1,
            duration=1000,
            drive_voltage=5.0,
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
            drive_voltage=2.5,
        )
    ),
    StimulatorTrial(
<<<<<<< HEAD
        stimuli=TCSStimulus(
            surface=0,
            baseline=32,
            target=20,
            rise_rate=10,
            return_speed=10,
            duration=3000,
=======
    stimuli=TCSStimulus(
        surface=0,
        baseline=32,
        target=20,
        rise_rate=10,
        return_speed=10,
        duration=3000,
>>>>>>> ff6c678 (Refactor TCSStimulus indentation for consistency)
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

blocks = [
    StimulatorBlock(trials=trials, trial_repetitions=2),
]

experiment = StimulatorRuntime(
    name="drv2605_tcs_test",
    sources=sources,
    sinks=sinks,
    blocks=blocks,
)

with experiment:
    experiment.run()
