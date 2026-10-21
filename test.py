from collections.abc import Sequence

from poulet_py import (
    CounterSource,
    EmptyStimulus,
    HDFSink,
    ResponceMonitor,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
)
from poulet_py.io.eventing.events import BaseEvent, SinkEvent

stimulus = EmptyStimulus(duration=3000)
trial = StimulatorTrial(stimuli=stimulus)
block = StimulatorBlock(trials=[trial], trial_repetitions=20)

sources = [CounterSource(name="counter")]
sinks = [HDFSink(file="test.h5", convert_timestamp="datetime")]

_counter = 0


def on_event(x: BaseEvent) -> bool:
    global _counter
    if isinstance(x, SinkEvent) and x.name == "counter":
        _counter = int(x.payload["counter"][-1])
        if (x.payload["counter"] > 10).any():
            return True
    return False


def before_trial(trial: StimulatorTrial) -> StimulatorTrial | None:
    global _counter
    if not isinstance(trial.stimuli, Sequence):
        trial.stimuli = [trial.stimuli]

    for stim in trial.stimuli:
        stim.duration += _counter

    return trial


monitor = ResponceMonitor(on_event_callable=on_event, before_trial_callable=before_trial)

runtime = StimulatorRuntime(
    name="test", blocks=[block], sources=sources, sinks=sinks, responce_monitor=monitor
)

with runtime:
    runtime.run()
