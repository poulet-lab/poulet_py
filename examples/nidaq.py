from nidaqmx.constants import AcquisitionType
from numpy import empty

from poulet_py import (
    CounterSource,
    Experiment,
    HDFSink,
    NIAnalogInputChannel,
    NIAnalogInputTask,
    NIAnalogOutputChannel,
    NIAnalogOutputTask,
    NIClockTask,
    NIDaQ,
    NIDigitalOutputChannel,
    NIDigitalOutputTask,
    TCSSource,
    TCSStimulus,
)

rate = 1000
duration_s = 20
samps_per_chan = rate * duration_s
device_name = "dev1"


ai_channels = [
    NIAnalogInputChannel(name=str(i), number=i, min_val=-10, max_val=10) for i in range(3)
]
ao_channels = [NIAnalogOutputChannel(name="fp-touch", number=0, min_val=-1, max_val=1)]
do_channels = [NIDigitalOutputChannel(name=str(i), port=0, line=i) for i in range(3)]

clk_task = NIClockTask(
    device=device_name,
    name="clock",
    line=0,
    rate=rate,
    samps_per_chan=samps_per_chan,
    sample_mode=AcquisitionType.FINITE,
)

tasks = {
    "ai": NIAnalogInputTask(device=device_name, name="ai", channels=ai_channels),
    "ao": NIAnalogOutputTask(device=device_name, name="ao", channels=ao_channels),
    "do": NIDigitalOutputTask(device=device_name, name="do", channels=do_channels),
}

nidac = NIDaQ(device=device_name)
nidac.add_task(clk_task)
for task in tasks.values():
    nidac.add_task(task)

ai_data = empty((len(ai_channels), samps_per_chan))

sinks = [HDFSink(file="./temp.h5")]
sources = [CounterSource(name="trial_source"), TCSSource(name="trial_source", port="/dev/ssf")]

stimuli = [
    TCSStimulus(
        surface=0,
        baseline=32,
        target=20,
        rise_rate=10,
        return_speed=10,
        duration=3000,
    ),
    TCSStimulus(
        surface=0,
        baseline=32,
        target=25,
        rise_rate=10,
        return_speed=10,
        duration=3000,
    ),
]

exp = Experiment(sources=sources, sinks=sinks, stimuli=stimuli)

with exp:
    exp.run()
