from nidaqmx.constants import AcquisitionType
from numpy import empty

from poulet_py import (
    CounterSource,
    HDFDataSink,
    NIAnalogInputChannel,
    NIAnalogInputTask,
    NIAnalogOutputChannel,
    NIAnalogOutputTask,
    NIClockTask,
    NIDaQ,
    NIDigitalOutputChannel,
    NIDigitalOutputTask,
)

rate = 1000
samps_per_chan = rate * 20

ai_channels = [
    NIAnalogInputChannel(name="0", number=0, max_val=10, min_val=-10),
    NIAnalogInputChannel(name="1", number=1, max_val=10, min_val=-10),
    NIAnalogInputChannel(name="2", number=2, max_val=10, min_val=-10),
]

ao_channels = [NIAnalogOutputChannel(name="fp-touch", number=0, max_val=1, min_val=-1)]

do_channels = [
    NIDigitalOutputChannel(name="0", port=0, line=0),
    NIDigitalOutputChannel(name="1", port=0, line=1),
    NIDigitalOutputChannel(name="2", port=0, line=2),
]

clk_task = NIClockTask(
    device="dev1",
    name="clock",
    line=0,
    rate=rate,
    samps_per_chan=samps_per_chan,
    sample_mode=AcquisitionType.FINITE,
)

ai_task = NIAnalogInputTask(device="dev1", name="ai", channels=ai_channels)
ao_task = NIAnalogOutputTask(device="dev1", name="ao", channels=ao_channels)
do_task = NIDigitalOutputTask(device="dev1", name="do", channels=do_channels)

nidac = NIDaQ(device="dev1")

nidac.add_task(clk_task)
nidac.add_task(ai_task)
nidac.add_task(ao_task)
nidac.add_task(do_task)

ai_data = empty((len(ai_channels), samps_per_chan))

writer = HDFDataSink("./temp")
qds = QueueDataSink(writer)
trial_source = CounterSource()
trial_source.attach(qds)

with nidac:
    trial_source.next_trial()
    ai_task.read()
