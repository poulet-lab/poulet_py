from time import time_ns

import cv2
import numpy as np

from poulet_py import (
    DCAM,
    CounterSource,
    DCAMSource,
    EmptyStimulus,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
    TCSSource,
)


def show_framedata(data):
    maxval = np.amax(data)
    if data.dtype == np.uint16 and maxval > 0:
        imul = int(65535 / maxval)
        data = data * imul

    cv2.imshow("dcam", data)


cv2.namedWindow(
    "dcam",
    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
)

with DCAM(device_index=0) as dcam:
    while True:
        sample = dcam.read_sample()
        if sample is not None:
            show_framedata(sample["dcam"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()

sources = [
    CounterSource(name="counter"),
    StimuliMetadataSource(name="meta"),
    DCAMSource(name="dcam", device_index=0),
    TCSSource(name="tcs", port="/dev"),
]
sinks = [HDFSink(name="h5sink", file=f"widefield_{time_ns}.h5", meta={"timestamp": time_ns()})]

empty_stim = EmptyStimulus(duration=3000, pre_delay=100)

blocks = [
    StimulatorBlock(
        trials=[StimulatorTrial(stimuli=empty_stim)], trial_repetitions=30, isi=range(500, 4000)
    )
]
exp = StimulatorRuntime(name="widefield", sources=sources, sinks=sinks, blocks=blocks)


with exp:
    exp.run()
