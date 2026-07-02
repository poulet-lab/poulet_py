from time import sleep, time_ns

import cv2
import numpy as np

from poulet_py import (
    DCAM,
    AcquisitionType,
    CounterSource,
    DCAMSource,
    EmptyStimulus,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    StimuliMetadataSource,
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

with DCAM(device_index=0, acquisition_type=AcquisitionType.CONTINUOUS) as dcam:
    while True:
        sample = dcam.read_sample()
        if sample is not None:
            show_framedata(sample["dcam"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()

sleep(1)
sources = [
    CounterSource(name="counter"),
    StimuliMetadataSource(name="meta"),
    DCAMSource(name="dcam", device_index=0, acquisition_type=AcquisitionType.FINITE),
    # TCSSource(name="tcs", port="/dev/ttyUSB0"),
]
sinks = [HDFSink(name="h5sink", file=f"widefield_{time_ns()}.h5", meta={"timestamp": time_ns()})]

empty_stim = EmptyStimulus(duration=3000, pre_delay=100, post_delay=300)
# tcs_stim = TCSStimulus(duration=5000, pre_delay=100, post_delay=300)
blocks = [
    StimulatorBlock(
        trials=[
            StimulatorTrial(stimuli=empty_stim),
            # StimulatorTrial(stimuli=tcs_stim),
        ],
        trial_repetitions=5,
    )
]
exp = StimulatorRuntime(name="widefield", sources=sources, sinks=sinks, blocks=blocks)


with exp:
    exp.run()
