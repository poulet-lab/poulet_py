from poulet_py import (
    DCAM,
    CounterSource,
    DCAMSource,
    EmptyStimulus,
    HDFSink,
    StimulatorBlock,
    StimulatorRuntime,
    StimulatorTrial,
    TCSSource,
)

import cv2
import numpy as np


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
        show_framedata(sample["dcam"])


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()

sources = [DCAMSource()]