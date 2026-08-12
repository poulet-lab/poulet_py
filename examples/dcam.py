"""Widefield run script using direct DCAMSource settings.

This script assumes:
1. The patched drop-in DCAM class is exported as poulet_py.DCAM.
2. The drop-in direct-inheritance DCAMSource is exported as poulet_py.DCAMSource.
3. DCAMSource inherits DCAM directly, so camera settings are passed directly to
   DCAMSource(...), not via dcam_kwargs.
"""

from inspect import getfile
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
from poulet_py.hardware.camera.hamamatzu._api import DCAMPROP

WINDOW_NAME = "dcam"

# Settings verified in the standalone/V5 path:
# - 10 fps via Master Pulse
# - timing output high during global exposure
# - connector 1
DCAM_SETTINGS = dict(
    debug_output=True,
    exposure_time=50,  # ms; increase if global-exposure window is too short
    frame_rate=10.0,  # fps -> MASTERPULSE_INTERVAL = 0.1 s
    timing_mode="masterpulse",
    masterpulse_mode=DCAMPROP.MASTERPULSE_MODE.CONTINUOUS,
    masterpulse_triggersource=DCAMPROP.MASTERPULSE_TRIGGERSOURCE.SOFTWARE,
    trigger_source=DCAMPROP.TRIGGERSOURCE.MASTERPULSE,
    trigger_mode=DCAMPROP.TRIGGER_MODE.NORMAL,
    trigger_active=DCAMPROP.TRIGGERACTIVE.EDGE,
    trigger_polarity=DCAMPROP.TRIGGERPOLARITY.POSITIVE,
    trigger_global_exposure=DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED,
    output_trigger_connector=1,
    output_trigger_kind=DCAMPROP.OUTPUTTRIGGER_KIND.GLOBALEXPOSURE,
    output_trigger_polarity=DCAMPROP.OUTPUTTRIGGER_POLARITY.POSITIVE,
    output_trigger_basesensor=DCAMPROP.OUTPUTTRIGGER_BASESENSOR.VIEW1,
)

REQUIRED_DCAM_FIELDS = {
    "debug_output",
    "exposure_time",
    "frame_rate",
    "timing_mode",
    "masterpulse_mode",
    "masterpulse_triggersource",
    "output_trigger_connector",
    "output_trigger_kind",
    "output_trigger_polarity",
    "output_trigger_basesensor",
    "trigger_global_exposure",
}


def model_field_names(cls) -> set[str]:
    fields = getattr(cls, "model_fields", None)  # pydantic v2
    if fields is None:
        fields = getattr(cls, "__fields__", {})  # pydantic v1
    return set(fields.keys())


def verify_patched_classes() -> None:
    print(f"Using DCAM class from: {getfile(DCAM)}", flush=True)
    print(f"Using DCAMSource class from: {getfile(DCAMSource)}", flush=True)
    print(f"DCAMSource MRO: {DCAMSource.__mro__}", flush=True)

    missing_dcam = REQUIRED_DCAM_FIELDS - model_field_names(DCAM)
    missing_source = REQUIRED_DCAM_FIELDS - model_field_names(DCAMSource)

    if missing_dcam:
        raise RuntimeError(
            "Imported poulet_py.DCAM is not the patched trigger-capable class. "
            f"Missing DCAM fields: {sorted(missing_dcam)}"
        )

    if missing_source:
        raise RuntimeError(
            "Imported poulet_py.DCAMSource is not inheriting the patched DCAM fields. "
            f"Missing DCAMSource fields: {sorted(missing_source)}. "
            "Replace the DCAMSource module with the direct-inheritance drop-in file."
        )


def show_framedata(data):
    maxval = np.amax(data)
    if data.dtype == np.uint16 and maxval > 0:
        imul = int(65535 / maxval)
        data = data * imul

    cv2.imshow(WINDOW_NAME, data)


def should_exit() -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return True

    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run_preview() -> None:
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
    )

    with DCAM(
        device_index=0,
        acquisition_type=AcquisitionType.CONTINUOUS,
        **DCAM_SETTINGS,
    ) as dcam:
        print(
            "Live preview running with 10 fps Master Pulse + GLOBALEXPOSURE output. "
            "Press q in the OpenCV window to stop preview and continue to experiment.",
            flush=True,
        )

        while True:
            sample = dcam.read_sample()
            if sample is not None:
                show_framedata(sample["dcam"])

            if should_exit():
                break

            sleep(0.001)

    cv2.destroyAllWindows()
    sleep(1)


def run_experiment() -> None:
    sources = [
        CounterSource(name="counter"),
        StimuliMetadataSource(name="meta"),
        DCAMSource(
            name="dcam",
            device_index=0,
            acquisition_type=AcquisitionType.FINITE,
            **DCAM_SETTINGS,
        ),
        # TCSSource(name="tcs", port="/dev/ttyUSB0"),
    ]

    sinks = [
        HDFSink(
            name="h5sink",
            file=f"widefield_{time_ns()}.h5",
            meta={"timestamp": time_ns()},
        )
    ]

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


if __name__ == "__main__":
    verify_patched_classes()
    run_preview()
    run_experiment()
