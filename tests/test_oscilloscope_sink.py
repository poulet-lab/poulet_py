import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("numpydantic")

from poulet_py import Oscilloscope, OscilloscopeSink, SinkEvent


@pytest.fixture
def scope_dict():
    return {
        "voltages": Oscilloscope(
            title="voltages",
            max_samples=20,
            max_points=10,
            animation_interval=100,
        ),
        "thermal": Oscilloscope(
            title="thermal",
            max_samples=20,
            max_points=10,
            animation_interval=100,
        ),
    }


def test_extract_ina228_and_decimate(scope_dict):
    sink = OscilloscopeSink(scopes=scope_dict, decimate_hz=10.0, enable_camera_preview=False)
    sink._open()

    dtype = np.dtype([("timestamp", "u8"), ("bus_voltage", "f8")])
    row = np.array([(1_000_000_000, 3.3)], dtype=dtype)
    chunk = np.array([row, row], dtype=dtype)

    event = SinkEvent(name="ina228_pad", payload=chunk)
    sink._on_event(event)
    sink._on_event(event)

    scope = scope_dict["voltages"]
    assert len(scope._y) == 1
    assert scope._y[0]["ina228_pad"] == pytest.approx(3.3)
    sink._close()


def test_dcam_frame_stored_without_oscilloscope_push(scope_dict):
    sink = OscilloscopeSink(scopes=scope_dict, enable_camera_preview=False)
    sink._open()

    frame = np.zeros((4, 4), dtype=np.uint16)
    dtype = np.dtype([("timestamp", "u8"), ("dcam", "O")])
    row = np.array([(2_000_000_000, frame)], dtype=dtype)
    chunk = np.array([row], dtype=dtype)

    sink._on_event(SinkEvent(name="dcam", payload=chunk))
    with sink._frame_lock:
        assert sink._latest_frame is not None
    assert len(scope_dict["voltages"]._y) == 0
    sink._close()
