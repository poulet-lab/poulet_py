"""Lifecycle tests for ThermalCamera without real UVC hardware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakePointer:
    def __init__(self, value=None):
        self.value = value


def test_start_streaming_raises_instead_of_exiting(monkeypatch):
    import poulet_py.hardware.camera.thermal_camera as module

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    calls = []

    class FakeLib:
        def uvc_init(self, *_args):
            return -1

        def uvc_exit(self, *_args):
            calls.append("exit")

        def uvc_stop_streaming(self, *_args):
            calls.append("stop")

        def uvc_close(self, *_args):
            calls.append("close")

        def uvc_unref_device(self, *_args):
            calls.append("unref")

    monkeypatch.setattr(module, "libuvc", FakeLib())
    monkeypatch.setattr(
        module,
        "POINTER",
        lambda *_args, **_kwargs: (lambda: _FakePointer()),
    )
    monkeypatch.setattr(module, "byref", lambda value: value)
    monkeypatch.setattr(module, "uvc_context", object)
    monkeypatch.setattr(module, "uvc_device", object)
    monkeypatch.setattr(module, "uvc_device_handle", object)
    monkeypatch.setattr(module, "uvc_stream_ctrl", lambda: SimpleNamespace())

    camera = module.ThermalCamera(vminT=18, vmaxT=42, emissivity=0.95)
    with pytest.raises(RuntimeError, match="uvc_init"):
        camera.start_streaming()
    assert camera._streaming is False


def test_stop_streaming_releases_handles_in_order(monkeypatch):
    import poulet_py.hardware.camera.thermal_camera as module

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    order = []

    class FakeLib:
        def uvc_stop_streaming(self, handle):
            order.append(("stop", handle))

        def uvc_close(self, handle):
            order.append(("close", handle))

        def uvc_unref_device(self, handle):
            order.append(("unref", handle))

        def uvc_exit(self, handle):
            order.append(("exit", handle))

    monkeypatch.setattr(module, "libuvc", FakeLib())
    camera = module.ThermalCamera(vminT=18, vmaxT=42)
    camera.windows = False
    camera._streaming = True
    camera._devh = "devh"
    camera._dev = "dev"
    camera._ctx = "ctx"

    camera.stop_streaming()

    assert order == [
        ("stop", "devh"),
        ("close", "devh"),
        ("unref", "dev"),
        ("exit", "ctx"),
    ]
    assert camera._streaming is False
    assert camera._devh is None
    assert camera._dev is None
    assert camera._ctx is None


def test_constructor_rejects_frame_rate_kwarg():
    from poulet_py.hardware.camera.thermal_camera import ThermalCamera

    with pytest.raises(TypeError):
        ThermalCamera(frame_rate_fps=8.0)


def test_tcam_close_releases_camera_when_not_marked_open():
    from poulet_py.io.sources.tcam import TCAMSource

    source = TCAMSource(name="tcam", vminT=18, vmaxT=42)
    camera = MagicMock()
    source._camera = camera
    source._is_open = False

    source.close()

    camera.stop_streaming.assert_called_once()
    assert source._camera is None
