"""StimulatorRuntime open-failure cleanup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeIO:
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.opened = False
        self.closed = False
        self.bus = None

    def open(self):
        if self.fail:
            raise RuntimeError(f"{self.name} open failed")
        self.opened = True

    def close(self):
        self.closed = True


def test_open_failure_closes_already_opened_sources(monkeypatch):
    from poulet_py.utils.stimulator import StimulatorRuntime

    sources = [
        FakeIO("s0"),
        FakeIO("s1"),
        FakeIO("s2", fail=True),
        FakeIO("s3"),
    ]
    sinks = [FakeIO("sink0")]

    runtime = StimulatorRuntime.model_construct(
        name="test",
        sources=sources,
        sinks=sinks,
        blocks=[],
        isi=0,
    )
    runtime._external_bus = True
    runtime._is_open = False
    runtime.bus = SimpleNamespace(open=lambda: None, close=lambda: None)
    runtime._create_key_bindings = lambda: {}
    runtime._started = SimpleNamespace(clear=lambda: None)
    runtime._paused = SimpleNamespace(clear=lambda: None)
    runtime._aborted = SimpleNamespace(clear=lambda: None)
    runtime._stopped = SimpleNamespace(clear=lambda: None)

    with pytest.raises(RuntimeError, match="s2 open failed"):
        runtime.open()

    assert sources[0].opened and sources[0].closed
    assert sources[1].opened and sources[1].closed
    assert not sources[2].opened
    assert not sources[3].opened
    assert not sinks[0].opened
    assert runtime._is_open is False
