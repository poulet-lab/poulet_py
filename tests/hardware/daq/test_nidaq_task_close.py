"""NIBaseTask close must not leave a replacement Task for GC warnings."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_nibaseline_close_sets_task_to_none(monkeypatch):
    from poulet_py.hardware.daq import nidaq as module

    fake_task = MagicMock()
    monkeypatch.setattr(module, "Task", MagicMock(return_value=fake_task))

    class DummyTask(module.NIBaseTask):
        _requires_clock = False

        def _open(self) -> None:
            return None

    task = DummyTask.model_construct(
        name="ai",
        device="Dev1",
        clock=None,
    )
    task._requires_clock = False
    task._task = fake_task
    task._is_open = True

    task.close()

    fake_task.close.assert_called_once()
    assert task._task is None
    assert task._is_open is False

    # Re-open allocates a fresh Task instead of reusing a leaked empty one.
    task.open()
    assert task._task is fake_task
    assert task._is_open is True
