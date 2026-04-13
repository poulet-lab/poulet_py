from datetime import datetime

import numpy as np
import pytest

from poulet_py.utils.widefield import Session, Trial, WidefieldAnalysis


def _build_analysis(trial_dir):
    now = datetime.now()
    session = Session(path=trial_dir.parent, start=now, end=now)
    wf = WidefieldAnalysis(session=session)
    wf.session.add_trial(Trial(path=trial_dir))
    return wf


def test_validate_trial_resolves_required_and_optional_files(monkeypatch, tmp_path):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()
    (trial_dir / "recording.tiff").touch()

    def fake_load_imaging(self):
        return np.ones((1, 2, 2), dtype=np.uint16)

    monkeypatch.setattr("poulet_py.utils.widefield.Trial._load_imaging", fake_load_imaging)

    wf = _build_analysis(trial_dir)
    wf.load()

    assert wf.active_trial.imaging_path == trial_dir / "recording.tiff"
    assert wf.active_trial.timestamps_path is None
    assert wf.active_trial.analog_output_data_path is None
    assert wf.active_trial.reference_image_path is None


def test_validate_trial_requires_video_file(tmp_path):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()

    wf = _build_analysis(trial_dir)
    with pytest.raises(ValueError, match="recording.tiff/.tif/.npy"):
        wf.load()


def test_load_session_handles_missing_optional_files(monkeypatch, tmp_path):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()
    (trial_dir / "recording.tiff").touch()

    def fake_load_imaging(self):
        return np.ones((4, 8, 8), dtype=np.uint16)

    monkeypatch.setattr("poulet_py.utils.widefield.Trial._load_imaging", fake_load_imaging)
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_reference_image",
        lambda self: None,
    )
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_timestamps",
        lambda self: None,
    )
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_analog_output",
        lambda self: ({}, {}, {}),
    )

    wf = _build_analysis(trial_dir)
    wf.load()

    assert wf.active_trial.imaging_data is not None
    assert wf.active_trial.imaging_data.shape == (4, 8, 8)
    assert wf.active_trial.timestamps is None
    assert wf.active_trial.analog_output_data == {}


def test_close_releases_loaded_trial_resources(monkeypatch, tmp_path):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()
    (trial_dir / "recording.tiff").touch()

    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_imaging",
        lambda self: np.ones((2, 4, 4), dtype=np.uint16),
    )
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_reference_image",
        lambda self: np.zeros((4, 4), dtype=np.uint16),
    )
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_timestamps",
        lambda self: None,
    )
    monkeypatch.setattr(
        "poulet_py.utils.widefield.Trial._load_analog_output",
        lambda self: ({}, {}, {}),
    )

    wf = _build_analysis(trial_dir)
    wf.load()
    assert wf.active_trial.imaging_data is not None

    wf.session.close()

    assert wf.active_trial.imaging_data is None
    assert wf.active_trial.green_reference is None
    assert wf.active_trial.timestamps is None
    assert wf.active_trial.analog_output_data == {}
