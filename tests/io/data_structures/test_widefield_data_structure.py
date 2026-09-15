from pathlib import Path

import h5py

from poulet_py.io.data_structures.widefield import (
    WidefieldData,
    WidefieldMetadata,
)


def _create_trial_folder(tmp_path: Path) -> Path:
    trial_path = tmp_path / "trial"
    trial_path.mkdir()
    (trial_path / "recording.tiff").touch()
    (trial_path / "recording.csv").touch()
    (trial_path / "green.tiff").touch()
    return trial_path


def _write_complete_metadata(trial_path: Path) -> None:
    with h5py.File(trial_path / "data.h5", "w") as hdf:
        hdf.attrs["camera_format"] = "fast mode"
        hdf.attrs["camera_fps"] = 30
        hdf.attrs["camera_exposure"] = 12.5
        hdf.attrs["camera_roi_active"] = 1
        hdf.attrs["camera_roi"] = [10, 20, 512, 512]
        hdf.attrs["binning"] = 2
        hdf.attrs["magnification"] = 1.6
        hdf.attrs["filterset"] = "GFP"
        hdf.attrs["led_power"] = 4.5
        hdf.attrs["mouse_id"] = "JPCM-09695"
        hdf.attrs["weight"] = 24.2
        hdf.attrs["anesthesia"] = "isoflurane"
        hdf.attrs["isoflurane"] = 1.5
        hdf.attrs["protocol_name"] = "temperature_short"
        hdf.attrs["time"] = "260504_115040"
        hdf.attrs["timestamp"] = 1777895440.0
        hdf.attrs["experimenter"] = "experimenter"
        hdf.attrs["comment"] = "baseline"
        hdf.attrs["folder"] = "trial"
        hdf.attrs["unexpected"] = "not part of the v1 schema"
        channel = hdf.create_dataset("data/in/temperature", data=[1.0])
        channel.attrs["id"] = "Dev1/ai0"
        channel.attrs["name"] = "temperature in"
        channel.attrs["sr"] = 1000
        channel.attrs["device"] = "TCS"
        channel.attrs["unexpected"] = "ignored"


def _assert_complete_metadata(metadata: WidefieldMetadata) -> None:
    assert metadata.camera.format == "fast mode"
    assert metadata.camera.fps == 30
    assert metadata.camera.exposure == 12.5
    assert metadata.camera.roi_active is True
    assert metadata.camera.roi == (10, 20, 512, 512)
    assert metadata.camera.binning == 2
    assert metadata.camera.magnification == 1.6
    assert metadata.camera.filterset == "GFP"
    assert metadata.camera.led_power == 4.5
    assert metadata.subject.mouse_id == "JPCM-09695"
    assert metadata.subject.weight == 24.2
    assert metadata.subject.anesthesia == "isoflurane"
    assert metadata.subject.isoflurane == 1.5
    assert metadata.acquisition.protocol_name == "temperature_short"
    assert metadata.acquisition.time == "260504_115040"
    assert metadata.acquisition.timestamp == 1777895440.0
    assert metadata.acquisition.experimenter == "experimenter"
    assert metadata.acquisition.comment == "baseline"
    assert metadata.acquisition.folder == "trial"
    assert not hasattr(metadata, "unexpected")

    channel_metadata = metadata.analog_output["data/in/temperature"]
    assert channel_metadata.id == "Dev1/ai0"
    assert channel_metadata.name == "temperature in"
    assert channel_metadata.sr == 1000
    assert channel_metadata.device == "TCS"
    assert not hasattr(channel_metadata, "unexpected")


def test_v1_metadata_is_loaded_into_explicit_categories(
    tmp_path: Path,
) -> None:
    trial_path = _create_trial_folder(tmp_path)
    _write_complete_metadata(trial_path)

    metadata = WidefieldData(path=trial_path).metadata

    _assert_complete_metadata(metadata)


def test_missing_string_metadata_defaults_to_empty_strings(
    tmp_path: Path,
) -> None:
    trial_path = _create_trial_folder(tmp_path)
    with h5py.File(trial_path / "data.h5", "w"):
        pass

    metadata = WidefieldData(path=trial_path).metadata

    assert metadata.camera.format == ""
    assert metadata.camera.filterset == ""
    assert metadata.subject.mouse_id == ""
    assert metadata.subject.anesthesia == ""
    assert metadata.acquisition.protocol_name == ""
    assert metadata.acquisition.time == ""
    assert metadata.acquisition.experimenter == ""
    assert metadata.acquisition.comment == ""
    assert metadata.acquisition.folder == ""
