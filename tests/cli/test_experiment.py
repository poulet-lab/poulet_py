from pathlib import Path

import pytest

from poulet_py.cli import experiment


@pytest.fixture
def template_builder():
    def build_template(root: Path) -> Path:
        template_root = root / "experiment-template-main"
        template_root.mkdir()
        (template_root / ".git").mkdir()
        (template_root / "README.md").write_text("experiment-template", encoding="utf-8")
        (template_root / "data.txt").write_text("content", encoding="utf-8")
        nested = template_root / "nested"
        nested.mkdir()
        (nested / "file.txt").write_text("nested", encoding="utf-8")
        return template_root

    return build_template


def test_create_experiment_from_template(tmp_path, monkeypatch, capsys, template_builder):
    def fake_download(dest: Path, repo: str = "", branch: str = "") -> Path:  # noqa: ARG001
        return template_builder(dest)

    monkeypatch.setattr(experiment, "download_template", fake_download)

    target = tmp_path / "my-exp"
    experiment.create_experiment(str(target))

    assert target.exists()
    assert not (target / ".git").exists()
    assert (target / "data.txt").read_text(encoding="utf-8") == "content"
    assert (target / "nested" / "file.txt").read_text(encoding="utf-8") == "nested"
    assert (target / "README.md").read_text(encoding="utf-8") == "my-exp"

    out = capsys.readouterr().out
    assert "✅ Experiment created" in out
    assert "Next steps:" in out


def test_create_experiment_existing_directory(tmp_path, capsys):
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        experiment.create_experiment(str(target))

    assert excinfo.value.code == 1
    assert "already exists" in capsys.readouterr().err


def test_main_invokes_create(monkeypatch):
    calls: dict[str, str] = {}

    def fake_create(name: str, *, repo: str = "", branch: str = "") -> None:  # noqa: ARG001
        calls["name"] = name

    monkeypatch.setattr(experiment, "create_experiment", fake_create)

    experiment.main(["init", "new-exp"])

    assert calls == {"name": "new-exp"}


def test_main_shows_usage_on_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        experiment.main(["wrong"])

    assert excinfo.value.code == 1
    assert "Usage: build-exp init" in capsys.readouterr().err