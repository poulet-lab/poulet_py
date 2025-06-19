from poulet_py import SETTINGS, Settings


def test_settings_defaults():
    # Add all new settings to this test with default values
    assert SETTINGS.log.level == "info"
    assert SETTINGS.log.file is None


def test_settings_env_vars(monkeypatch):
    # add all new settings to this test with env var values
    monkeypatch.setenv("LOG__LEVEL", "debug")
    monkeypatch.setenv("LOG__FILE", "/tmp/poulet_py.log")  # noqa: S108

    settings = Settings()
    assert settings.log.level == "debug"
    assert settings.log.file == "/tmp/poulet_py.log"  # noqa: S108
