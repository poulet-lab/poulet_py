from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.config.logging import LOGGER, setup_logging
    from poulet_py.config.settings import SETTINGS, Settings

IMPORTS = {
    "LOGGER": "config.logging",
    "setup_logging": "config.logging",
    "SETTINGS": "config.settings",
    "Settings": "config.settings",
}


def __getattr__(name: str) -> Any:
    if name in IMPORTS:
        module_path = IMPORTS[name]
        module = __import__(f"poulet_py.{module_path}", fromlist=[name])
        return getattr(module, name)
    msg = f"module 'poulet_py.config' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["LOGGER", "SETTINGS", "Settings", "setup_logging"]
