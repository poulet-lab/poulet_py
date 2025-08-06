from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.config import LOGGER, SETTINGS, Settings, setup_logging
    from poulet_py.hardware import TCS, TCSCommand, TCSStimulus
    from poulet_py.tools import (
        check_or_create,
        define_folder_name,
        generate_stimulus_sequence,
        go_to,
        json_serializer,
        sanitize_path,
        save_metadata_exp,
    )
    from poulet_py.utils import Oscilloscope, TCSInterface

IMPORTS = {
    "config": "config",
    "hardware": "hardware",
    "tools": "tools",
    "utils": "utils",
    "LOGGER": "config",
    "SETTINGS": "config",
    "TCS": "hardware",
    "TCSCommand": "hardware",
    "TCSStimulus": "hardware",
    "generate_stimulus_sequence": "tools",
    "check_or_create": "tools",
    "define_folder_name": "tools",
    "sanitize_path": "tools",
    "json_serializer": "tools",
    "save_metadata_exp": "tools",
    "Oscilloscope": "utils",
    "TCSInterface": "utils",
}


def __getattr__(name: str) -> Any:
    if name in IMPORTS:
        module_path = IMPORTS[name]
        module = __import__(f"poulet_py.{module_path}", fromlist=[name])
        return getattr(module, name)

    msg = f"module 'poulet_py' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [
    "LOGGER",
    "SETTINGS",
    "TCS",
    "Oscilloscope",
    "Settings",
    "TCSCommand",
    "TCSInterface",
    "TCSStimulus",
    "check_or_create",
    "config",
    "define_folder_name",
    "generate_stimulus_sequence",
    "go_to",
    "hardware",
    "json_serializer",
    "sanitize_path",
    "save_metadata_exp",
    "setup_logging",
    "tools",
    "utils",
]
