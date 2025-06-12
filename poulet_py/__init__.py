from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py import hardware, tools
    from poulet_py.config import LOGGER, SETTINGS
    from poulet_py.converters.seq import Seq
    from poulet_py.tools.generators import generate_stimulus_sequence
    from poulet_py.tools.organizational import (
        check_or_create,
        define_folder_name,
        sanitize_path,
    )
    from poulet_py.tools.serializers import json_serializer, save_metadata_exp


IMPORTS = {
    "LOGGER": "config.logging",
    "SETTINGS": "config.settings",
    "Seq": "converters.seq",
    "generate_stimulus_sequence": "tools.generators",
    "check_or_create": "tools.organizational",
    "define_folder_name": "tools.organizational",
    "sanitize_path": "tools.organizational",
    "json_serializer": "tools.serializers",
    "save_metadata_exp": "tools.serializers",
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
    "Seq",
    "check_or_create",
    "define_folder_name",
    "generate_stimulus_sequence",
    "hardware",
    "json_serializer",
    "sanitize_path",
    "save_metadata_exp",
    "tools",
]
