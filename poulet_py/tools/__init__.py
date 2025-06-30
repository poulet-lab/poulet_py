from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.tools.generators import generate_stimulus_sequence
    from poulet_py.tools.organizational import (
        check_or_create,
        define_folder_name,
        go_to,
        sanitize_path,
    )
    from poulet_py.tools.serializers import json_serializer, save_metadata_exp

IMPORTS = {
    "generate_stimulus_sequence": "tools.generators",
    "check_or_create": "tools.organizational",
    "define_folder_name": "tools.organizational",
    "go_to": "tools.organizational",
    "sanitize_path": "tools.organizational",
    "json_serializer": "tools.serializers",
    "save_metadata_exp": "tools.serializers",
}


def __getattr__(name: str) -> Any:
    if name in IMPORTS:
        module_path = IMPORTS[name]
        try:
            module = __import__(f"poulet_py.{module_path}", fromlist=[name])
            return getattr(module, name)
        except ImportError as e:
            # Extract the submodule name (e.g., "utils" or "tools")
            submodule = module_path.split(".")[-2]
            msg = f"❌ Missing '{submodule}' module. Import failed: "
            "Install it with: pip install poulet_py[{submodule}]"
            raise ImportError(msg) from e

    msg = f"module 'poulet_py.tools' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [
    "check_or_create",
    "define_folder_name",
    "generate_stimulus_sequence",
    "go_to",
    "json_serializer",
    "sanitize_path",
    "save_metadata_exp",
]
