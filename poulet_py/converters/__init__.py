from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.converters.seq import Seq

IMPORTS = {"Seq": "converters.seq"}


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

    msg = f"module 'poulet_py.converters' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["Seq"]
