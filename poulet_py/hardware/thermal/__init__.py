from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.hardware.thermal.qst import (
        TCSIIController,
        TCSIIStimulus,
        TCSStimulus,
    )

IMPORTS = {
    "TCSStimulus": "hardware.thermal_stimulators.qst",
    "TCSIIStimulus": "hardware.thermal_stimulators.qst",
    "TCSIIController": "hardware.thermal_stimulators.qst",
}


def __getattr__(name: str) -> Any:
    if name in IMPORTS:
        module_path = IMPORTS[name]
        try:
            module = __import__(f"poulet_py.{module_path}", fromlist=[name])
            return getattr(module, name)
        except ImportError as e:
            submodule = module_path.split(".")[-1]
            msg = "Missing '{submodule}' module. "
            f"To install it use: pip install poulet_py[{submodule}]"
            raise ImportError(msg) from e

    msg = "module 'poulet_py.hardware.thermal_stimulators' "
    f"has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["TCSIIController", "TCSIIStimulus", "TCSStimulus"]
