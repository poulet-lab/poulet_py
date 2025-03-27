from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poulet_py.hardware.camera.basler import BaslerCamera
    from poulet_py.hardware.camera.group_gets import PureThermal
    from poulet_py.hardware.controller.arduino import Arduino
    from poulet_py.hardware.thermal import (
        TCSIIController,
        TCSIIStimulus,
        TCSStimulus,
    )
    from poulet_py.hardware.thermal.julabo_chiller import JulaboChiller


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
            msg = f"Missing '{submodule}' module. "
            "To install it use: pip install poulet_py[{submodule}]"
            raise ImportError(msg) from e

    msg = f"module 'poulet_py.hardware' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [
    "Arduino",
    "BaslerCamera",
    "JulaboChiller",
    "PureThermal",
    "TCSIIController",
    "TCSIIStimulus",
    "TCSStimulus",
]
