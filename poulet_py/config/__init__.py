from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poulet_py.config.logging import LOGGER, setup_logging
    from poulet_py.config.settings import SETTINGS, Settings


__all__ = ["LOGGER", "SETTINGS", "Settings", "setup_logging"]
