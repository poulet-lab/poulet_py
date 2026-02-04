# ruff: noqa TID252
from .common import DataSource
from .nidaq import NIAnalogInputSource

__all__ = ["DataSource", "NIAnalogInputSource"]
