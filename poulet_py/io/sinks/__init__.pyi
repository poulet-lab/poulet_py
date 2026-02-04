# ruff: noqa TID252
from .common import DataSink
from .queued import QueueDataSink

__all__ = ["DataSink", "QueueDataSink"]
