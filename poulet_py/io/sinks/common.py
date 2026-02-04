try:
    from abc import ABC, abstractmethod

    from poulet_py import DataPacket
except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DataSink(ABC):
    @abstractmethod
    def push(self, packet: DataPacket) -> None: ...
