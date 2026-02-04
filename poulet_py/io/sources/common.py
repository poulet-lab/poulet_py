try:
    from abc import ABC, abstractmethod

    from poulet_py import DataSink
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DataSource(ABC):
    @abstractmethod
    def attach(self, sink: DataSink): ...
