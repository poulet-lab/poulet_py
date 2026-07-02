try:
    from poulet_py import DCAM, BaseSource, precise_sleep
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class DCAMSource(DCAM, BaseSource):
    def _set_buffer_dtype(self):
        self._buffer_dtype = [
            ("timestamp", "uint64"),
            (
                "dcam",
                self.DTYPE_MAP.get(self.pixel_type, "float32"),
                (self.__dcam_internal_buffer.height, self.__dcam_internal_buffer.width),
            ),
        ]

    def _open(self):
        DCAM.open(self)

    def _close(self):
        DCAM.close(self)

    def _trigger(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True
