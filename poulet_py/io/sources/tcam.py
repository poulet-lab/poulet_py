
try:
    from numpy import dtype

    from poulet_py import BaseSource, ThermalCamera, precise_sleep

except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class TCAMSource(BaseSource, ThermalCamera):
    """Publish radiometric Y16 frames from a thermal camera."""

    def model_post_init(self, context) -> None:
        ThermalCamera.model_post_init(self, context)

    def _set_buffer_dtype(self) -> None:
        width, height = self.frame_size
        self._source_buffer_dtype = dtype(
            [
                ("timestamp", "uint64"),
                ("sequence", "uint32"),
                ("emissivity", "float32"),
                ("raw", "uint16", (height, width)),
            ]
        )

    def _open(self) -> None:
        ThermalCamera.open(self)
        ThermalCamera.start_streaming(self)

    def _close(self) -> None:
        ThermalCamera.close(self)

    def _fire(self) -> bool:
        precise_sleep(self._max_stimulus_duration_ms / 1000.0)
        return True

    def _acquire(self) -> None:
        while sample := ThermalCamera.read_sample(self, timeout=0):
            self._write_sample(
                (
                    sample.timestamp,
                    sample.sequence,
                    self.emissivity,
                    sample.raw,
                )
            )
