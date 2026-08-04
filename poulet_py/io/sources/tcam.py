"""Continuous PureThermal/Lepton source for Poulet_py.

The camera implementation owns the UVC callback and its small frame queue.
This adapter only drains that queue, timestamps each frame, and publishes it
through :class:`BaseSource` so the normal sinks (including ``HDFSink``) can
store it alongside the other synchronized streams.
"""

try:
    from queue import Empty
    from threading import Event, Thread
    from time import monotonic_ns

    import numpy as np
    from pydantic import Field, PrivateAttr

    from poulet_py import LOGGER, BaseSource
    from poulet_py.hardware.camera.thermal_camera import ThermalCamera, q
except ImportError as e:
    raise ImportError("""
Missing thermal-camera source dependencies. Install options:
- Dedicated:    pip install poulet_py[camera,sources]
- Full:         pip install poulet_py[all]
""") from e


class TCAMSource(BaseSource):
    """Continuously publish radiometric thermal-camera frames in Celsius."""

    vminT: int = Field(default=30, description="Lower preview temperature in Celsius")
    vmaxT: int = Field(default=34, description="Upper preview temperature in Celsius")
    frame_timeout_s: float = Field(
        default=0.25,
        gt=0,
        description="Maximum queue wait used while stopping the drain thread",
    )

    _camera: ThermalCamera | None = PrivateAttr(default=None)
    _acquisition_thread: Thread | None = PrivateAttr(default=None)
    _stop_acquisition: Event = PrivateAttr(default_factory=Event)

    def open(self) -> None:
        if self._is_open:
            return

        # BaseSource creates its circular buffer before this thread may write.
        super().open()
        self._stop_acquisition.clear()
        self._acquisition_thread = Thread(
            target=self._drain_frames,
            daemon=True,
            name=f"{self.name}-acquisition",
        )
        self._acquisition_thread.start()

    def close(self) -> None:
        if not self._is_open:
            return

        # Stop all writes before BaseSource removes its circular buffer.
        self._stop_acquisition.set()
        if self._acquisition_thread is not None:
            self._acquisition_thread.join(timeout=self.frame_timeout_s + 1.0)
            if self._acquisition_thread.is_alive():
                LOGGER.warning("%s frame-drain thread did not stop", self.name)
        self._acquisition_thread = None
        super().close()

    def _set_buffer_dtype(self) -> None:
        if self._camera is None:
            raise RuntimeError("Thermal camera is not initialized")

        self._source_buffer_dtype = np.dtype(
            [
                ("timestamp", "uint64"),
                (
                    "temperature_celsius",
                    "float32",
                    (self._camera.height, self._camera.width),
                ),
            ]
        )

    def _open(self) -> None:
        # Discard frames left by an earlier camera session before streaming.
        while True:
            try:
                q.get_nowait()
            except Empty:
                break

        self._camera = ThermalCamera(vminT=self.vminT, vmaxT=self.vmaxT)
        self._camera.start_streaming()

    def _close(self) -> None:
        if self._camera is not None:
            self._camera.stop_streaming()
            self._camera = None

    def _drain_frames(self) -> None:
        if self._camera is None:
            return

        expected_shape = (self._camera.height, self._camera.width)

        while not self._stop_acquisition.is_set():
            try:
                raw = q.get(timeout=self.frame_timeout_s)
            except Empty:
                continue

            timestamp = monotonic_ns()
            raw = np.asarray(raw, dtype=np.uint16)

            if raw.shape != expected_shape:
                LOGGER.warning(
                    "%s dropped thermal frame with shape %s; expected %s",
                    self.name,
                    raw.shape,
                    expected_shape,
                )
                continue

            # Copy/convert immediately because the UVC callback buffer is reused.
            temperature_celsius = raw.astype(np.float32)
            temperature_celsius *= 0.01
            temperature_celsius -= 273.15
            self._write_sample((timestamp, temperature_celsius))
