try:
    import threading
    from time import monotonic_ns, sleep

    import cv2
    import numpy as np
    from numpy import ndarray
    from pydantic import ConfigDict, Field, PrivateAttr

    from poulet_py import BaseEvent, BaseSink, LOGGER, Oscilloscope, SinkEvent

except ImportError as e:
    raise ImportError(
        """
Missing oscilloscope sink dependencies.

Install options:
- pip install poulet_py[io,osc,camera]
- pip install poulet_py[all]
"""
    ) from e


DEFAULT_SOURCE_PANELS = {
    "ina228_mouse": "voltages",
    "ina228_pad": "voltages",
    "tcs": "thermal",
    "counter": "events",
    "trial": "events",
    "drv2605": "events",
    "metadata": "events",
}


class OscilloscopeSink(BaseSink):
    """Forward SinkEvent chunks to grouped Oscilloscope panels with decimation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scopes: dict[str, Oscilloscope] = Field(
        ...,
        description="Panel name to Oscilloscope instance",
    )
    source_panels: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SOURCE_PANELS),
        description="Source name to panel name",
    )
    decimate_hz: float = Field(default=10.0, gt=0.0)
    history_seconds: float = Field(default=30.0, gt=0.0)
    enable_camera_preview: bool = Field(default=True)
    camera_window_name: str = Field(default="dcam_preview")

    _start_mono_ns: int = PrivateAttr(default=0)
    _last_push_ns: dict[str, int] = PrivateAttr(default_factory=dict)
    _decimate_interval_ns: int = PrivateAttr(default=0)
    _latest_frame: ndarray | None = PrivateAttr(default=None)
    _frame_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _preview_thread: threading.Thread | None = PrivateAttr(default=None)
    _preview_stop: threading.Event = PrivateAttr(default_factory=threading.Event)

    def _open(self) -> None:
        self._start_mono_ns = monotonic_ns()
        self._last_push_ns = {}
        self._decimate_interval_ns = int(1_000_000_000 / self.decimate_hz)
        for scope in self.scopes.values():
            scope.start()
        if self.enable_camera_preview:
            self._preview_stop.clear()
            self._preview_thread = threading.Thread(
                target=self._camera_preview_loop,
                daemon=True,
                name="OscilloscopeSink-CameraPreview",
            )
            self._preview_thread.start()

    def _close(self) -> None:
        self._preview_stop.set()
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
        self._preview_thread = None
        for scope in self.scopes.values():
            scope.stop()
        try:
            cv2.destroyWindow(self.camera_window_name)
        except cv2.error:
            pass

    def _on_event(self, event: BaseEvent) -> None:
        if not isinstance(event, SinkEvent):
            return

        payload = event.payload
        if not isinstance(payload, ndarray) or payload.size == 0:
            return

        if event.name == "dcam" and "dcam" in payload.dtype.names:
            self._store_latest_frame(payload[-1]["dcam"])
            return

        panel = self.source_panels.get(event.name)
        if panel is None or panel not in self.scopes:
            return

        now = monotonic_ns()
        last = self._last_push_ns.get(event.name, 0)
        if now - last < self._decimate_interval_ns:
            return
        self._last_push_ns[event.name] = now

        row = payload[-1]
        values = self._extract_series(event.name, row)
        if not values:
            return

        elapsed_s = (int(row["timestamp"]) - self._start_mono_ns) / 1_000_000_000
        self.scopes[panel].add_data(values, x=elapsed_s)

    def _extract_series(self, source_name: str, row: ndarray) -> dict[str, float]:
        values: dict[str, float] = {}
        if source_name in {"ina228_mouse", "ina228_pad"}:
            values[source_name] = float(row["bus_voltage"])
            return values

        if source_name == "tcs":
            for idx in range(5):
                key = f"s{idx}"
                if key in row.dtype.names:
                    values[key] = float(row[key])
            return values

        if source_name in {"counter", "trial"}:
            values["counter"] = float(row["counter"])
            return values

        if source_name == "drv2605":
            if "drive_voltage" in row.dtype.names:
                values["drv_voltage"] = float(row["drive_voltage"])
            if "waveform" in row.dtype.names:
                values["waveform"] = float(row["waveform"])
            return values

        if source_name == "metadata":
            return values

        for name in row.dtype.names or ():
            if name == "timestamp":
                continue
            try:
                values[name] = float(row[name])
            except (TypeError, ValueError):
                continue
        return values

    def _store_latest_frame(self, frame: ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def _camera_preview_loop(self) -> None:
        cv2.namedWindow(
            self.camera_window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
        )
        interval_s = self._decimate_interval_ns / 1_000_000_000
        while not self._preview_stop.is_set():
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
            if frame is not None:
                display = self._prepare_frame(frame)
                cv2.imshow(self.camera_window_name, display)
                cv2.waitKey(1)
            sleep(interval_s)

    @staticmethod
    def _prepare_frame(data: ndarray) -> ndarray:
        if data.dtype == np.uint16:
            maxval = np.amax(data)
            if maxval > 0:
                scale = int(65535 / maxval)
                return data * scale
        return data
