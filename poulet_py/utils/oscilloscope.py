from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

import datoviz as dvz
import monotonic
import numpy as np
from numpy.typing import NDArray
from datoviz import dvz_scene
class Oscilloscope:
    """
    Datoviz oscilloscope containing all 1D signals in one panel.

    This class knows nothing about SinkEvent or BaseSink.

    Data flow::

        add_trace()
            |
            +-- append()
            |
            +-- update()
                    |
                    v
              Datoviz path
              with subpaths
    """

    def __init__(
        self,
        scene: Any,
        panel: Any,
        *,
        max_field_of_view: float = 30.0,
        timestamp_scale: float = 1e-9,
        y_autoscale: bool = True,
        y_padding: float = 0.05,
        follow_latest: bool = True,
    ):
        self.scene = scene
        self.panel = panel

        self.max_field_of_view = max_field_of_view
        self.timestamp_scale = timestamp_scale

        self.y_autoscale = y_autoscale
        self.y_padding = y_padding

        self.follow_latest = follow_latest

        self.traces: dict[str, Trace] = {}

        self._lock = Lock()

        self._path = None

        self._positions = np.empty(
            (0, 3),
            dtype=np.float32,
        )

        self._colors = np.empty(
            (0, 4),
            dtype=np.uint8,
        )

        self._widths = np.empty(
            0,
            dtype=np.float32,
        )

        self._subpaths = np.empty(
            0,
            dtype=np.uint32,
        )

        self._dirty = False

        self._x_min: float | None = None
        self._x_max: float | None = None

        self._user_has_moved = False

        self._setup_panel()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_panel(self) -> None:
        if (
            dvz.dvz_panel_set_domain(
                self.panel,
                dvz.DVZ_DIM_X,
                0.0,
                self.max_field_of_view,
            )
            != 0
        ):
            raise RuntimeError("Failed to set oscilloscope X domain")

        if (
            dvz.dvz_panel_set_domain(
                self.panel,
                dvz.DVZ_DIM_Y,
                -1.0,
                1.0,
            )
            != 0
        ):
            raise RuntimeError("Failed to set oscilloscope Y domain")

    def _create_path(self) -> None:
        if self._path is not None:
            return

        self._path = dvz.dvz_path(
            self.scene,
            0,
        )

        if not self._path:
            raise RuntimeError("dvz_path() failed")

        if (
            dvz.dvz_path_set_caps(
                self._path,
                dvz.DVZ_SEGMENT_CAP_ROUND,
                dvz.DVZ_SEGMENT_CAP_ROUND,
            )
            != 0
        ):
            raise RuntimeError("dvz_path_set_caps() failed")

        if (
            dvz.dvz_path_set_join(
                self._path,
                dvz.DVZ_PATH_JOIN_ROUND,
                4.0,
            )
            != 0
        ):
            raise RuntimeError("dvz_path_set_join() failed")

        if (
            dvz.dvz_visual_set_depth_test(
                self._path,
                False,
            )
            != 0
        ):
            raise RuntimeError("dvz_visual_set_depth_test() failed")

        if (
            dvz.dvz_panel_add_visual(
                self.panel,
                self._path,
                None,
            )
            != 0
        ):
            raise RuntimeError("Failed to add oscilloscope path")

    # ------------------------------------------------------------------
    # Trace management
    # ------------------------------------------------------------------

    def add_trace(
        self,
        name: str,
        dtype: np.dtype,
        timestamps: NDArray[np.float64],
        values: NDArray,
    ) -> Trace:
        with self._lock:
            if name in self.traces:
                return self.traces[name]

            capacity = self._estimate_capacity(
                timestamps,
            )

            trace = Trace.create(
                name=name,
                dtype=dtype,
                capacity=capacity,
                color=self._trace_color(len(self.traces)),
            )

            self.traces[name] = trace

            self._create_path()

            return trace

    def _estimate_capacity(
        self,
        timestamps: NDArray[np.float64],
    ) -> int:
        if len(timestamps) < 2:
            return 4096

        dt = np.diff(timestamps)

        dt = dt[dt > 0]

        if len(dt) == 0:
            return 4096

        sample_period = float(np.median(dt))

        capacity = int(np.ceil(self.max_field_of_view / sample_period))

        return max(
            capacity,
            len(timestamps),
            1024,
        )

    @staticmethod
    def _trace_color(index: int) -> tuple[int, int, int, int]:
        # Simple deterministic oscilloscope palette.
        palette = (
            (76, 201, 240, 255),
            (128, 255, 219, 255),
            (255, 183, 3, 255),
            (239, 71, 111, 255),
            (155, 89, 182, 255),
            (46, 204, 113, 255),
            (255, 121, 63, 255),
            (241, 196, 15, 255),
        )

        return palette[index % len(palette)]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def append(
        self,
        name: str,
        timestamps: NDArray,
        values: NDArray,
    ) -> None:
        timestamps = np.asarray(
            timestamps,
            dtype=np.float64,
        )

        values = np.asarray(values)

        timestamps = timestamps * self.timestamp_scale

        with self._lock:
            trace = self.traces.get(name)

            if trace is None:
                trace = self.add_trace(
                    name,
                    values.dtype,
                    timestamps,
                    values,
                )

            trace.append(
                timestamps,
                values,
            )

            self._trim_trace(trace)

            self._dirty = True

    def _trim_trace(
        self,
        trace: Trace,
    ) -> None:
        if trace.size == 0:
            return

        timestamps, values = trace.snapshot()

        cutoff = timestamps[-1] - self.max_field_of_view

        first = np.searchsorted(
            timestamps,
            cutoff,
        )

        if first <= 0:
            return

        timestamps = timestamps[first:]
        values = values[first:]

        trace.size = len(timestamps)

        trace.timestamps[: trace.size] = timestamps
        trace.values[: trace.size] = values

        trace.write_pos = trace.size % trace.capacity

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(
        self,
        name: str,
        visible: bool,
    ) -> None:
        trace = self.traces[name]

        trace.visible = visible

        self._dirty = True

    def toggle(self, name: str) -> None:
        self.set_visible(
            name,
            not self.traces[name].visible,
        )

    def show_all(self) -> None:
        for trace in self.traces.values():
            trace.visible = True

        self._dirty = True

    def hide_all(self) -> None:
        for trace in self.traces.values():
            trace.visible = False

        self._dirty = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        Upload the current traces to Datoviz.

        All traces are represented by one path with multiple subpaths.
        Hidden traces have zero-length subpaths.
        """

        if not self._dirty:
            return

        with self._lock:
            visible = [trace for trace in self.traces.values() if trace.visible]

            if not visible:
                self._dirty = False
                return

            positions = []
            colors = []
            widths = []
            subpaths = []

            newest_timestamp = None

            for trace in visible:
                timestamps, values = trace.snapshot()

                if len(timestamps) < 2:
                    continue

                finite = np.isfinite(values)

                if not np.any(finite):
                    continue

                timestamps = timestamps[finite]
                values = values[finite]

                pos = np.empty(
                    (len(timestamps), 3),
                    dtype=np.float32,
                )

                pos[:, 0] = timestamps
                pos[:, 1] = values
                pos[:, 2] = 0.0

                color = np.empty(
                    (len(pos), 4),
                    dtype=np.uint8,
                )

                color[:] = trace.color

                width = np.full(
                    len(pos),
                    1.75,
                    dtype=np.float32,
                )

                positions.append(pos)
                colors.append(color)
                widths.append(width)

                subpaths.append(len(pos))

                newest = timestamps[-1]

                if newest_timestamp is None or newest > newest_timestamp:
                    newest_timestamp = newest

            if not positions:
                return

            self._positions = np.concatenate(positions)

            self._colors = np.concatenate(colors)

            self._widths = np.concatenate(widths)

            self._subpaths = np.asarray(
                subpaths,
                dtype=np.uint32,
            )

            if (
                dvz.dvz_visual_set_data_many(
                    self._path,
                    {
                        "position": self._positions,
                        "color": self._colors,
                        "stroke_width_px": self._widths,
                    },
                )
                != 0
            ):
                raise RuntimeError("Failed to update oscilloscope traces")

            lengths = np.ctypeslib.as_ctypes(self._subpaths)

            if (
                dvz.dvz_path_set_subpaths(
                    self._path,
                    len(self._subpaths),
                    lengths,
                )
                != 0
            ):
                raise RuntimeError("Failed to update oscilloscope subpaths")

            if newest_timestamp is not None and self.follow_latest:
                self._set_follow_domain(newest_timestamp)

            self._update_y_domain()

            self._dirty = False

    def _set_follow_domain(
        self,
        newest: float,
    ) -> None:
        self._x_max = newest
        self._x_min = newest - self.max_field_of_view

        if (
            dvz.dvz_panel_set_domain(
                self.panel,
                dvz.DVZ_DIM_X,
                self._x_min,
                self._x_max,
            )
            != 0
        ):
            raise RuntimeError("Failed to update oscilloscope X domain")

    def _update_y_domain(self) -> None:
        if not self.y_autoscale:
            return

        values = []

        for trace in self.traces.values():
            if not trace.visible:
                continue

            _, y = trace.snapshot()

            if len(y):
                finite = y[np.isfinite(y)]

                if len(finite):
                    values.append(finite)

        if not values:
            return

        values = np.concatenate(values)

        ymin = float(np.min(values))
        ymax = float(np.max(values))

        if ymin == ymax:
            delta = max(
                abs(ymin) * 0.05,
                1.0,
            )
        else:
            delta = (ymax - ymin) * self.y_padding

        ymin -= delta
        ymax += delta

        self._x_min = self._x_min

        if (
            dvz.dvz_panel_set_domain(
                self.panel,
                dvz.DVZ_DIM_Y,
                ymin,
                ymax,
            )
            != 0
        ):
            raise RuntimeError("Failed to update oscilloscope Y domain")

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def stop_following(self) -> None:
        self.follow_latest = False

    def start_following(self) -> None:
        self.follow_latest = True

    def toggle_following(self) -> None:
        self.follow_latest = not self.follow_latest

    def clear(self) -> None:
        with self._lock:
            for trace in self.traces.values():
                trace.size = 0
                trace.write_pos = 0

            self._dirty = True
