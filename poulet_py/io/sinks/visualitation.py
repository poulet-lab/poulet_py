from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import datoviz as dvz


# ============================================================================
# Shared data structures
# ============================================================================


@dataclass
class Trace:
    name: str
    dtype: np.dtype
    capacity: int

    timestamps: NDArray[np.float64]
    values: NDArray

    size: int = 0
    write_pos: int = 0

    visual: Any = None
    color: Any = None

    visible: bool = True

    y_min: float | None = None
    y_max: float | None = None

    @classmethod
    def create(
        cls,
        name: str,
        dtype: np.dtype,
        capacity: int,
        color: Any = None,
    ) -> Trace:
        return cls(
            name=name,
            dtype=dtype,
            capacity=capacity,
            timestamps=np.empty(
                capacity,
                dtype=np.float64,
            ),
            values=np.empty(
                capacity,
                dtype=dtype,
            ),
            color=color,
        )

    def append(
        self,
        timestamps: NDArray[np.float64],
        values: NDArray,
    ) -> None:
        n = len(timestamps)

        if n == 0:
            return

        # If one event is larger than the complete buffer, retain only
        # the newest samples.
        if n >= self.capacity:
            timestamps = timestamps[-self.capacity :]
            values = values[-self.capacity :]
            n = self.capacity

        end = self.write_pos + n

        if end <= self.capacity:
            self.timestamps[self.write_pos:end] = timestamps
            self.values[self.write_pos:end] = values
        else:
            first = self.capacity - self.write_pos

            self.timestamps[self.write_pos:] = timestamps[:first]
            self.values[self.write_pos:] = values[:first]

            remaining = n - first

            self.timestamps[:remaining] = timestamps[first:]
            self.values[:remaining] = values[first:]

        self.write_pos = (self.write_pos + n) % self.capacity
        self.size = min(
            self.size + n,
            self.capacity,
        )

    def snapshot(self) -> tuple[
        NDArray[np.float64],
        NDArray,
    ]:
        if self.size == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=self.dtype),
            )

        start = (self.write_pos - self.size) % self.capacity

        if start + self.size <= self.capacity:
            return (
                self.timestamps[start : start + self.size].copy(),
                self.values[start : start + self.size].copy(),
            )

        first = self.capacity - start

        timestamps = np.concatenate(
            (
                self.timestamps[start:],
                self.timestamps[: self.size - first],
            )
        )

        values = np.concatenate(
            (
                self.values[start:],
                self.values[: self.size - first],
            )
        )

        return timestamps, values


@dataclass
class Image:
    name: str
    shape: tuple[int, int]
    dtype: np.dtype

    data: NDArray

    panel: Any = None
    visual: Any = None
    sampled_field: Any = None
    scale: Any = None

    visible: bool = True


# ============================================================================
# Oscilloscope
# ============================================================================


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
        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_X,
            0.0,
            self.max_field_of_view,
        ) != 0:
            raise RuntimeError(
                "Failed to set oscilloscope X domain"
            )

        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_Y,
            -1.0,
            1.0,
        ) != 0:
            raise RuntimeError(
                "Failed to set oscilloscope Y domain"
            )

    def _create_path(self) -> None:
        if self._path is not None:
            return

        self._path = dvz.dvz_path(
            self.scene,
            0,
        )

        if not self._path:
            raise RuntimeError(
                "dvz_path() failed"
            )

        if dvz.dvz_path_set_caps(
            self._path,
            dvz.DVZ_SEGMENT_CAP_ROUND,
            dvz.DVZ_SEGMENT_CAP_ROUND,
        ) != 0:
            raise RuntimeError(
                "dvz_path_set_caps() failed"
            )

        if dvz.dvz_path_set_join(
            self._path,
            dvz.DVZ_PATH_JOIN_ROUND,
            4.0,
        ) != 0:
            raise RuntimeError(
                "dvz_path_set_join() failed"
            )

        if dvz.dvz_visual_set_depth_test(
            self._path,
            False,
        ) != 0:
            raise RuntimeError(
                "dvz_visual_set_depth_test() failed"
            )

        if dvz.dvz_panel_add_visual(
            self.panel,
            self._path,
            None,
        ) != 0:
            raise RuntimeError(
                "Failed to add oscilloscope path"
            )

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
                color=self._trace_color(
                    len(self.traces)
                ),
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

        sample_period = float(
            np.median(dt)
        )

        capacity = int(
            np.ceil(
                self.max_field_of_view
                / sample_period
            )
        )

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

        return palette[
            index % len(palette)
        ]

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

        timestamps = (
            timestamps
            * self.timestamp_scale
        )

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

        cutoff = (
            timestamps[-1]
            - self.max_field_of_view
        )

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

        trace.write_pos = (
            trace.size
            % trace.capacity
        )

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
            visible = [
                trace
                for trace in self.traces.values()
                if trace.visible
            ]

            if not visible:
                self._dirty = False
                return

            positions = []
            colors = []
            widths = []
            subpaths = []

            newest_timestamp = None

            for trace in visible:
                timestamps, values = (
                    trace.snapshot()
                )

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

                subpaths.append(
                    len(pos)
                )

                newest = timestamps[-1]

                if (
                    newest_timestamp is None
                    or newest > newest_timestamp
                ):
                    newest_timestamp = newest

            if not positions:
                return

            self._positions = np.concatenate(
                positions
            )

            self._colors = np.concatenate(
                colors
            )

            self._widths = np.concatenate(
                widths
            )

            self._subpaths = np.asarray(
                subpaths,
                dtype=np.uint32,
            )

            if dvz.dvz_visual_set_data_many(
                self._path,
                {
                    "position": self._positions,
                    "color": self._colors,
                    "stroke_width_px": self._widths,
                },
            ) != 0:
                raise RuntimeError(
                    "Failed to update oscilloscope traces"
                )

            lengths = np.ctypeslib.as_ctypes(
                self._subpaths
            )

            if dvz.dvz_path_set_subpaths(
                self._path,
                len(self._subpaths),
                lengths,
            ) != 0:
                raise RuntimeError(
                    "Failed to update oscilloscope subpaths"
                )

            if (
                newest_timestamp is not None
                and self.follow_latest
            ):
                self._set_follow_domain(
                    newest_timestamp
                )

            self._update_y_domain()

            self._dirty = False

    def _set_follow_domain(
        self,
        newest: float,
    ) -> None:
        self._x_max = newest
        self._x_min = (
            newest
            - self.max_field_of_view
        )

        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_X,
            self._x_min,
            self._x_max,
        ) != 0:
            raise RuntimeError(
                "Failed to update oscilloscope X domain"
            )

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
            delta = (
                ymax - ymin
            ) * self.y_padding

        ymin -= delta
        ymax += delta

        self._x_min = self._x_min

        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_Y,
            ymin,
            ymax,
        ) != 0:
            raise RuntimeError(
                "Failed to update oscilloscope Y domain"
            )

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def stop_following(self) -> None:
        self.follow_latest = False

    def start_following(self) -> None:
        self.follow_latest = True

    def toggle_following(self) -> None:
        self.follow_latest = (
            not self.follow_latest
        )

    def clear(self) -> None:
        with self._lock:
            for trace in self.traces.values():
                trace.size = 0
                trace.write_pos = 0

            self._dirty = True


# ============================================================================
# Image viewer
# ============================================================================


class ImageViewer:
    """
    One Datoviz panel displaying one 2D NumPy array.

    The image visual and sampled field are created once.

    Subsequent frames update the sampled field rather than recreating
    the visual.
    """

    def __init__(
        self,
        scene: Any,
        panel: Any,
        name: str,
        data: NDArray,
    ):
        if data.ndim != 2:
            raise ValueError(
                f"ImageViewer requires 2D data, got {data.shape}"
            )

        self.scene = scene
        self.panel = panel
        self.name = name

        self.shape = data.shape
        self.dtype = data.dtype

        self.data = np.array(
            data,
            copy=True,
            order="C",
        )

        self.visual = None
        self.sampled_field = None
        self.scale = None

        self.visible = True

        self._dirty = True

        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        height, width = self.shape

        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_X,
            0.0,
            float(width),
        ) != 0:
            raise RuntimeError(
                f"Failed to set X domain for '{self.name}'"
            )

        if dvz.dvz_panel_set_domain(
            self.panel,
            dvz.DVZ_DIM_Y,
            0.0,
            float(height),
        ) != 0:
            raise RuntimeError(
                f"Failed to set Y domain for '{self.name}'"
            )

        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, float(height), 0.0],
                [float(width), 0.0, 0.0],
                [float(width), float(height), 0.0],
            ],
            dtype=np.float32,
        )

        texcoords = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.visual = dvz.dvz_image(
            self.scene,
            0,
        )

        if not self.visual:
            raise RuntimeError(
                f"dvz_image() failed for '{self.name}'"
            )

        if dvz.dvz_visual_set_data_many(
            self.visual,
            {
                "position": positions,
                "texcoords": texcoords,
            },
        ) != 0:
            raise RuntimeError(
                f"Failed to configure image '{self.name}'"
            )

        if dvz.dvz_visual_set_depth_test(
            self.visual,
            False,
        ) != 0:
            raise RuntimeError(
                f"Failed to configure image '{self.name}'"
            )

        self.sampled_field = (
            dvz.dvz_sampled_field_from_array(
                self.scene,
                self.data,
            )
        )

        if not self.sampled_field:
            raise RuntimeError(
                f"Failed to create sampled field "
                f"for '{self.name}'"
            )

        if dvz.dvz_visual_set_field(
            self.visual,
            b"field",
            self.sampled_field,
        ) != 0:
            raise RuntimeError(
                f"Failed to bind sampled field "
                f"for '{self.name}'"
            )

        if dvz.dvz_panel_add_visual(
            self.panel,
            self.visual,
            None,
        ) != 0:
            raise RuntimeError(
                f"Failed to add image '{self.name}'"
            )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_data(
        self,
        data: NDArray,
    ) -> None:
        data = np.asarray(data)

        if data.shape != self.shape:
            raise ValueError(
                f"Image '{self.name}' changed shape: "
                f"{self.shape} -> {data.shape}. "
                "Create a new ImageViewer for a different shape."
            )

        if data.dtype != self.dtype:
            raise ValueError(
                f"Image '{self.name}' changed dtype: "
                f"{self.dtype} -> {data.dtype}"
            )

        np.copyto(
            self.data,
            data,
        )

        self._dirty = True

    def update(self) -> None:
        if not self._dirty:
            return

        # Keep the sampled field alive and update its contents.
        #
        # dvz_sampled_field_from_array() is used during construction;
        # this call is the public update mechanism for an existing field.
        if dvz.dvz_sampled_field_set_data(
            self.sampled_field,
            self.data,
        ) != 0:
            raise RuntimeError(
                f"Failed to update image '{self.name}'"
            )

        self._dirty = False

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(
        self,
        visible: bool,
    ) -> None:
        self.visible = visible

        if dvz.dvz_visual_set_visible(
            self.visual,
            visible,
        ) != 0:
            raise RuntimeError(
                f"Failed to change visibility "
                f"of '{self.name}'"
            )

    def toggle(self) -> None:
        self.set_visible(
            not self.visible
        )


# ============================================================================
# VisSink
# ============================================================================


class VisSink(BaseSink):
    """
    Automatic visualization sink.

    Structured-array fields are dispatched according to dimensionality:

        timestamp + 1D fields
            -> one Oscilloscope

        2D fields
            -> one ImageViewer per field

    Example payload::

        dtype = np.dtype([
            ("timestamp", np.int64),
            ("voltage", np.float32),
            ("current", np.float32),
            ("camera", np.float32, (512, 512)),
        ])

    Results in::

        ┌──────────────────────────────────────────┐
        │              Oscilloscope                │
        │ voltage ──────────────────────────────── │
        │ current ──────────────────────────────── │
        └──────────────────────────────────────────┘

        ┌────────────────────┐  ┌──────────────────┐
        │ camera              │  │ another_image    │
        │                     │  │                  │
        │      IMAGE          │  │      IMAGE       │
        └────────────────────┘  └──────────────────┘
    """

    max_field_of_view: float = Field(
        default=30.0,
        gt=0,
    )

    fps: float = Field(
        default=30.0,
        gt=0,
    )

    timestamp_scale: float = Field(
        default=1e-9,
    )

    width: int = Field(
        default=1600,
        gt=0,
    )

    height: int = Field(
        default=1000,
        gt=0,
    )

    columns: int = Field(
        default=2,
        gt=0,
    )

    max_image_viewers: int = Field(
        default=8,
        gt=0,
    )

    follow_latest: bool = Field(
        default=True,
    )

    _scene: Any = PrivateAttr(default=None)
    _figure: Any = PrivateAttr(default=None)
    _grid: Any = PrivateAttr(default=None)
    _view: Any = PrivateAttr(default=None)
    _app: Any = PrivateAttr(default=None)

    _oscilloscope: Oscilloscope | None = PrivateAttr(
        default=None
    )

    _images: dict[str, ImageViewer] = PrivateAttr(
        default_factory=dict
    )

    _panels: list[Any] = PrivateAttr(
        default_factory=list
    )

    _lock: Lock = PrivateAttr(
        default_factory=Lock
    )

    _last_render: float = PrivateAttr(
        default=0.0
    )

    # ------------------------------------------------------------------
    # BaseSink
    # ------------------------------------------------------------------

    def _open(self):
        self._scene = dvz.dvz_scene()

        if not self._scene:
            raise RuntimeError(
                "dvz_scene() failed"
            )

        self._figure = dvz.dvz_figure(
            self._scene,
            self.width,
            self.height,
            0,
        )

        if not self._figure:
            raise RuntimeError(
                "dvz_figure() failed"
            )

        self._create_layout()

        self._oscilloscope = Oscilloscope(
            self._scene,
            self._panels[0],
            max_field_of_view=self.max_field_of_view,
            timestamp_scale=self.timestamp_scale,
            follow_latest=self.follow_latest,
        )

        # The rest of the GUI initialization should be connected to
        # whatever application/view runner you use in your Datoviz setup.
        #
        # Your supplied Datoviz example uses:
        #
        #     ex.run_with_view(scene, figure, ..., configure)
        #
        # and configures pan/zoom separately.  Keep that application
        # lifecycle outside the data classes.

    def _create_layout(self) -> None:
        """
        Create the fixed layout.

        Row 0:
            oscilloscope, spanning all columns.

        Remaining rows:
            one image viewer per cell.
        """

        image_rows = int(
            np.ceil(
                self.max_image_viewers
                / self.columns
            )
        )

        rows = 1 + image_rows

        self._grid = dvz.dvz_figure_grid(
            self._figure,
            rows,
            self.columns,
        )

        if not self._grid:
            raise RuntimeError(
                "dvz_figure_grid() failed"
            )

        # Oscilloscope spans the complete first row.
        osc_panel = dvz.dvz_grid_panel_span(
            self._grid,
            0,
            0,
            1,
            self.columns,
        )

        if not osc_panel:
            raise RuntimeError(
                "Failed to create oscilloscope panel"
            )

        self._panels.append(
            osc_panel
        )

        # Image panels.
        for index in range(
            self.max_image_viewers
        ):
            row = (
                index // self.columns
            ) + 1

            column = (
                index % self.columns
            )

            panel = dvz.dvz_grid_panel(
                self._grid,
                row,
                column,
            )

            if not panel:
                raise RuntimeError(
                    f"Failed to create image panel {index}"
                )

            self._panels.append(
                panel
            )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event(
        self,
        event: BaseEvent,
    ):
        if not isinstance(event, SinkEvent):
            return

        payload = event.payload

        if not isinstance(payload, np.ndarray):
            return

        if payload.dtype.names is None:
            raise TypeError(
                "VisSink expects a structured NumPy array"
            )

        if "timestamp" not in payload.dtype.names:
            raise ValueError(
                "VisSink requires a 'timestamp' field"
            )

        timestamps = np.asarray(
            payload["timestamp"],
            dtype=np.float64,
        )

        for name in payload.dtype.names:
            if name == "timestamp":
                continue

            values = payload[name]

            if values.ndim == 1:
                self._handle_trace(
                    name,
                    timestamps,
                    values,
                )

            elif values.ndim == 2:
                self._handle_image(
                    name,
                    values,
                )

            else:
                raise ValueError(
                    f"Field '{name}' has unsupported "
                    f"shape {values.shape}. "
                    "Only 1D and 2D fields are supported."
                )

    def _handle_trace(
        self,
        name: str,
        timestamps: NDArray,
        values: NDArray,
    ) -> None:
        if self._oscilloscope is None:
            raise RuntimeError(
                "Oscilloscope has not been initialized"
            )

        self._oscilloscope.append(
            name,
            timestamps,
            values,
        )

    def _handle_image(
        self,
        name: str,
        values: NDArray,
    ) -> None:
        with self._lock:
            viewer = self._images.get(name)

            if viewer is None:
                viewer = self._create_image_viewer(
                    name,
                    values,
                )

                self._images[name] = viewer

            viewer.set_data(values)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def _create_image_viewer(
        self,
        name: str,
        values: NDArray,
    ) -> ImageViewer:
        index = len(self._images)

        if index >= self.max_image_viewers:
            raise RuntimeError(
                f"Maximum number of image viewers "
                f"({self.max_image_viewers}) exceeded"
            )

        panel = self._panels[
            index + 1
        ]

        return ImageViewer(
            self._scene,
            panel,
            name,
            values,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        Render at most once per FPS interval.

        This should be called by the Datoviz application's timer/event
        callback.
        """

        now = monotonic()

        if (
            now - self._last_render
            < 1.0 / self.fps
        ):
            return

        self._last_render = now

        if self._oscilloscope:
            self._oscilloscope.update()

        with self._lock:
            viewers = list(
                self._images.values()
            )

        for viewer in viewers:
            viewer.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def oscilloscope(self) -> Oscilloscope:
        if self._oscilloscope is None:
            raise RuntimeError(
                "VisSink is not open"
            )

        return self._oscilloscope

    @property
    def images(
        self,
    ) -> dict[str, ImageViewer]:
        return self._images

    def show(
        self,
        name: str,
    ) -> None:
        if name in self._images:
            self._images[name].set_visible(
                True
            )
        elif name in self.oscilloscope.traces:
            self.oscilloscope.set_visible(
                name,
                True,
            )
        else:
            raise KeyError(name)

    def hide(
        self,
        name: str,
    ) -> None:
        if name in self._images:
            self._images[name].set_visible(
                False
            )
        elif name in self.oscilloscope.traces:
            self.oscilloscope.set_visible(
                name,
                False,
            )
        else:
            raise KeyError(name)

    def toggle(
        self,
        name: str,
    ) -> None:
        if name in self._images:
            self._images[name].toggle()
        elif name in self.oscilloscope.traces:
            self.oscilloscope.toggle(
                name
            )
        else:
            raise KeyError(name)

    def follow_latest(
        self,
        enabled: bool,
    ) -> None:
        self.oscilloscope.follow_latest = enabled

    def clear(self) -> None:
        self.oscilloscope.clear()
```
