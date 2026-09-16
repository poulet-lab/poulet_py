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
