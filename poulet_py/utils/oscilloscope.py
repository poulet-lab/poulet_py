import ctypes
import time
from queue import Full, Queue
from threading import Thread
from typing import Any, ClassVar, Literal

import datoviz as dvz
import numpy as np
from darkdetect import isDark
from pydantic import BaseModel, Field, PrivateAttr
from screeninfo import ScreenInfoError, get_monitors

from poulet_py.config.logging import LOGGER


class Trace(BaseModel):
    key: str = Field(...)
    label: str = Field(...)
    color: tuple[int, int, int, int] | Literal["random"] = Field(default="random")

    _current_bin: int = PrivateAttr(default=0)
    _gid: np.ndarray  # (n_bins,) int64 global bin id stored in each slot

    _positions: np.ndarray  # (n_pts - 1, 2, 3) LINE_LIST segments

    def model_post_init(self, context):
        self._gid = (np.full(self._n_bins, _INVALID, dtype="int64"),)

    def _fill_positions(self):
        n = self.n_bins
        pos = self.positions
        valid = (
            (self._gid > self.g_latest - n)  # newer than one sweep
            & (self._gid > _INVALID)
        )
        idx = np.flatnonzero(valid)
        k = idx.size
        if k == 0:
            pos[:] = 0.0
            self.lo, self.hi = float("inf"), float("-inf")
            return

        w = self._bin_w
        px = np.empty(2 * k, np.float32)
        py = np.empty(2 * k, np.float32)
        px[0::2], px[1::2] = (idx + 0.25) * w, (idx + 0.75) * w
        py[0::2], py[1::2] = self.mn[idx], self.mx[idx]

        m = 2 * k
        pos[: m - 1, 0, 0], pos[: m - 1, 1, 0] = px[:-1], px[1:]
        pos[: m - 1, 0, 1], pos[: m - 1, 1, 1] = py[:-1], py[1:]
        pos[m - 1 :, :, 0] = px[-1]  # unused tail: zero-length at the last point
        pos[m - 1 :, :, 1] = py[-1]

        # Break the line where new data (left of the cursor) meets the previous sweep.
        j = int(np.searchsorted(idx, self.g_latest % n, side="right"))
        if 0 < j < k:
            seam = 2 * j - 1
            pos[seam, 1, :2] = pos[seam, 0, :2]
        self.lo, self.hi = float(py.min()), float(py.max())


class Panel2D(BaseModel):
    CURSOR_COLOR: ClassVar[np.ndarray] = np.array([250, 183, 3, 255], dtype=np.uint8)
    CURSOR_PAD = 20.0

    name: str = Field(...)
    scene: Any = Field(...)
    figure: Any = Field(...)
    view: Any = Field(...)

    x_lim: tuple[float, float] | None = Field(default=None)
    y_lim: tuple[float, float] | None = Field(default=None)
    x_label: str | None = Field(default=None)
    y_label: str | None = Field(default=None)
    cursor: bool = Field(default=True)
    grid: bool = Field(default=True)
    legend: bool = Field(default=True)
    static: bool = Field(default=False)

    _x_latest: float = PrivateAttr(DEFAULT=float("-inf"))
    _x_lim: tuple[float, float] = PrivateAttr(default=(0, 1))
    _y_lim: tuple[float, float] = PrivateAttr(default=(-1, 1))
    _position: tuple[float, float, float, float] = PrivateAttr(default=(0.0, 0.0, 1.0, 1.0))

    _panel: Any | None = PrivateAttr(None)
    _x_axis: Any | None = PrivateAttr(None)
    _y_axis: Any | None = PrivateAttr(None)

    _cursor: Any = PrivateAttr(None)
    _traces: dict[str, Trace] = PrivateAttr(default_factory=dict)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos: tuple[float, float, float, float]):
        if not isinstance(pos, tuple[float, float, float, float]):
            raise RuntimeError("position should be tuple[float, float, float, float]")

        desc = dvz.DvzPanelDesc(x=pos[0], y=pos[1], width=pos[2], height=pos[3])
        if dvz.dvz_panel_set_desc(self._panel, desc) != 0:
            raise RuntimeError(f"dvz_panel_set_desc() failed for {self.name!r}")

        self._position = pos

    def model_post_init(self):
        self._set_panel()
        self._set_theme()
        self._set_axes()
        self._set_labels()
        self._set_grid()
        self._set_domain()
        self._bind_panzoom()

        self._create_cursor()

        self._create_trace()
        self._create_legend()

    def _set_panel(self):
        desc = dvz.DvzPanelDesc(self._panel_position)

        self._panel = dvz.dvz_panel(self.scene, self.figure, desc)
        if not self._panel:
            raise RuntimeError(f"dvz_panel() failed for {self.name!r}")

    def _set_theme(self):
        dvz.dvz_panel_set_background_color(
            self._panel, self.BG_DARK if self._dark else self.BG_WHITE
        )

    def _set_axes(self):
        self._x_axis = dvz.dvz_panel_axis(self._panel, dvz.DVZ_DIM_X)
        self._y_axis = dvz.dvz_panel_axis(self._panel, dvz.DVZ_DIM_X)
        if not self._x_axis or not self._y_axis:
            raise RuntimeError("dvz_panel_axis() failed")

    def _set_labels(self):
        if self.x_label:
            dvz.dvz_axis_set_label(self._x_axis, self.x_label.encode())
        if self.y_label:
            dvz.dvz_axis_set_label(self._y_axis, self.y_label.encode())

    def _set_grid(self):
        if self.grid:
            dvz.dvz_axis_set_grid(self._x_axis, True)
            dvz.dvz_axis_set_grid(self._y_axis, True)

    def _set_domain(self):
        if (
            dvz.dvz_panel_set_domain(self._panel, dvz.DVZ_DIM_X, self._x_lim[0], self._x_lim[1])
            != 0
        ):
            raise RuntimeError("dvz_panel_set_domain(X) failed")

        if (
            dvz.dvz_panel_set_domain(self._panel, dvz.DVZ_DIM_Y, self._y_lim[0], self._y_lim[1])
            != 0
        ):
            raise RuntimeError("dvz_panel_set_domain(Y) failed")

    def _bind_panzoom(self):
        if self.static:
            return

        controller = dvz.dvz_panzoom(self.scene, None)
        if not controller:
            raise RuntimeError("dvz_panzoom() failed")

        if (
            dvz.dvz_view_bind_controller(self.view, self._panel, controller, dvz.DVZ_DIM_MASK_XY)
            != 0
        ):
            raise RuntimeError("dvz_view_bind_controller() failed")

    def _create_cursor(self):
        if not self.cursor:
            return

        self._cursor = dvz.dvz_primitive(self.scene, dvz.DVZ_PRIMITIVE_TOPOLOGY_LINE_LIST, 0)
        if not self._cursor:
            raise RuntimeError("dvz_primitive() failed")

        dvz.dvz_visual_set_depth_test(self._cursor, False)
        dvz.dvz_panel_add_visual(self._panel, self._cursor, None)

    def _create_legend(self):
        if not self.legend:
            return

        if self.split_traces or len(self.traces) == 0:
            return

        palette = self._PALETTE_DARK if self._dark else self._PALETTE_WHITE

        for i, trace in enumerate(panel.traces):
            color = palette[i % len(palette)]

            style = dvz.dvz_text_style()
            style.size_px = 14.0
            style.renderer = dvz.DVZ_TEXT_RENDERER_MSDF_ATLAS
            style.color[:] = tuple(int(c) for c in color)

            placement = dvz.dvz_text_placement()
            placement.mode = dvz.DVZ_TEXT_PLACEMENT_SCREEN
            placement.anchor = dvz.DVZ_SCENE_ANCHOR_PANEL_TOP_LEFT
            placement.position[:] = (12.0, 14.0 + i * 20.0, 0.0)
            placement.text_anchor[:] = (0.0, 0.5)
            placement.has_text_anchor = True
            placement.depth_test = False

            desc = dvz.dvz_label_desc()
            desc.text = f"{trace.key}[{trace.label}]".encode()

            annotation = dvz.dvz_annotation_label(self.handle, ctypes.byref(desc))
            if not annotation:
                raise RuntimeError("dvz_annotation_label() failed")

            if dvz.dvz_annotation_set_style(annotation, ctypes.byref(style)) != 0:
                raise RuntimeError("dvz_annotation_set_style() failed")

            if dvz.dvz_annotation_set_placement(annotation, ctypes.byref(placement)) != 0:
                raise RuntimeError("dvz_annotation_set_placement() failed")

    def _create_trace(self, key: str, label: str, color_index: int):
        palette = self._PALETTE_DARK if self._dark else self._PALETTE_WHITE
        n_seg = 2 * self._n_bins - 1
        positions = np.zeros((n_seg, 2, 3), dtype=np.float32)  # all zero-length = invisible
        visual = self._make_line_visual(
            positions.reshape(-1, 3), palette[color_index % len(palette)], self
        )
        self.traces.append(Trace(key=key, label=label))

    def _update_cursor(self):
        pad = self.CURSOR_PAD * (self._y_lim[1] - self._y_lim[0])
        cx = self._x_latest % self.duration
        points = np.array(
            [[cx, self._y_lim[0] - pad, 0.0], [cx, self._y_lim[1] + pad, 0.0]], dtype=np.float32
        )
        if dvz.dvz_visual_set_data(self._cursor, "position", points) != 0:
            raise RuntimeError("cursor update failed")

    def _panel_specs(self):
        if self.split_traces:
            for key, (_x, _Y, labels) in packet.items():
                for label in labels:
                    name = key if len(labels) == 1 else f"{key}[{label}]"
                    yield name, [(key, label)]
        else:
            for key, (_x, _Y, labels) in packet.items():
                yield key, [(key, label) for label in labels]

    def _autoscale(self):
        los = [t.lo for t in panel.traces if np.isfinite(t.lo)]
        his = [t.hi for t in panel.traces if np.isfinite(t.hi)]
        if not los:
            return
        lo, hi = min(los), max(his)
        span = hi - lo
        pad = 0.1 * span if span > 1e-9 else max(0.1 * abs(hi), 1.0)
        lo, hi = lo - pad, hi + pad
        # expand immediately, shrink only when clearly oversized -> no jitter
        if lo < panel.y_lo or hi > panel.y_hi or (hi - lo) < 0.5 * (panel.y_hi - panel.y_lo):
            panel.y_lo, panel.y_hi = lo, hi
            self._set_domain(panel.handle, lo, hi)

    def set_data(self, data: dict[str, np.ndarray]):
        for name, d in data:
            if name not in self._traces.key():
                self._create_trace(name)
                self._update_legend()

            self._traces[name].set_data(d)
            self._autoscale()


class Oscilloscope(BaseModel):
    BG_DARK: ClassVar[Any] = dvz.DvzColor(20, 20, 20, 255)
    BG_WHITE: ClassVar[Any] = dvz.DvzColor(240, 240, 240, 255)

    _PALETTE_DARK: ClassVar[np.ndarray] = np.array(
        [
            [94, 213, 220, 255],
            [255, 103, 129, 255],
            [132, 224, 210, 255],
            [180, 160, 255, 255],
        ],
        dtype=np.uint8,
    )
    _PALETTE_WHITE: ClassVar[np.ndarray] = np.array(
        [
            [20, 90, 160, 255],
            [190, 40, 80, 255],
            [20, 130, 110, 255],
            [110, 70, 190, 255],
        ],
        dtype=np.uint8,
    )

    title: str = Field(default="Oscilloscope")
    duration: float = Field(default=6.0, gt=0, description="Sweep length in x units")
    fps: int = Field(default=25, gt=0, description="Maximum GPU upload rate")
    max_points: int = Field(default=2000, ge=16, description="Display points per trace")
    max_panels: int = Field(default=64, gt=0, description="Safety cap on panel count")
    split_traces: bool = Field(
        default=True,
        description="True: one panel per trace. False: all traces of dataset in one panel.",
    )
    y_range: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = autoscale per panel."
    )
    size: Literal["auto"] | tuple[int, int] = Field(default="auto")
    gutter: tuple[float, float] = Field(default=(0.0, 7.0))
    theme: Literal["auto", "dark", "white"] = Field(default="auto")
    queue_size: int = Field(default=1000, gt=0, description="Size of the internal queue")

    # Datoviz visual consts
    _size: tuple[int, int] = PrivateAttr(default=(1280, 720))
    _dark: bool = PrivateAttr(default=True)
    # check why:
    _n_bins: int = PrivateAttr(default=8)
    _bin_w: float = PrivateAttr(default=1.0)
    _inv_w: float = PrivateAttr(default=1.0)

    _queue: Queue[dict[str, np.ndarray]] = PrivateAttr()

    # Datoviz handles + layout (render thread only)
    _app: Any = PrivateAttr(default=None)
    _view: Any = PrivateAttr(default=None)
    _scene: Any = PrivateAttr(default=None)
    _figure: Any = PrivateAttr(default=None)
    _panels: dict[str, Panel2D] = PrivateAttr(default_factory=dict)

    # lifecycle
    _thread: Thread | None = PrivateAttr(default=None)

    _last_upload: float = PrivateAttr(default=0.0)
    _frame_errors: int = PrivateAttr(default=0)
    _started: bool = PrivateAttr(default=False)

    _is_open: bool = PrivateAttr(default=False)

    def model_post_init(self, context):
        if self.size == "auto":
            try:
                monitors = get_monitors()
                monitor = next(
                    (m for m in monitors if getattr(m, "is_primary", False)),
                    monitors[0],
                )
                self._size = (int(monitor.width * 0.9), int(monitor.height * 0.9))
            except (ScreenInfoError, IndexError):
                pass  # keep the 1280x720 fallback
        else:
            self._size = self.size

        self._dark = self.theme == "dark" or (self.theme == "auto" and bool(isDark()))

        # TODO
        self._n_bins = max(8, self.max_points // 2)
        self._bin_w = self.duration / self._n_bins
        self._inv_w = self._n_bins / self.duration

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self):
        if self._is_open:
            return

        self._queue = Queue(maxsize=self.queue_size)

        self._set_scene()
        self._app = dvz.dvz_app(self._scene)
        if not self._app:
            raise RuntimeError("dvz_app() failed")

        self._view = dvz.dvz_view_window(
            self._app, self._figure, self._size[0], self._size[1], self.title.encode()
        )
        if not self._view:
            raise RuntimeError("dvz_view_window() failed")

        if dvz.dvz_view_set_frame_callback(self._view, self._frame_callback, None) != 0:
            raise RuntimeError("dvz_view_set_frame_callback() failed")

        self._thread = Thread(
            target=self._run_dvz_app, name=f"{type(self).__name__}_thread", daemon=True
        )
        self._thread.start()

        self._is_open = True

    def close(
        self,
    ):
        if not self._is_open:
            return

        self._thread.join()
        if self._thread.is_alive():
            LOGGER.warning("Render thread still running; close the window.")
        else:
            self._thread = None

        self._reset()
        self._release_scene()

        del self._queue

        self._is_open = False

    def add_data(self, data: dict[str, np.ndarray]):
        self._ensure_open()
        try:
            self._queue.put_nowait(data)
        except Full:
            LOGGER.warning("Oscilloscope queue full - packet dropped")

    def _ensure_open(self):
        if not self._is_open:
            raise RuntimeError(f"{type(self)} need to be opened first")

    def _set_scene(self):
        self._scene = dvz.dvz_scene()
        if not self._scene:
            raise RuntimeError("dvz_scene() failed")

        self._figure = dvz.dvz_figure(self._scene, self._size[0], self._size[1], 0)
        if not self._figure:
            raise RuntimeError("dvz_figure() failed")

    def _release_scene(self):
        if self._figure:
            dvz.dvz_figure_destroy(self._figure)
            self._figure = None

        if self._scene:
            dvz.dvz_scene_destroy(self._scene)
            self._scene = None

    def _reset(self):
        # TODO move to separated functions
        if self._view:
            dvz.dvz_view_set_frame_callback(self._view, None, None)
            self._view = None

        if self._app:
            dvz.dvz_app_stop(self._app)
            dvz.dvz_app_destroy(self._app)
            self._app = None

        self._callback = None

        self._panels.clear()
        self._index.clear()

    def _ensure_panels(self, data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]):
        for key, (_x, _Y, labels) in data:
            if key in self._panels:
                continue

            if self.split_traces:
                for label in labels:
                    self._add_panel(name=f"{key}[{label}]")
            else:
                self._add_panel(name=key)

        self._layout_panels()

    def _add_panel(self, name: str):
        if len(self._panels.keys()) >= self.max_panels:
            LOGGER.warning(
                "Maximum number of panels (%d) reached; ignoring %r",
                self.max_panels,
                name,
            )
            return

        self._panels[name] = Panel2D(name=name)

    def _layout_panels(self):
        n = len(self._panels.keys())
        if n == 0:
            return

        gutter_px = self.gutter[1]
        figure_height = max(self._size[1], 1)

        gutter = gutter_px / figure_height

        available = 1.0 - gutter * (n - 1)
        panel_height = available / n

        if panel_height <= 0:
            raise RuntimeError(f"Too many panels ({n}) for vertical gutter {gutter_px}px")

        for i, (name, panel) in enumerate(self._panels.items()):
            y = 1.0 - panel_height - i * (panel_height + gutter)
            panel.resize(x=0.0, y=y, width=1.0, height=panel_height)

    def _get_data_from_queue(self) -> list[dict[str, np.ndarray]]:
        data = []
        while self._queue.not_empty:
            data.append(self._queue.get_nowait())
        return data

    def _parse_data(
        self, data: list[dict[str, np.ndarray]]
    ) -> list[dict[str, tuple[np.ndarray, np.ndarray, list[str]] | None]]:
        parsed: list[dict[str, tuple[np.ndarray, np.ndarray, list[str]] | None]] = {}

        for d in data:
            for name, arr in d.items():
                parsed.append({name: self._parse(name, arr)})

        return parsed

    def _merge_data(
        self, data: list[dict[str, tuple[np.ndarray, np.ndarray, list[str]] | None]]
    ) -> dict[str, list[tuple[np.ndarray, np.ndarray, list[str]]]]:
        merged: dict[
            str,
            list[tuple[np.ndarray, np.ndarray, list[str]]],
        ] = {}

        for d in data:
            for name, arr in d.items():
                if arr is not None:
                    merged.setdefault(name, []).append(arr)

        return merged

    def _ingest(self, merged):

        for key, items in merged.items():
            labels = items[0][2]

            items = [item for item in items if item[2] == labels]

            if not items:
                continue

            if len(items) == 1:
                x, Y = items[0][0], items[0][1]
            else:
                x = np.concatenate([item[0] for item in items])
                Y = np.concatenate([item[1] for item in items])

            if x.size > 1 and (np.diff(x) < 0).any():
                order = np.argsort(x, kind="stable")
                x, Y = x[order], Y[order]

        n_bins = self._n_bins
        g = np.floor(x * self._inv_w).astype(np.int64)  # global bin id of every sample
        starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
        gu = g[starts]
        mn = np.fmin.reduceat(Y, starts, axis=0)  # NaN-ignoring; (runs, m)
        mx = np.fmax.reduceat(Y, starts, axis=0)

        first = int(np.searchsorted(gu, gu[-1] - n_bins, side="right"))  # drop > 1 sweep old
        gu, mn, mx = gu[first:], mn[first:], mx[first:]
        slots = gu % n_bins  # unique within one sweep

        x_last = float(x[-1])
        for j, label in enumerate(labels):
            trace = self._index.get((key, label))
            if trace is None:
                continue
            old = trace.gid[slots]
            newer = gu > old  # fresh sweep -> overwrite
            same = gu == old  # same bin continued across chunks -> merge
            s = slots[newer]
            trace.gid[s], trace.mn[s], trace.mx[s] = (
                gu[newer],
                mn[newer, j],
                mx[newer, j],
            )
            s = slots[same]
            trace.mn[s] = np.fmin(trace.mn[s], mn[same, j])
            trace.mx[s] = np.fmax(trace.mx[s], mx[same, j])
            trace.g_latest = max(trace.g_latest, int(gu[-1]))
            trace.dirty = True
            trace.panel.dirty = True
            trace.panel.x_latest = max(trace.panel.x_latest, x_last)

    def _frame_callback(self, *_args) -> None:
        try:
            now = time.monotonic_ns()
            if now - self._last_upload >= 1.0 / self.fps:
                data = self._get_data_from_queue()
                self._ensure_panels(data)

                data = self._parse_data(data)
                data = self._merge_data(data)

                self._ingest(data)

                self._last_upload = now
        except Exception:
            LOGGER.exception("Oscilloscope frame callback failed")

    def _run_dvz_app(self):
        try:
            dvz.dvz_app_run(self._app, 0)
        except Exception as exc:
            LOGGER.exception(exc)
        finally:
            self._reset()

    @staticmethod
    def _parse(name: str, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """Validate one dataset -> (x float64 (n,), Y float32 (n, m), labels). Always copies."""
        a = np.asarray(arr)
        if a.ndim == 0:
            a = a.reshape(1)
        if a.dtype.names:
            if a.ndim != 1 or len(a.dtype.names) < 2:
                raise ValueError(f"{name!r}: structured array needs 1-D data and >= 2 fields")
            fields = a.dtype.names
            x = a[fields[0]].astype(np.float64)
            Y = np.column_stack([a[f].astype(np.float32) for f in fields[1:]])
            labels = list(fields[1:])
        else:
            if a.ndim != 2 or a.shape[1] < 2:
                raise ValueError(f"{name!r}: expected shape (n, 1 + m) with x in column 0")
            x = a[:, 0].astype(np.float64)
            Y = a[:, 1:].astype(np.float32)
            labels = [str(i) for i in range(1, a.shape[1])]
        if x.size == 0:
            return None

        finite_x = np.isfinite(x)
        if not finite_x.all():
            x, Y = x[finite_x], Y[finite_x]
            if x.size == 0:
                return None

        Y[~np.isfinite(Y)] = np.nan  # NaN/inf become gaps, never GPU garbage
        if x.size > 1 and (np.diff(x) < 0).any():
            order = np.argsort(x, kind="stable")
            x, Y = x[order], Y[order]

        return x, Y, labels

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    fs, block = 1000.0, 20
    rng = np.random.default_rng(0)
    pair_dtype = np.dtype([("t", "f8"), ("a", "f4"), ("b", "f4")])

    with Oscilloscope(split_traces=False) as scope:  # try split_traces=False
        t = 0.0
        try:
            while scope.is_open:
                ts = t + np.arange(block) / fs
                sine = np.column_stack(
                    [ts, np.sin(2 * np.pi * ts) + 0.05 * rng.standard_normal(block)]
                )
                pair = np.empty(block, dtype=pair_dtype)  # structured: x then two traces
                pair["t"], pair["a"], pair["b"] = (
                    ts,
                    np.cos(2 * np.pi * 3 * ts),
                    np.sin(2 * np.pi * 5 * ts),
                )
                scope.add_data({"sine": sine, "pair": pair})
                t += block / fs
                time.sleep(block / fs)
        except KeyboardInterrupt:
            pass
