
import threading
import time
from collections.abc import Callable
from logging import getLogger
from queue import Empty, Full, Queue
from typing import Any, Literal

import dearpygui.dearpygui as dpg
import numpy as np
from darkdetect import isDark
from pydantic import BaseModel, Field, PrivateAttr
from screeninfo import ScreenInfoError, get_monitors

LOGGER = getLogger(__name__)
RGBA = tuple[int, int, int, int]

BG_DARK: RGBA = (20, 20, 20, 255)
BG_WHITE: RGBA = (240, 240, 240, 255)
CURSOR_COLOR: RGBA = (250, 183, 3, 255)
PALETTE_DARK: list[RGBA] = [
    (94, 213, 220, 255),
    (255, 103, 129, 255),
    (132, 224, 210, 255),
    (180, 160, 255, 255),
    (255, 179, 71, 255),
    (160, 220, 100, 255),
]
PALETTE_WHITE: list[RGBA] = [
    (20, 90, 160, 255),
    (190, 40, 80, 255),
    (20, 130, 110, 255),
    (110, 70, 190, 255),
    (200, 100, 20, 255),
    (80, 140, 20, 255),
]

_INVALID = -(1 << 62)  # "no data" marker for int64 bin ids
_MAX_CONSECUTIVE_FRAME_ERRORS = 5
_THEMES: dict[tuple[RGBA, int, float], int] = {}  # per-context cache of series themes


def _series_theme(color: RGBA, series_type: int, weight: float = 1.5) -> int:
    key = (color, series_type, weight)
    if key not in _THEMES:
        with dpg.theme() as theme:
            with dpg.theme_component(series_type):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, weight, category=dpg.mvThemeCat_Plots
                )
        _THEMES[key] = theme
    return _THEMES[key]


def _global_theme(dark: bool) -> int:
    """Window padding 0 + dark/white colors for window, text and plots."""
    bg = BG_DARK if dark else BG_WHITE
    fg = (225, 225, 225, 255) if dark else (20, 20, 20, 255)
    plot_bg = (28, 28, 28, 255) if dark else (255, 255, 255, 255)
    grid = (255, 255, 255, 40) if dark else (0, 0, 0, 40)
    legend_bg = (20, 20, 20, 220) if dark else (255, 255, 255, 230)
    border = (110, 110, 110, 255)

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, bg, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, fg, category=dpg.mvThemeCat_Core)
        with dpg.theme_component(dpg.mvPlot):
            for name, value in (
                ("mvPlotCol_FrameBg", bg),
                ("mvPlotCol_PlotBg", plot_bg),
                ("mvPlotCol_PlotBorder", border),
                ("mvPlotCol_LegendBg", legend_bg),
                ("mvPlotCol_LegendBorder", border),
                ("mvPlotCol_LegendText", fg),
                ("mvPlotCol_TitleText", fg),
                ("mvPlotCol_InlayText", fg),
                ("mvPlotCol_AxisText", fg),
                ("mvPlotCol_AxisGrid", grid),
            ):
                dpg.add_theme_color(getattr(dpg, name), value, category=dpg.mvThemeCat_Plots)
    return theme


class Trace(BaseModel):
    """One curve: a ring of x-bins holding (min, max), drawn as a line series."""

    key: str = Field(...)
    label: str = Field(...)
    parent: Any = Field(..., description="Dear PyGui y-axis the series is added to")
    color: RGBA = Field(...)
    duration: float = Field(..., gt=0, description="Sweep length in x units")
    n_bins: int = Field(..., ge=8)

    _series: Any = PrivateAttr(default=None)
    _gid: np.ndarray = PrivateAttr(default=None)  # (n_bins,) int64 global bin id per slot
    _mn: np.ndarray = PrivateAttr(default=None)  # (n_bins,) float32
    _mx: np.ndarray = PrivateAttr(default=None)
    _bin_w: float = PrivateAttr(default=1.0)
    _g_latest: int = PrivateAttr(default=_INVALID)
    _x_latest: float = PrivateAttr(default=float("-inf"))
    _dirty: bool = PrivateAttr(default=False)

    def model_post_init(self, context):
        n = self.n_bins
        self._bin_w = self.duration / n
        self._gid = np.full(n, _INVALID, dtype=np.int64)
        self._mn = np.full(n, np.nan, dtype=np.float32)
        self._mx = np.full(n, np.nan, dtype=np.float32)
        self._series = dpg.add_line_series([], [], label=self.label, parent=self.parent)
        dpg.bind_item_theme(self._series, _series_theme(self.color, dpg.mvLineSeries))

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def x_latest(self) -> float:
        return self._x_latest

    def set_data(self, data: np.ndarray):
        """``data``: structured array with a float ``"ts"`` field (sorted) and ``self.label``."""
        names = data.dtype.names
        if not names or "ts" not in names or self.label not in names:
            raise RuntimeError(f"Data needs 'ts' and {self.label!r} fields")
        if data.size == 0:
            return

        x = np.asarray(data["ts"], dtype=np.float64)
        y = np.asarray(data[self.label], dtype=np.float32)
        n = self.n_bins

        # reduce the chunk to one (min, max) per x-bin
        g = np.floor(x / self._bin_w).astype(np.int64)  # global bin id of every sample
        starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
        gu = g[starts]
        mn = np.fmin.reduceat(y, starts)  # NaN-ignoring
        mx = np.fmax.reduceat(y, starts)

        first = int(np.searchsorted(gu, gu[-1] - n, side="right"))  # drop > 1 sweep old
        gu, mn, mx = gu[first:], mn[first:], mx[first:]
        slots = gu % n  # unique within one sweep

        old = self._gid[slots]
        newer = gu > old  # new sweep over this slot -> overwrite
        same = gu == old  # same bin continued across chunks -> merge
        s = slots[newer]
        self._gid[s], self._mn[s], self._mx[s] = gu[newer], mn[newer], mx[newer]
        s = slots[same]
        self._mn[s] = np.fmin(self._mn[s], mn[same])
        self._mx[s] = np.fmax(self._mx[s], mx[same])

        self._g_latest = max(self._g_latest, int(gu[-1]))
        self._x_latest = max(self._x_latest, float(x[-1]))
        self._dirty = True

    def update(self):
        """Push the current sweep to the series (render thread only)."""
        px, py = self._build_xy()
        dpg.set_value(self._series, [px, py])
        self._dirty = False

    def _build_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Valid bins in x order as (min, max) point pairs; NaN breaks the line at the cursor."""
        n = self.n_bins
        valid = (
            (self._gid > self._g_latest - n)  # newer than one sweep
            & (self._gid > _INVALID)
            & np.isfinite(self._mn)
            & np.isfinite(self._mx)
        )
        idx = np.flatnonzero(valid)
        k = idx.size
        if k == 0:
            return np.empty(0), np.empty(0)

        w = self._bin_w
        px = np.empty(2 * k, np.float64)
        py = np.empty(2 * k, np.float64)
        px[0::2], px[1::2] = (idx + 0.25) * w, (idx + 0.75) * w
        py[0::2], py[1::2] = self._mn[idx], self._mx[idx]

        # Bins left of the cursor are new, bins right of it belong to the previous sweep:
        # insert a NaN between them so ImPlot draws a gap instead of a connecting line.
        j = int(np.searchsorted(idx, self._g_latest % n, side="right"))
        if 0 < j < k:
            cut = 2 * j
            px = np.insert(px, cut, px[cut - 1])
            py = np.insert(py, cut, np.nan)
        return px, py


class Panel2D(BaseModel):
    name: str = Field(...)
    parent: Any = Field(..., description="Dear PyGui subplots container")

    dark: bool = Field(default=True)
    x_lim: tuple[float, float] = Field(default=(0.0, 1.0))
    y_lim: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = auto-fit."
    )
    x_label: str | None = Field(default=None)
    max_points: int = Field(default=2000, ge=16, description="Display points per trace")
    cursor: bool = Field(default=True)
    grid: bool = Field(default=True)
    legend: bool = Field(default=True)
    static: bool = Field(default=False, description="True disables mouse interaction")

    _plot: Any = PrivateAttr(default=None)
    _x_axis: Any = PrivateAttr(default=None)
    _y_axis: Any = PrivateAttr(default=None)
    _cursor: Any = PrivateAttr(default=None)
    _traces: dict[str, Trace] = PrivateAttr(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.x_lim[1] - self.x_lim[0]

    @property
    def traces(self) -> dict[str, Trace]:
        return self._traces

    def model_post_init(self, context):
        self._plot = dpg.add_plot(
            label=self.name, parent=self.parent, width=-1, height=-1, no_inputs=self.static
        )
        self._x_axis = dpg.add_plot_axis(
            dpg.mvXAxis, label=self.x_label or "", parent=self._plot, no_gridlines=not self.grid
        )
        self._y_axis = dpg.add_plot_axis(
            dpg.mvYAxis,
            parent=self._plot,
            no_gridlines=not self.grid,
            auto_fit=self.y_lim is None,
        )
        dpg.set_axis_limits(self._x_axis, *self.x_lim)  # the sweep window is fixed
        if self.y_lim is not None:
            dpg.set_axis_limits(self._y_axis, *self.y_lim)
        if self.legend:
            dpg.add_plot_legend(parent=self._plot)
        if self.cursor:
            # "##" hides the entry from the legend
            self._cursor = dpg.add_inf_line_series(
                [self.x_lim[0]], label="##cursor", parent=self._y_axis
            )
            dpg.bind_item_theme(
                self._cursor, _series_theme(CURSOR_COLOR, dpg.mvInfLineSeries, weight=1.0)
            )

    def set_data(self, data: np.ndarray, labels: list[str] | None = None):
        """Feed a structured array (``"ts"`` + trace fields); creates traces on first sight."""
        if not data.dtype.names or "ts" not in data.dtype.names:
            raise RuntimeError("Data must be a structured array with a 'ts' field")

        for label in labels if labels is not None else data.dtype.names[1:]:
            trace = self._traces.get(label) or self._create_trace(label)
            trace.set_data(data)

    def update(self):
        """Redraw dirty traces and move the cursor (render thread only)."""
        dirty = [t for t in self._traces.values() if t.dirty]
        if not dirty:
            return
        for trace in dirty:
            trace.update()
        if self.cursor:
            x_latest = max(t.x_latest for t in self._traces.values())
            if np.isfinite(x_latest):
                cx = self.x_lim[0] + (x_latest - self.x_lim[0]) % self.duration
                dpg.set_value(self._cursor, [[cx]])

    def _create_trace(self, label: str) -> Trace:
        palette = PALETTE_DARK if self.dark else PALETTE_WHITE
        trace = Trace(
            key=self.name,
            label=label,
            parent=self._y_axis,
            color=palette[len(self._traces) % len(palette)],
            duration=self.duration,
            n_bins=max(8, self.max_points // 2),
        )
        self._traces[label] = trace
        if self._cursor is not None:
            dpg.move_item(self._cursor, parent=self._y_axis)  # keep the cursor drawn last
        return trace


class Oscilloscope(BaseModel):
    title: str = Field(default="Oscilloscope")
    duration: float = Field(default=20.0, gt=0, description="Sweep length in x units (seconds)")
    max_points: int = Field(
        default=2000, ge=16, description="Display points per trace per duration"
    )
    max_panels: int = Field(default=64, gt=0, description="Safety cap on panel count")
    split_traces: bool = Field(
        default=True,
        description="True: one panel per trace. False: one panel per dataset (key).",
    )
    legend: bool = Field(
        default=True, description="Name the traces of each panel (only when split_traces=False)"
    )
    y_lim: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = auto-fit per panel."
    )
    size: Literal["auto"] | tuple[int, int] = Field(default="auto")
    theme: Literal["auto", "dark", "white"] = Field(default="auto")
    queue_size: int = Field(default=1000, gt=0, description="Size of the internal queue")

    _size: tuple[int, int] = PrivateAttr(default=(1280, 720))
    _dark: bool = PrivateAttr(default=True)
    _window: Any = PrivateAttr(default=None)
    _placeholder: Any = PrivateAttr(default=None)
    _subplots: Any = PrivateAttr(default=None)
    _panels: dict[str, Panel2D] = PrivateAttr(default_factory=dict)
    _rejected: set[str] = PrivateAttr(default_factory=set)

    _queue: Queue = PrivateAttr()
    _thread: threading.Thread | None = PrivateAttr(default=None)
    _ready: threading.Event = PrivateAttr(default_factory=threading.Event)
    _stop: threading.Event = PrivateAttr(default_factory=threading.Event)
    _error: BaseException | None = PrivateAttr(default=None)
    _started: bool = PrivateAttr(default=False)
    _active: bool = PrivateAttr(default=False)
    _frame_errors: int = PrivateAttr(default=0)
    _dropped: int = PrivateAttr(default=0)
    _last_drop_log: float = PrivateAttr(default=0.0)

    def model_post_init(self, context):
        if self.size == "auto":
            try:
                monitors = get_monitors()
                monitor = next(
                    (m for m in monitors if getattr(m, "is_primary", False)), monitors[0]
                )
                self._size = (int(monitor.width * 0.9), int(monitor.height * 0.9))
            except (ScreenInfoError, IndexError):
                pass
        else:
            self._size = self.size

        self._dark = self.theme == "dark" or (self.theme == "auto" and bool(isDark()))
        self._queue = Queue(maxsize=self.queue_size)


    @property
    def is_open(self) -> bool:
        return self._active and not self._stop.is_set()

    def open(self, timeout: float = 15.0):
        if self.is_open:
            return

        self._reset()
        self._thread = threading.Thread(
            target=self._run, name=f"{type(self).__name__}_thread", daemon=True
        )
        self._thread.start()
       
        if not self._ready.wait(timeout):
            raise TimeoutError(f"{type(self).__name__} did not start within {timeout}s")
       
        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed to start") from self._error

    def run(self, producer: Callable[["Oscilloscope"], None] | None = None):
        """Blocking alternative to :meth:`open`: renders on the CALLING (main) thread.

        ``producer(scope)`` optionally runs in a background thread; loop on
        ``scope.is_open`` inside it and feed ``scope.add_data``.
        """
        self._reset()
        worker = None
        if producer is not None:
            worker = threading.Thread(
                target=producer, args=(self,), name="OscilloscopeProducer", daemon=True
            )
            worker.start()
        try:
            self._run()
        finally:
            self._stop.set()
            if worker is not None:
                worker.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed") from self._error

    def close(self, timeout: float = 5.0):
        """Ask the render loop to exit (it ends within one frame) and wait for it."""
        self._stop.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            LOGGER.warning("Render thread still running after %.1fs", timeout)
        else:
            self._thread = None

    def add_data(self, data: dict[str, np.ndarray]) -> bool:
        """Queue new samples (thread-safe, non-blocking). See the module docstring.

        Returns False (without raising) once the window has been closed, so a producer
        loop can simply stop; raises only for bad input or if never started.
        """
        if not self._started:
            raise RuntimeError(f"{type(self).__name__} needs to be opened first")

        packet: dict[str, np.ndarray] = {}
        for name, arr in data.items():
            normalized = self._normalize(name, arr)
            if normalized is not None:
                packet[name] = normalized

        if self._stop.is_set():
            return False
        if not packet:
            return True
        try:
            self._queue.put_nowait(packet)
        except Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 1.0:
                LOGGER.warning("Oscilloscope queue full - dropped %d packet(s)", self._dropped)
                self._last_drop_log = now
        return True

    # ------------------------------------------------------------ input parsing

    @staticmethod
    def _normalize(name: str, arr: Any) -> np.ndarray | None:
        """-> structured array [("ts", f8), (label, f4), ...], sorted, finite x. Always copies."""
        a = np.asarray(arr)
        if a.ndim == 0:
            a = a.reshape(1)

        if a.dtype.names:  # structured: first field is x
            if a.ndim != 1 or len(a.dtype.names) < 2:
                raise ValueError(f"{name!r}: structured array needs 1-D data and >= 2 fields")
            fields = a.dtype.names
            x = a[fields[0]].astype(np.float64)
            labels = list(fields[1:])
            Y = np.column_stack([a[f].astype(np.float32) for f in labels])
        else:  # plain: column 0 is x
            if a.ndim != 2 or a.shape[1] < 2:
                raise ValueError(f"{name!r}: expected shape (n, 1 + m) with x in column 0")
            x = a[:, 0].astype(np.float64)
            Y = a[:, 1:].astype(np.float32)
            labels = [str(i) for i in range(1, a.shape[1])]

        if "ts" in labels:
            raise ValueError(f"{name!r}: 'ts' is reserved for the x field")
        if x.size == 0:
            return None

        ok = np.isfinite(x)
        if not ok.all():
            x, Y = x[ok], Y[ok]
            if x.size == 0:
                return None
        Y[~np.isfinite(Y)] = np.nan  # NaN/inf become gaps
        if x.size > 1 and (np.diff(x) < 0).any():
            order = np.argsort(x, kind="stable")
            x, Y = x[order], Y[order]

        out = np.empty(x.size, dtype=[("ts", "f8")] + [(l, "f4") for l in labels])
        out["ts"] = x
        for j, label in enumerate(labels):
            out[label] = Y[:, j]
        return out

    # ---------------------------------------------------------------- UI building

    def _reset(self):
        self._stop.clear()
        self._ready.clear()
        self._started = True
        self._active = True
        self._error = None
        self._frame_errors = 0
        self._rejected.clear()
        while True:  # discard stale packets from a previous session
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _build_ui(self):
        dpg.bind_theme(_global_theme(self._dark))
        self._window = dpg.add_window(
            no_title_bar=True, no_scrollbar=True, no_move=True, no_resize=True, no_collapse=True
        )
        self._placeholder = dpg.add_text("Waiting for data...", parent=self._window, indent=12)

    def _add_panel(self, name: str) -> bool:
        if len(self._panels) >= self.max_panels:
            if name not in self._rejected:
                self._rejected.add(name)
                LOGGER.warning(
                    "Maximum number of panels (%d) reached; ignoring %r", self.max_panels, name
                )
            return False

        rows = len(self._panels) + 1
        if self._subplots is None:
            if self._placeholder is not None:
                dpg.delete_item(self._placeholder)
                self._placeholder = None
            self._subplots = dpg.add_subplots(
                rows, 1, parent=self._window, width=-1, height=-1, link_all_x=True
            )
        else:
            dpg.configure_item(self._subplots, rows=rows)  # one more row, equal heights

        self._panels[name] = Panel2D(
            name=name,
            parent=self._subplots,
            dark=self._dark,
            x_lim=(0.0, self.duration),
            y_lim=self.y_lim,
            x_label="Time (s)",
            max_points=self.max_points,
            legend=self.legend and not self.split_traces,
            grid=True,  # TODO: make this configurable
            static=False,  # TODO: make this configurable
        )
        return True

    # -------------------------------------------------------------- frame logic

    def _panel_names(self, key: str, labels: tuple[str, ...]) -> list[tuple[str, list[str]]]:
        """-> [(panel_name, [labels shown in it]), ...] for one dataset."""
        if not self.split_traces or len(labels) == 1:
            return [(key, list(labels))]
        return [(f"{key}[{label}]", [label]) for label in labels]

    def _drain_queue(self) -> dict[str, np.ndarray]:
        chunks: dict[str, list[np.ndarray]] = {}
        for _ in range(self._queue.qsize()):  # bounded: ignore packets arriving meanwhile
            try:
                packet = self._queue.get_nowait()
            except Empty:
                break
            for key, arr in packet.items():
                chunks.setdefault(key, []).append(arr)

        merged: dict[str, np.ndarray] = {}
        for key, items in chunks.items():
            dtype = items[0].dtype
            items = [a for a in items if a.dtype == dtype]  # skip packets whose columns changed
            arr = items[0] if len(items) == 1 else np.concatenate(items)
            if arr.size > 1 and (np.diff(arr["ts"]) < 0).any():
                arr = arr[np.argsort(arr["ts"], kind="stable")]
            merged[key] = arr
        return merged

    def _ensure_panels(self, data: dict[str, np.ndarray]):
        for key, arr in data.items():
            for name, _labels in self._panel_names(key, arr.dtype.names[1:]):
                if name not in self._panels and name not in self._rejected:
                    self._add_panel(name)

    def _ingest(self, data: dict[str, np.ndarray]):
        for key, arr in data.items():
            for name, labels in self._panel_names(key, arr.dtype.names[1:]):
                panel = self._panels.get(name)
                if panel is not None:
                    panel.set_data(arr, labels=labels)

    def _frame(self):
        """One iteration of the render loop (render thread only)."""
        try:
            data = self._drain_queue()
            if data:
                self._ensure_panels(data)
                self._ingest(data)
            for panel in self._panels.values():
                panel.update()
            self._frame_errors = 0
        except Exception:
            LOGGER.exception("Oscilloscope frame failed")
            self._frame_errors += 1
            if self._frame_errors >= _MAX_CONSECUTIVE_FRAME_ERRORS:
                LOGGER.error("Too many consecutive frame errors; stopping")
                self._stop.set()

    # ---------------------------------------------------------------- lifecycle

    def _run(self):
        context_created = False
        try:
            dpg.create_context()
            context_created = True
            _THEMES.clear()  # theme ids belong to one context; never reuse across sessions
            self._build_ui()
            dpg.create_viewport(
                title=self.title,
                width=self._size[0],
                height=self._size[1],
                clear_color=BG_DARK if self._dark else BG_WHITE,
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.set_primary_window(self._window, True)
            self._ready.set()

            # Ends when the window is closed or close() is called.
            while dpg.is_dearpygui_running() and not self._stop.is_set():
                self._frame()
                dpg.render_dearpygui_frame()
        except Exception as exc:
            self._error = exc
            LOGGER.exception(exc)
        finally:
            self._stop.set()  # is_open turns False immediately; add_data starts returning False
            self._active = False
            self._panels.clear()
            self._subplots = self._window = self._placeholder = None
            _THEMES.clear()
            if context_created:
                try:
                    dpg.destroy_context()
                except Exception:
                    LOGGER.exception("dpg.destroy_context() failed")
            self._ready.set()  # never leave open() waiting

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
