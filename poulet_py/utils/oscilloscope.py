try:
    from collections.abc import Callable
    from queue import Empty, Full, Queue
    from threading import Event, Thread, current_thread
    from time import monotonic_ns
    from types import TracebackType
    from typing import Any, Literal, Self

    from darkdetect import isDark
    from dearpygui.dearpygui import (
        add_inf_line_series,
        add_line_series,
        add_plot,
        add_plot_axis,
        add_plot_legend,
        add_subplots,
        add_text,
        add_theme_color,
        add_theme_style,
        add_window,
        bind_item_theme,
        bind_theme,
        configure_item,
        create_context,
        create_viewport,
        delete_item,
        destroy_context,
        get_axis_limits,
        get_item_rect_min,
        get_item_rect_size,
        get_text_size,
        is_dearpygui_running,
        is_item_hovered,
        is_key_down,
        is_mouse_button_double_clicked,
        is_mouse_button_down,
        move_item,
        mvAll,
        mvInfLineSeries,
        mvKey_R,
        mvKey_Spacebar,
        mvLineSeries,
        mvMouseButton_Left,
        mvPlot,
        mvPlot_Location_NorthEast,
        mvPlotCol_AxisGrid,
        mvPlotCol_AxisText,
        mvPlotCol_FrameBg,
        mvPlotCol_InlayText,
        mvPlotCol_LegendBg,
        mvPlotCol_LegendBorder,
        mvPlotCol_LegendText,
        mvPlotCol_Line,
        mvPlotCol_PlotBg,
        mvPlotCol_PlotBorder,
        mvPlotCol_TitleText,
        mvPlotStyleVar_LineWeight,
        mvStyleVar_WindowPadding,
        mvThemeCat_Core,
        mvThemeCat_Plots,
        mvThemeCol_Text,
        mvThemeCol_WindowBg,
        mvXAxis,
        mvYAxis,
        render_dearpygui_frame,
        set_axis_limits,
        set_axis_limits_auto,
        set_item_pos,
        set_primary_window,
        set_value,
        set_viewport_title,
        setup_dearpygui,
        show_viewport,
        theme,
        theme_component,
        tooltip,
    )
    from numpy import (
        argsort,
        asarray,
        column_stack,
        concatenate,
        diff,
        empty,
        flatnonzero,
        floor,
        fmax,
        fmin,
        full,
        insert,
        isfinite,
        nan,
        ndarray,
        searchsorted,
    )
    from pydantic import BaseModel, Field, PrivateAttr
    from screeninfo import ScreenInfoError, get_monitors
    from screeninfo.common import Monitor

    from poulet_py.config.logging import LOGGER
except ImportError as e:
    raise ImportError("""
Missing 'osc' module. Install options:
- Dedicated:    pip install poulet_py[osc]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e

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
_LOCK_FRAMES = 2  # frames an axis stays locked after set_axis_limits before the user gets it back
_THEMES: dict[tuple[RGBA, int, float], int] = {}  # per-context cache of series themes


def _series_theme(color: RGBA, series_type: int, weight: float = 1.5) -> int:
    key = (color, series_type, weight)
    if key not in _THEMES:
        with theme() as th:
            with theme_component(series_type):
                add_theme_color(mvPlotCol_Line, color, category=mvThemeCat_Plots)
                add_theme_style(mvPlotStyleVar_LineWeight, weight, category=mvThemeCat_Plots)
        _THEMES[key] = th
    return _THEMES[key]


def _global_theme(dark: bool) -> int:
    """Window padding 0 + dark/white colors for window, text and plots."""
    bg = BG_DARK if dark else BG_WHITE
    fg = (225, 225, 225, 255) if dark else (20, 20, 20, 255)
    plot_bg = (28, 28, 28, 255) if dark else (255, 255, 255, 255)
    grid = (255, 255, 255, 40) if dark else (0, 0, 0, 40)
    legend_bg = (20, 20, 20, 220) if dark else (255, 255, 255, 230)
    border = (110, 110, 110, 255)

    with theme() as th:
        with theme_component(mvAll):
            add_theme_style(mvStyleVar_WindowPadding, 0, 0, category=mvThemeCat_Core)
            add_theme_color(mvThemeCol_WindowBg, bg, category=mvThemeCat_Core)
            add_theme_color(mvThemeCol_Text, fg, category=mvThemeCat_Core)
        with theme_component(mvPlot):
            for func, value in (
                (mvPlotCol_FrameBg, bg),
                (mvPlotCol_PlotBg, plot_bg),
                (mvPlotCol_PlotBorder, border),
                (mvPlotCol_LegendBg, legend_bg),
                (mvPlotCol_LegendBorder, border),
                (mvPlotCol_LegendText, fg),
                (mvPlotCol_TitleText, fg),
                (mvPlotCol_InlayText, fg),
                (mvPlotCol_AxisText, fg),
                (mvPlotCol_AxisGrid, grid),
            ):
                add_theme_color(func, value, category=mvThemeCat_Plots)
    return th


class Trace(BaseModel):
    # TODO one buffer linear downscale?
    duration: float = Field(..., gt=0, description="Sweep length in x units")
    n_bins: int = Field(..., ge=8)

    _gid: ndarray = PrivateAttr()
    _mn: ndarray = PrivateAttr()
    _mx: ndarray = PrivateAttr()
    _bin_w: float = PrivateAttr(default=1.0)
    _g_latest: int = PrivateAttr(default=_INVALID)
    _x_latest: float = PrivateAttr(default=float("-inf"))
    _dirty: bool = PrivateAttr(default=False)
    _bounds: tuple[float, float] | None = PrivateAttr(default=None)
    _hi: float = PrivateAttr(default=float("-inf"))

    def model_post_init(self, context: Any, /) -> None:  # noqa: ARG002
        n = self.n_bins
        self._bin_w = self.duration / n
        self._gid = full(n, _INVALID, dtype="int64")
        self._mn = full(n, nan, dtype="float32")
        self._mx = full(n, nan, dtype="float32")

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def x_latest(self) -> float:
        return self._x_latest

    @property
    def bounds(self) -> tuple[float, float] | None:
        """(min, max) of the plotted values, or None when empty."""
        return self._bounds

    def add(self, x: ndarray, y: ndarray) -> None:
        """``x``: sorted float64 samples, ``y``: matching float32 values (NaN = gap)."""
        if x.size == 0:
            return
        n = self.n_bins
        g = floor(x / self._bin_w).astype("int64")  # global bin id of every sample
        starts = flatnonzero(concatenate(([True], g[1:] != g[:-1])))
        gu = g[starts]
        mn = fmin.reduceat(y, starts)  # NaN-ignoring (min, max) per bin
        mx = fmax.reduceat(y, starts)

        first = int(searchsorted(gu, gu[-1] - n, side="right"))  # drop > 1 sweep old
        gu, mn, mx = gu[first:], mn[first:], mx[first:]
        slots = gu % n  # unique within one sweep

        old = self._gid[slots]
        newer = gu > old  # new sweep over this slot -> overwrite
        same = gu == old  # same bin continued across chunks -> merge
        s = slots[newer]
        self._gid[s], self._mn[s], self._mx[s] = gu[newer], mn[newer], mx[newer]
        s = slots[same]
        self._mn[s] = fmin(self._mn[s], mn[same])
        self._mx[s] = fmax(self._mx[s], mx[same])

        self._g_latest = max(self._g_latest, int(gu[-1]))
        self._x_latest = max(self.x_latest, float(x[-1]))
        self._dirty = True

    def xy(self) -> tuple[ndarray, ndarray]:
        """Points to draw: (min, max) pair per valid bin, in x order. Clears ``dirty``.

        Bins left of the newest one are this sweep, bins right of it are the previous sweep;
        a NaN between them makes ImPlot draw a gap instead of a connecting line.
        """
        self._dirty = False
        n = self.n_bins
        valid = (
            (self._gid > self._g_latest - n)  # newer than one sweep
            & (self._gid > _INVALID)
            & isfinite(self._mn)
            & isfinite(self._mx)
        )
        idx = flatnonzero(valid)
        k = idx.size
        if k == 0:
            self._bounds = None
            return empty(0), empty(0)

        w = self._bin_w
        px = empty(2 * k, "float64")
        py = empty(2 * k, "float64")
        px[0::2], px[1::2] = (idx + 0.25) * w, (idx + 0.75) * w
        py[0::2], py[1::2] = self._mn[idx], self._mx[idx]
        self._bounds = (float(py.min()), float(py.max()))

        j = int(searchsorted(idx, self._g_latest % n, side="right"))
        if 0 < j < k:
            cut = 2 * j
            px = insert(px, cut, px[cut - 1])
            py = insert(py, cut, nan)
        return px, py


class SeriesPanel(BaseModel):
    name: str = Field(...)
    parent: int | str = Field(..., description="Dear PyGui subplots container")
    overlay: int | str = Field(..., description="Dear PyGui window")
    duration: float = Field(default=6.0)
    x_label: str | None = Field(default=None)
    legend: bool = Field(default=True)
    show_x_axis: bool = Field(default=True)
    dark: bool = Field(default=True)
    static: bool = Field(default=False, description="True disables mouse interaction")

    y_lim: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = follow the data."
    )
    max_points: int = Field(default=2000, ge=16, description="Display points per trace")
    title_max_chars: int = Field(
        default=24,
        ge=4,
        description="Panel names longer than this are truncated (full name on hover)",
    )

    _plot: int | str | None = PrivateAttr(default=None)
    _x_axis: Any = PrivateAttr(default=None)
    _y_axis: Any = PrivateAttr(default=None)
    _cursor: Any = PrivateAttr(default=None)
    _buffers: dict[str, Trace] = PrivateAttr(default_factory=dict)
    _series: dict[str, int | str] = PrivateAttr(default_factory=dict)
    _x_release: int = PrivateAttr(default=0)
    _y_release: int = PrivateAttr(default=0)
    _follow_y: bool = PrivateAttr(default=True)
    _y_expected: tuple[float, float] | None = PrivateAttr(default=None)
    _bounds: tuple[float, float] | None = PrivateAttr(default=None)
    _title_pos: tuple[int, int] | None = PrivateAttr(default=None)

    @property
    def plot(self) -> Any:
        return self._plot

    @property
    def traces(self) -> dict[str, Trace]:
        return self._buffers

    @property
    def follows_y(self) -> bool:
        return self._follow_y

    def model_post_init(self, context: Any, /) -> None:  # noqa: ARG002
        self._plot = add_plot(
            label=self.name,
            parent=self.parent,
            width=-1,
            height=-1,
            no_inputs=self.static,
            no_title=True,
        )
        self._x_axis = add_plot_axis(
            mvXAxis,
            label=self.x_label or "",
            parent=self.plot,
            no_tick_labels=not self.show_x_axis,
            no_label=not self.show_x_axis,
        )
        self._y_axis = add_plot_axis(mvYAxis, parent=self.plot)

        if self.legend:
            add_plot_legend(parent=self._plot, location=mvPlot_Location_NorthEast, horizontal=True)

        self._cursor = add_inf_line_series([0], label="##cursor", parent=self._y_axis)
        bind_item_theme(self._cursor, _series_theme(CURSOR_COLOR, mvInfLineSeries, weight=1.0))

        self._make_title()
        self.reset_view()

    def feed(self, data: ndarray, labels: list[str]) -> None:
        """``data``: structured array ("ts" + trace fields). Creates curves on first sight."""
        for label in labels:
            if label not in self._buffers:
                self._add_curve(label)
            self._buffers[label].add(
                asarray(data["ts"], dtype="float64"),
                asarray(data[label], dtype="float32"),
            )

    def update(self) -> None:
        """Redraw dirty curves, follow Y and move the cursor."""
        dirty = [label for label, buf in self._buffers.items() if buf.dirty]
        if not dirty:
            return

        for label in dirty:
            px, py = self._buffers[label].xy()
            set_value(self._series[label], [px, py])

        bounds = [b.bounds for b in self._buffers.values() if b.bounds is not None]
        self._bounds = (min(b[0] for b in bounds), max(b[1] for b in bounds)) if bounds else None
        if not (is_item_hovered(self.plot) and is_mouse_button_down(mvMouseButton_Left)):
            self._follow_fit()  # not while the user is dragging this panel

        x_latest = max(b.x_latest for b in self._buffers.values())
        if isfinite(x_latest):
            cx = x_latest % self.duration
            set_value(self._cursor, [[cx]])

    def tick(self) -> None:
        """Every frame: keep the title in place, release locked axes, notice a user Y change."""
        self._place_title()
        if self.static:
            return
        if self._x_release > 0:
            self._x_release -= 1
            if self._x_release == 0:
                set_axis_limits_auto(self._x_axis)
        if self._y_release > 0:
            self._y_release -= 1
            if self._y_release == 0:
                set_axis_limits_auto(self._y_axis)
        elif self._follow_y and self._y_expected is not None:
            lo, hi = get_axis_limits(self._y_axis)
            e_lo, e_hi = self._y_expected
            tol = 1e-3 * max(e_hi - e_lo, 1e-9)
            if abs(lo - e_lo) > tol or abs(hi - e_hi) > tol:
                self._follow_y = False  # the user zoomed/panned Y: stop following

    def reset_view(self) -> None:
        """X back to the full sweep; Y back to following the data (or ``y_lim``)."""
        self._lock_x()
        self._y_release = 0
        self._y_expected = None
        if self.y_lim is not None:
            self._follow_y = False
            self._lock_y(*self.y_lim)
        else:
            self._follow_y = True
            if self._bounds is None:
                self._lock_y(-1.0, 1.0)
            else:
                self._follow_fit(force=True)

    def set_show_x_axis(self, show: bool) -> None:
        """Show/hide the x tick labels and x label (only the bottom panel keeps them)."""
        configure_item(self._x_axis, no_tick_labels=not show, no_label=not show)

    def _lock_x(self) -> None:
        set_axis_limits(self._x_axis, 0, self.duration)
        self._x_release = 0 if self.static else _LOCK_FRAMES

    def _lock_y(self, lo: float, hi: float) -> None:
        set_axis_limits(self._y_axis, lo, hi)
        self._y_expected = (lo, hi)
        self._y_release = 0 if self.static else _LOCK_FRAMES

    def _follow_fit(self, force: bool = False) -> None:
        """Set Y to the data range (+padding) -- only when it no longer fits or is much smaller."""
        if self.y_lim is not None or not self._follow_y or self._bounds is None:
            return
        if self._y_release > 0 and not force:
            return
        lo, hi = self._bounds
        span = hi - lo
        cur = self._y_expected
        # Extra headroom when the data outgrows the view, so a drifting/growing signal
        # does not re-lock (and block mouse input) every few frames.
        grow = cur is not None and (lo < cur[0] or hi > cur[1])
        frac = 0.25 if grow else 0.1
        pad = frac * span if span > 1e-9 else max(frac * abs(hi), 1.0)
        lo, hi = lo - pad, hi + pad
        if (
            force
            or cur is None
            or lo < cur[0]
            or hi > cur[1]
            or (hi - lo) < 0.5 * (cur[1] - cur[0])
        ):
            self._lock_y(lo, hi)

    def _add_curve(self, label: str) -> None:
        palette = PALETTE_DARK if self.dark else PALETTE_WHITE
        color = palette[len(self._buffers) % len(palette)]
        self._buffers[label] = Trace(duration=self.duration, n_bins=max(8, self.max_points // 2))
        series = add_line_series([], [], label=label, parent=self._y_axis)
        bind_item_theme(series, _series_theme(color, mvLineSeries))
        self._series[label] = series
        move_item(self._cursor, parent=self._y_axis)  # keep the cursor drawn last

    def _make_title(self) -> None:
        """Short left-aligned name over the plot; hovering it shows the full name if truncated."""
        limit = self.title_max_chars
        short = self.name if len(self.name) <= limit else self.name[: limit - 3] + "..."
        text_size = get_text_size("0")
        self._char_w = float(text_size[0] if text_size else 7.0)
        self._title = add_text(short, parent=self.overlay, pos=(0, 0))
        if short != self.name:
            with tooltip(self._title):
                add_text(self.name)

    def _place_title(self) -> None:
        """Anchor the title to the plot's top-left, just right of the y tick labels."""
        size = get_item_rect_size(self.plot)
        if size[0] <= 1 or size[1] <= 1:
            return  # not laid out yet
        left, top = get_item_rect_min(self.plot)
        # Width of the y tick labels ("-0.5", "12", "1500"): digits + sign + a decimal for small
        # ranges. Estimated from the limits (ImPlot exposes no inner-rect API), so titles line up.
        lo, hi = get_axis_limits(self._y_axis)
        peak = max(abs(lo), abs(hi))
        digits = len(str(int(peak))) if peak >= 1 else 1
        chars = max(4, digits + (1 if lo < 0 else 0) + (2 if peak < 10 else 0))
        pos: tuple[int, int] = (round(left + chars * self._char_w + 12), round(top + 6))
        if pos != self._title_pos:
            set_item_pos(self._title, list(pos))
            self._title_pos = pos


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
    legend: bool = Field(default=True, description="Name the traces of each panel")
    y_lim: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = auto-fit per panel."
    )
    x_label: str | None = Field(default="Time (s)", description="Shown under the bottom panel only")
    title_max_chars: int = Field(
        default=24,
        ge=4,
        description="Panel names longer than this are truncated (full name on hover)",
    )
    interactive: bool = Field(default=True, description="Mouse zoom/pan/reset")
    size: Literal["auto"] | tuple[int, int] = Field(default="auto")
    theme: Literal["auto", "dark", "white"] = Field(default="auto")
    queue_size: int = Field(default=1000, gt=0, description="Size of the internal queue")

    _size: tuple[int, int] = PrivateAttr(default=(1280, 720))
    _dark: bool = PrivateAttr(default=True)
    _window: Any = PrivateAttr(default=None)
    _placeholder: Any = PrivateAttr(default=None)
    _subplots: int | str | None = PrivateAttr(default=None)
    _panels: dict[str, SeriesPanel] = PrivateAttr(default_factory=dict)
    _rejected: set[str] = PrivateAttr(default_factory=set)

    _osc_queue: Queue[tuple[int, dict[str, ndarray]]] = PrivateAttr()
    _osc_thread: Thread | None = PrivateAttr(default=None)
    _ready: Event = PrivateAttr(default_factory=Event)
    _stop: Event = PrivateAttr(default_factory=Event)
    _error: BaseException | None = PrivateAttr(default=None)
    _started: bool = PrivateAttr(default=False)
    _active: bool = PrivateAttr(default=False)
    _paused: bool = PrivateAttr(default=False)
    _title_paused: bool = PrivateAttr(default=False)
    _reset_requested: Event = PrivateAttr(default_factory=Event)
    _keys: dict[str, bool] = PrivateAttr(default_factory=dict)
    _frame_errors: int = PrivateAttr(default=0)
    _dropped: int = PrivateAttr(default=0)
    _last_drop_log: float = PrivateAttr(default=0.0)

    def model_post_init(self, context: Any, /) -> None:  # noqa: ARG002
        if self.size == "auto":
            try:
                monitors: list[Monitor] = get_monitors()
                monitor: Monitor = next(
                    (m for m in monitors if getattr(m, "is_primary", False)), monitors[0]
                )
                self._size = (int(monitor.width * 0.9), int(monitor.height * 0.9))
            except (ScreenInfoError, IndexError):
                pass  # keep the 1280x720 fallback
        else:
            self._size = self.size

        self._dark = self.theme == "dark" or (self.theme == "auto" and bool(isDark()))
        self._osc_queue = Queue(maxsize=self.queue_size)

    @property
    def is_open(self) -> bool:
        return self._active and not self._stop.is_set()

    def open(self) -> None:
        if self.is_open:
            return

        self._reset()

        self._osc_thread = Thread(
            target=self._run, name=f"{type(self).__name__}_thread", daemon=True
        )
        self._osc_thread.start()

        if not self._ready.wait(15):
            raise TimeoutError(f"{type(self).__name__} did not start within 15s")

        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed to start") from self._error

    def run(self, producer: Callable[["Oscilloscope"], None] | None = None) -> None:
        self._reset()

        worker = None
        if producer is not None:
            worker = Thread(target=producer, args=(self,), name="OscilloscopeProducer", daemon=True)
            worker.start()

        try:
            self._run()
        finally:
            self._stop.set()

            if worker is not None:
                worker.join(timeout=2.0)

        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed") from self._error

    def close(self) -> None:
        """Ask the render loop to exit (it ends within one frame) and wait for it."""
        self._stop.set()
        thread = self._osc_thread
        if thread is None or thread is current_thread():
            return
        thread.join(5)
        if thread.is_alive():
            LOGGER.warning("Render thread still running after 5s")
        else:
            self._osc_thread = None

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Freeze the display (data keeps being collected). Thread-safe."""
        self._paused = True

    def resume(self) -> None:
        """Unfreeze the display; it catches up to the latest data. Thread-safe."""
        self._paused = False

    def reset_view(self) -> None:
        """Reset zoom/pan of every panel on the next frame. Thread-safe."""
        self._reset_requested.set()

    def add_data(self, data: dict[str, ndarray]) -> bool:
        if not self._started:
            raise RuntimeError(f"{type(self).__name__} needs to be opened first")

        if self._stop.is_set():
            return False

        try:
            # TODO use internal ts?
            self._osc_queue.put_nowait((monotonic_ns(), data))
        except Full:
            self._dropped += 1
            now = monotonic_ns()
            if now - self._last_drop_log > 1.0:
                LOGGER.warning("Oscilloscope queue full - dropped %d packet(s)", self._dropped)
                self._last_drop_log = now
        return True

    def _reset(self) -> None:
        self._stop.clear()
        self._ready.clear()
        self._started = True
        self._active = True
        self._error = None
        self._frame_errors = 0
        self._paused = False
        self._title_paused = False
        self._reset_requested.clear()
        self._keys = {}
        self._rejected.clear()
        while True:  # discard stale packets from a previous session
            try:
                self._osc_queue.get_nowait()
            except Empty:
                break

    def _build_ui(self) -> None:
        bind_theme(_global_theme(self._dark))
        self._window = add_window(
            no_title_bar=True, no_scrollbar=True, no_move=True, no_resize=True, no_collapse=True
        )
        self._placeholder = add_text("Waiting for data...", parent=self._window, indent=12)

    def _add_panel(self, name: str) -> bool:
        if len(self._panels) >= self.max_panels:
            if name not in self._rejected:
                self._rejected.add(name)
                LOGGER.warning(
                    "Maximum number of panels (%d) reached; ignoring %r", self.max_panels, name
                )
            return False

        rows: int = len(self._panels) + 1
        if self._subplots is None:
            if self._placeholder is not None:
                delete_item(self._placeholder)
                self._placeholder = None
            self._subplots = add_subplots(
                rows, 1, parent=self._window, width=-1, height=-1, link_all_x=True
            )
        else:
            configure_item(self._subplots, rows=rows)  # one more row, equal heights

        for previous in self._panels.values():
            previous.set_show_x_axis(False)

        self._panels[name] = SeriesPanel(
            name=name,
            parent=self._subplots,
            overlay=self._window,
            x_label="Time[s]",
            legend=self.legend,
            show_x_axis=True,
            dark=self._dark,
            static=not self.interactive,
            y_lim=self.y_lim,
            max_points=self.max_points,
            title_max_chars=self.title_max_chars,
        )
        return True

    def _panel_names(self, key: str, labels: tuple[str, ...]) -> list[tuple[str, list[str]]]:
        if not self.split_traces or len(labels) == 1:
            return [(key, list(labels))]
        return [(f"{key}[{label}]", [label]) for label in labels]

    def _drain_queue(self) -> dict[str, ndarray]:
        chunks: dict[str, list[ndarray]] = {}
        for _ in range(self._osc_queue.qsize()):  # bounded: ignore packets arriving meanwhile
            try:
                ts, p = self._osc_queue.get_nowait()
            except Empty:
                break

            for key, arr in p.items():
                normalized = self._normalize(key, arr)
                if normalized is not None:
                    chunks.setdefault(key, []).append(normalized)

        merged: dict[str, ndarray] = {}
        for key, items in chunks.items():
            dtype = items[0].dtype
            items = [a for a in items if a.dtype == dtype]
            arr = items[0] if len(items) == 1 else concatenate(items)

            if arr.size > 1 and (diff(arr["ts"]) < 0).any():
                arr = arr[argsort(arr["ts"], kind="stable")]

            merged[key] = arr

        return merged

    @staticmethod
    def _normalize(name: str, arr: Any) -> ndarray | None:
        a = asarray(arr)
        if a.ndim == 0:
            a = a.reshape(1)

        if a.dtype.names:  # structured: first field is x
            if a.ndim != 1 or len(a.dtype.names) < 2:
                raise ValueError(f"{name!r}: structured array needs 1-D data and >= 2 fields")
            fields = a.dtype.names
            x = a[fields[0]].astype("float64")
            labels = list(fields[1:])
            Y = column_stack([a[f].astype("float32") for f in labels])
        else:  # plain: column 0 is x
            if a.ndim != 2 or (len(a.shape) > 1 and a.shape[1] < 2):
                raise ValueError(f"{name!r}: expected shape (n, 1 + m) with x in column 0")

            x = a[:, 0].astype("float64")
            Y = a[:, 1:].astype("float32")
            labels = [str(i) for i in range(1, (len(a.shape) > 1 and a.shape[1]))]

        if "ts" in labels:
            raise ValueError(f"{name!r}: 'ts' is reserved for the x field")
        if x.size == 0:
            return None

        ok = isfinite(x)
        if not ok.all():
            x, Y = x[ok], Y[ok]
            if x.size == 0:
                return None
        Y[~isfinite(Y)] = nan  # NaN/inf become gaps
        if x.size > 1 and (diff(x) < 0).any():
            order = argsort(x, kind="stable")
            x, Y = x[order], Y[order]

        out = empty(x.size, dtype=[("ts", "f8")] + [(l, "f4") for l in labels])
        out["ts"] = x
        for j, label in enumerate(labels):
            out[label] = Y[:, j]
        return out

    def _ensure_panels(self, data: dict[str, ndarray]) -> None:
        for key, arr in data.items():
            if not arr.dtype.names:
                raise RuntimeError("Not structured array in data packet")

            for name, _labels in self._panel_names(key, arr.dtype.names[1:]):
                if name not in self._panels and name not in self._rejected:
                    self._add_panel(name)

    def _ingest(self, data: dict[str, ndarray]) -> None:
        for key, arr in data.items():
            if not arr.dtype.names:
                raise RuntimeError("Not structured array in data packet")

            for name, labels in self._panel_names(key, arr.dtype.names[1:]):
                panel = self._panels.get(name)
                if panel is not None:
                    panel.feed(arr, labels=labels)

    def _handle_input(self) -> None:
        """Keyboard/mouse shortcuts, polled once per frame (render thread)."""
        reset = self._reset_requested.is_set()
        if self.interactive:
            down = {"pause": is_key_down(mvKey_Spacebar), "reset": is_key_down(mvKey_R)}
            pressed = {
                k: v and not self._keys.get(k, False) for k, v in down.items()
            }  # rising edge
            self._keys = down
            if pressed["pause"]:
                self._paused = not self._paused
            reset = reset or pressed["reset"]
            if is_mouse_button_double_clicked(mvMouseButton_Left):
                reset = reset or any(is_item_hovered(p.plot) for p in self._panels.values())

        if reset:
            self._reset_requested.clear()
            for panel in self._panels.values():
                panel.reset_view()
        if self._paused != self._title_paused:
            self._title_paused = self._paused
            set_viewport_title(f"{self.title} [paused]" if self._paused else self.title)

    def _frame(self) -> None:
        try:
            self._handle_input()

            data = self._drain_queue()
            if data:
                self._ensure_panels(data)
                self._ingest(data)

            for panel in self._panels.values():
                panel.tick()

                if not self._paused:
                    panel.update()

            self._frame_errors = 0
        except Exception:
            LOGGER.exception("Oscilloscope frame failed")
            self._frame_errors += 1
            if self._frame_errors >= _MAX_CONSECUTIVE_FRAME_ERRORS:
                LOGGER.error("Too many consecutive frame errors; stopping")
                self._stop.set()

    def _run(self) -> None:
        context_created = False
        try:
            create_context()
            context_created = True
            _THEMES.clear()  # theme ids belong to one context; never reuse across sessions

            self._build_ui()

            create_viewport(
                title=self.title,
                width=self._size[0],
                height=self._size[1],
                clear_color=BG_DARK if self._dark else BG_WHITE,
            )
            setup_dearpygui()
            show_viewport()
            set_primary_window(self._window, value=True)
            self._ready.set()

            while is_dearpygui_running() and not self._stop.is_set():
                self._frame()
                render_dearpygui_frame()
        except Exception as exc:
            self._error = exc
            LOGGER.exception(exc)
        finally:
            self._stop.set()
            self._active = False
            self._panels.clear()
            self._subplots = self._window = self._placeholder = None
            _THEMES.clear()
            if context_created:
                try:
                    destroy_context()
                except Exception:
                    LOGGER.exception("destroy_context() failed")
            self._ready.set()  # never leave open() waiting

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
