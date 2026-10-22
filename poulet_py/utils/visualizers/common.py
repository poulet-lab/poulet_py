from abc import abstractmethod


try:
    from multiprocessing import Event, get_context
    from multiprocessing.context import SpawnProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as TypeEvent
    from time import monotonic_ns
    from types import TracebackType
    from typing import Any, Literal, Self

    from darkdetect import isDark
    from dearpygui.dearpygui import (
        add_subplots,
        add_text,
        add_theme_color,
        add_theme_style,
        add_window,
        bind_theme,
        configure_item,
        create_context,
        create_viewport,
        delete_item,
        destroy_context,
        is_dearpygui_running,
        is_item_hovered,
        is_key_down,
        is_mouse_button_double_clicked,
        mvAll,
        mvKey_R,
        mvKey_Spacebar,
        mvMouseButton_Left,
        mvPlot,
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
        render_dearpygui_frame,
        set_primary_window,
        set_viewport_title,
        setup_dearpygui,
        show_viewport,
        theme,
        theme_component,
    )
    from numpy import (
        argsort,
        asarray,
        column_stack,
        concatenate,
        diff,
        empty,
        isfinite,
        nan,
        ndarray,
    )
    from pydantic import BaseModel, Field, PrivateAttr
    from screeninfo import ScreenInfoError, get_monitors
    from screeninfo.common import Monitor

    from poulet_py import LOGGER

except ImportError as e:
    raise ImportError("""
Missing 'osc' module. Install options:
- Dedicated:    pip install poulet_py[view]
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
_STATS_PERIOD = 0.5  # seconds between display-rate updates in the panel title
_CHANNELS = (1, 3, 4)


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


class BaseVisualizer(BaseModel):
    title: str = Field(default="Visualizer")
    max_data_points: int = Field(
        default=2000, ge=16, description="Display points per trace per duration"
    )
    max_panels: int = Field(default=64, gt=0, description="Safety cap on panel count")
    columns: int | None = Field(
        default=None, ge=1, description="Panels per row. None = roughly square grid."
    )
    link_axes: bool = Field(
        default=False, description="Link zoom/pan across panels (same-sized frames)"
    )
    show_axes: bool = Field(default=True, description="Pixel-coordinate ticks on every panel")
    show_stats: bool = Field(default=True, description="Frame size and display rate in the title")
    legend: bool = Field(default=True, description="Name the traces of each panel")
    y_lim: tuple[float, float] | None = Field(
        default=None, description="Fixed (ymin, ymax). None = auto-fit per panel."
    )
    x_label: str | None = Field(default="Time (s)", description="Shown under the bottom panel only")
    interactive: bool = Field(default=True, description="Mouse zoom/pan/reset")
    size: Literal["auto"] | tuple[int, int] = Field(default="auto")
    theme: Literal["auto", "dark", "white"] = Field(default="auto")
    queue_size: int = Field(default=1000, gt=0, description="Size of the internal queue")

    _size: tuple[int, int] = PrivateAttr(default=(1280, 720))
    _dark: bool = PrivateAttr(default=True)
    _window: Any = PrivateAttr(default=None)
    _placeholder: Any = PrivateAttr(default=None)
    _subplots: int | str | None = PrivateAttr(default=None)
    _panels: dict[str, BasePanel] = PrivateAttr(default_factory=dict)
    _rejected: set[str] = PrivateAttr(default_factory=set)

    _queue: Queue[tuple[int, dict[str, ndarray[Any, Any]]]] | None = PrivateAttr(default=None)
    _process: SpawnProcess | None = PrivateAttr(default=None)
    _ready: TypeEvent = PrivateAttr(default_factory=Event)
    _stop: TypeEvent = PrivateAttr(default_factory=Event)
    _error: BaseException | None = PrivateAttr(default=None)
    _started: bool = PrivateAttr(default=False)
    _active: bool = PrivateAttr(default=False)
    _paused: bool = PrivateAttr(default=False)
    _title_paused: bool = PrivateAttr(default=False)
    _reset_requested: TypeEvent = PrivateAttr(default_factory=Event)
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

    @property
    def is_open(self) -> bool:
        return self._active and not self._stop.is_set()

    @abstractmethod
    def _normalize(self, name: str, arr: Any) -> ndarray[Any, Any] | None:
        pass

    def open(self) -> None:
        if self.is_open:
            return

        self._reset()

        ctx = get_context("spawn")
        self._queue = ctx.Queue(maxsize=self.queue_size)
        # TODO test if this will work
        self._process = ctx.Process(target=_visualizer_process, args=(self), daemon=False)

        self._process.start()

    def close(self) -> None:
        self._stop.set()
        if self._process:
            self._process.join(5)
            if self._process.is_alive():
                LOGGER.warning("Render process still running after 5s")
            else:
                self._process = None

    def reset_view(self) -> None:
        self._reset_requested.set()

    def add_data(self, data: dict[str, ndarray[Any, Any]]) -> bool:
        if not self._started:
            raise RuntimeError(f"{type(self).__name__} needs to be opened first")

        if self._stop.is_set():
            return False

        if not self._queue:
            return False

        if self._queue.full():
            self._queue.get_nowait()

        self._queue.put_nowait((monotonic_ns(), data))

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

        if not self._queue:
            return

        while True:
            if self._queue.empty():
                break
            self._queue.get_nowait()

        self._queue = None

    def _build_ui(self) -> None:
        bind_theme(_global_theme(self._dark))
        self._window = add_window(
            no_title_bar=True, no_scrollbar=True, no_move=True, no_resize=True, no_collapse=True
        )
        self._placeholder = add_text("Waiting for data...", parent=self._window, indent=12)

    def _add_panel(self, name: str) -> bool:
        # TODO
        pass

    def _panel_names(self, key: str, labels: tuple[str, ...]) -> list[tuple[str, list[str]]]:
        # TODO
        pass

    def _drain_queue(self) -> dict[str, ndarray[Any, Any]]:
        if not self._queue:
            raise RuntimeError("No queue has been assinged")

        chunks: dict[str, list[ndarray[Any, Any]]] = {}
        while True:
            if self._queue.empty():
                break

            ts, p = self._queue.get_nowait()

            for key, arr in p.items():
                normalized = self._normalize(key, arr)
                if normalized is not None:
                    chunks.setdefault(key, []).append(normalized)

        merged: dict[str, ndarray[Any, Any]] = {}
        for key, items in chunks.items():
            dtype = items[0].dtype
            items = [a for a in items if a.dtype == dtype]
            arr = items[0] if len(items) == 1 else concatenate(items)

            if arr.size > 1 and (diff(arr["ts"]) < 0).any():
                arr = arr[argsort(arr["ts"], kind="stable")]

            merged[key] = arr

        return merged

    def _ensure_panels(self, data: dict[str, ndarray[Any, Any]]) -> None:
        for key, arr in data.items():
            if not arr.dtype.names:
                raise RuntimeError("Not structured array in data packet")

            for name, _labels in self._panel_names(key, arr.dtype.names[1:]):
                if name not in self._panels and name not in self._rejected:
                    self._add_panel(name)

    def _ingest(self, data: dict[str, ndarray[Any, Any]]) -> None:
        for key, arr in data.items():
            if not arr.dtype.names:
                raise RuntimeError("Not structured array in data packet")

            for name, labels in self._panel_names(key, arr.dtype.names[1:]):
                panel = self._panels.get(name)
                if panel is not None:
                    panel.feed(arr, labels=labels)

    def _handle_input(self) -> None:
        reset = self._reset_requested.is_set()

        if self.interactive:
            down = {"pause": is_key_down(mvKey_Spacebar), "reset": is_key_down(mvKey_R)}
            pressed = {k: v and not self._keys.get(k, False) for k, v in down.items()}
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
            if not self._paused:
                for panel in self._panels.values():
                    panel.catch_up()

    def _frame(self) -> None:
        try:
            self._handle_input()

            data = self._drain_queue()
            if data:
                self._ensure_panels(data)
                self._ingest(data)

            now = monotonic_ns()
            for panel in self._panels.values():
                panel.tick(now)

                if not self._paused:
                    panel.update(now)

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
            _THEMES.clear()

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


def _visualizer_process(visualizer: BaseVisualizer) -> None:
    try:
        visualizer._run()
    except Exception as e:
        LOGGER.exception(e)
    finally:
        pass
