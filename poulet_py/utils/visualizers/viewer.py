try:
    from collections import deque
    from multiprocessing import Event, get_context
    from multiprocessing.context import SpawnProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as TypeEvent
    from time import monotonic, monotonic_ns
    from types import TracebackType
    from typing import Any, Literal, NamedTuple, Self

    from darkdetect import isDark
    from dearpygui.dearpygui import (
        add_image_series,
        add_plot,
        add_plot_axis,
        add_raw_texture,
        add_subplots,
        add_text,
        add_texture_registry,
        add_window,
        bind_theme,
        configure_item,
        create_context,
        create_viewport,
        delete_item,
        destroy_context,
        get_item_rect_size,
        is_dearpygui_running,
        is_item_hovered,
        is_key_down,
        is_mouse_button_double_clicked,
        mvFormat_Float_rgba,
        mvKey_R,
        mvKey_Spacebar,
        mvMouseButton_Left,
        mvXAxis,
        mvYAxis,
        render_dearpygui_frame,
        set_axis_limits,
        set_axis_limits_auto,
        set_primary_window,
        set_value,
        set_viewport_title,
        setup_dearpygui,
        show_viewport,
    )
    from numpy import (
        array,
        asarray,
        ceil,
        clip,
        copyto,
        dtype,
        fmax,
        fmin,
        iinfo,
        isfinite,
        nan,
        nan_to_num,
        ndarray,
        sqrt,
    )
    from numpy._core.numeric import ones
    from pydantic import BaseModel, Field, PrivateAttr, field_validator
    from screeninfo import ScreenInfoError, get_monitors

    from poulet_py.config.logging import LOGGER
    from poulet_py.utils.visualizers.common import (
        _CHANNELS,
        _LOCK_FRAMES,
        _MAX_CONSECUTIVE_FRAME_ERRORS,
        _STATS_PERIOD,
        BG_DARK,
        BG_WHITE,
        _global_theme,
    )
except ImportError as e:
    raise ImportError("""
Missing 'vis' module. Install options:
- Dedicated:    pip install poulet_py[vis]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
""") from e


class _Clip(NamedTuple):
    """One queued entry: ``frames`` is ``(T, h, w, C)``; ``step`` is the pixel stride already applied."""

    frames: ndarray[Any, Any]
    step: int


def _dtype_range(dtype: dtype) -> tuple[float, float]:
    """Default display range of a dtype: integers 0..max, floats and bool 0..1."""
    if dtype.kind in "iu":
        return 0.0, float(iinfo(dtype).max)
    return 0.0, 1.0


def _to_texture(frame: ndarray[Any, Any], rgb: ndarray[Any, Any], lo: float, hi: float) -> None:
    """Write ``frame`` (h, w, 1|3, any numeric dtype) into ``rgb`` (h, w, 3 float32 view), scaled
    from ``[lo, hi]`` to ``[0, 1]``. In place, no temporaries the size of the frame."""
    copyto(rgb, frame, casting="unsafe")  # (h, w, 1) broadcasts over the three channels
    if lo != 0.0:
        rgb -= lo
    rgb *= 1.0 / (hi - lo) if hi > lo else 0.0
    if frame.dtype.kind == "f":
        nan_to_num(rgb, copy=False, nan=0.0)  # NaN pixels -> black
    clip(rgb, 0.0, 1.0, out=rgb)


class _Panel:
    """One video panel: its texture, view state and title. Render thread only.

    Settings (fps, value_range, show_axes, ...) are read from the owning ``Visualizer``.
    """

    def __init__(self, name: str, parent: Any, registry: Any, cfg: "Visualizer"):
        self.name, self.cfg, self._registry = name, cfg, registry
        self.static = not cfg.interactive
        self._pending: deque[tuple[ndarray[Any, Any], int]] = deque(
            maxlen=1 if cfg.fps is None else cfg.backlog
        )
        self._idle = True  # nothing was waiting at the previous update (restarts the paced clock)
        self._next_t = 0.0  # paced mode: when the next frame is due

        self._geom: tuple[int, int, int] | None = None  # (h, w, step) of the current texture
        self._buf: ndarray[Any, Any] | None = None  # flat float32 RGBA, owned by the texture
        self._rgb: ndarray[Any, Any] | None = None  # (h, w, 3) view into _buf
        self._texture: Any = None
        self._series: Any = None
        self._range: tuple[float, float] | None = None  # tracked range for value_range="auto"

        # View state: an axis is LOCKED for a few frames to apply limits, then released so the
        # user can zoom/pan it.
        self._need_fit = True
        self._release = 0
        self._last_size: tuple[float, float] = (0.0, 0.0)

        self._label = name
        self._count = 0
        self._fps = 0.0
        self._t0 = monotonic()

        self.plot = add_plot(
            label=name,
            parent=parent,
            width=-1,
            height=-1,
            no_inputs=self.static,
            equal_aspects=True,
        )
        decor = {
            "no_gridlines": True,
            "no_tick_labels": not cfg.show_axes,
            "no_tick_marks": not cfg.show_axes,
        }
        self._x_axis = add_plot_axis(mvXAxis, parent=self.plot, **decor)
        # inverted: y = row index, so axis coordinates match array indices (row 0 on top)
        self._y_axis = add_plot_axis(mvYAxis, parent=self.plot, invert=True, **decor)

    # ---------------------------------------------------------------- per frame

    def feed(self, clip: _Clip) -> None:
        """Queue the frames of one packet (render thread). Live mode keeps only the newest."""
        frames = clip.frames[-1:] if self.cfg.fps is None else clip.frames
        for frame in frames:
            self._pending.append((frame, clip.step))  # bounded deque: oldest falls off

    def catch_up(self) -> None:
        """After a pause: forget the backlog, keep only the newest frame."""
        while len(self._pending) > 1:
            self._pending.popleft()
        self._idle = True

    def update(self, now: float) -> None:
        """Upload the frame that is due (live: the newest; paced: the last one whose time came)."""
        if not self._pending:
            self._idle = True
            return
        fps = self.cfg.fps
        if fps is None:
            item = self._pending.pop()
            self._pending.clear()
        else:
            if self._idle:  # first frame, or the producer ran dry: restart the clock
                self._next_t, self._idle = now, False
            item = None
            while self._pending and self._next_t <= now:
                item = self._pending.popleft()
                self._next_t += 1.0 / fps
            if item is None:
                return
        self._show(*item)

    def tick(self, now: float) -> None:
        """Every frame: title/stats, (re)fit the view once laid out, release locked axes."""
        self._update_label(now)
        size = get_item_rect_size(self.plot)
        size = (float(size[0]), float(size[1]))
        if self.static and size != self._last_size:
            self._need_fit = True  # no user zoom in static mode: just refit on resize
        self._last_size = size
        if self._need_fit and self._geom is not None and size[0] > 1 and size[1] > 1:
            self._apply_fit(size)
        if self._release > 0:
            self._release -= 1
            if self._release == 0:
                set_axis_limits_auto(self._x_axis)
                set_axis_limits_auto(self._y_axis)

    # ------------------------------------------------------------------ view

    def reset_view(self) -> None:
        """Back to the whole frame (aspect preserved), applied on the next tick."""
        self._need_fit = True

    def _apply_fit(self, size: tuple[float, float]) -> None:
        """Lock both axes to the largest view that shows the whole frame at square pixels."""
        h, w, step = self._geom  # type: ignore[misc]
        full_w, full_h = float(w * step), float(h * step)
        # Inner plot area, estimated from the item rect (title row, borders and tick labels
        # eat some of it); ImPlot's equal-aspect constraint corrects the remaining error.
        pw = max(size[0] - (62.0 if self.cfg.show_axes else 14.0), 1.0)
        ph = max(size[1] - (56.0 if self.cfg.show_axes else 40.0), 1.0)
        units_per_px = 1.03 * max(full_w / pw, full_h / ph)  # 3% margin against the estimate
        span_x, span_y = pw * units_per_px, ph * units_per_px
        set_axis_limits(self._x_axis, (full_w - span_x) / 2, (full_w + span_x) / 2)
        set_axis_limits(self._y_axis, (full_h - span_y) / 2, (full_h + span_y) / 2)
        self._release = 0 if self.static else _LOCK_FRAMES
        self._need_fit = False

    # ----------------------------------------------------------------- widgets

    def _show(self, frame: ndarray[Any, Any], step: int) -> None:
        if frame.shape[2] == 4:
            frame = frame[..., :3]  # alpha ignored
        h, w, _ = frame.shape
        if self._geom != (h, w, step):
            self._rebuild(h, w, step)
        lo, hi = self._value_range(frame)
        _to_texture(frame, self._rgb, lo, hi)  # type: ignore[arg-type]
        set_value(self._texture, self._buf)
        self._count += 1

    def _rebuild(self, h: int, w: int, step: int) -> None:
        """(Re)create the texture + image series for a new frame size."""
        old_series, old_texture = self._series, self._texture
        self._geom = (h, w, step)
        self._buf = ones(h * w * 4, dtype="float32")  # alpha channel stays 1
        self._rgb = self._buf.reshape(h, w, 4)[..., :3]
        self._texture = add_raw_texture(
            w, h, self._buf, format=mvFormat_Float_rgba, parent=self._registry
        )
        self._series = add_image_series(
            self._texture, (0.0, 0.0), (float(w * step), float(h * step)), parent=self._y_axis
        )
        if old_series is not None:
            delete_item(old_series)
        if old_texture is not None:
            delete_item(old_texture)
        self.reset_view()

    def _value_range(self, frame: ndarray[Any, Any]) -> tuple[float, float]:
        vr = self.cfg.value_range
        if vr is None:
            return _dtype_range(frame.dtype)
        if not isinstance(vr, str):
            return vr
        lo = float(fmin.reduce(frame, axis=None))  # NaN-ignoring
        hi = float(fmax.reduce(frame, axis=None))
        if not (isfinite(lo) and isfinite(hi)):
            return self._range or (0.0, 1.0)  # all-NaN frame
        if self._range is None:
            self._range = (lo, hi)
        else:  # widen at once, relax slowly
            r_lo, r_hi = self._range
            r_lo = lo if lo < r_lo else r_lo + 0.05 * (lo - r_lo)
            r_hi = hi if hi > r_hi else r_hi + 0.05 * (hi - r_hi)
            self._range = (r_lo, r_hi)
        return self._range

    def _update_label(self, now: float) -> None:
        """Title: ``name  WxH  xx.x fps`` (the rate is measured over the last ~0.5 s)."""
        if now - self._t0 >= _STATS_PERIOD:
            self._fps = self._count / (now - self._t0)
            self._count, self._t0 = 0, now
        text = self.name
        if self.cfg.show_stats and self._geom is not None:
            h, w, step = self._geom
            text = f"{self.name}  {w * step}x{h * step}  {self._fps:.1f} fps"
        if text != self._label:
            self._label = text
            configure_item(self.plot, label=text)


class Viewer(BaseModel):
    title: str = Field(default="Visualizer")
    fps: float | None = Field(
        default=None,
        gt=0,
        description="None: live, always show the newest frame. Else: paced playback at this rate.",
    )
    backlog: int = Field(
        default=300, ge=1, description="Frames buffered per panel in paced mode (oldest dropped)"
    )
    value_range: tuple[float, float] | Literal["auto"] | None = Field(
        default=None,
        description="Intensity mapped to black..white. None = by dtype, (lo, hi) = fixed, "
        '"auto" = adaptive contrast stretch.',
    )
    max_pixels: int = Field(
        default=2_500_000,
        ge=64,
        description="Frames with more pixels are strided down by an integer factor",
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
    interactive: bool = Field(
        default=True, description="Mouse zoom/pan/reset and keyboard shortcuts (Space, R)"
    )
    size: Literal["auto"] | tuple[int, int] = Field(default="auto")
    theme: Literal["auto", "dark", "white"] = Field(default="auto")
    queue_size: int = Field(
        default=32, gt=0, description="Packets in the internal queue (oldest dropped when full)"
    )

    _size: tuple[int, int] = PrivateAttr(default=(1280, 720))
    _dark: bool = PrivateAttr(default=True)
    _window: Any = PrivateAttr(default=None)
    _placeholder: Any = PrivateAttr(default=None)
    _subplots: Any = PrivateAttr(default=None)
    _registry: Any = PrivateAttr(default=None)
    _panels: dict[str, _Panel] = PrivateAttr(default_factory=dict)
    _rejected: set[str] = PrivateAttr(default_factory=set)

    _viewer_queue: Queue[tuple[int, dict[str, _Clip]]] | None = PrivateAttr(default=None)
    _viewer_process: SpawnProcess | None = PrivateAttr(default=None)
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

    @field_validator("value_range")
    @classmethod
    def _check_value_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if isinstance(v, tuple) and not v[0] < v[1]:
            raise ValueError("value_range must be (lo, hi) with lo < hi")
        return v

    def model_post_init(self, context) -> None:
        if self.size == "auto":
            try:
                monitors = get_monitors()
                monitor = next(
                    (m for m in monitors if getattr(m, "is_primary", False)), monitors[0]
                )
                self._size = (int(monitor.width * 0.9), int(monitor.height * 0.9))
            except (ScreenInfoError, IndexError):
                pass  # keep the 1280x720 fallback
        else:
            self._size = self.size

        self._dark = self.theme == "dark" or (self.theme == "auto" and bool(isDark()))
        self._viewer_queue = Queue(maxsize=self.queue_size)

    # -------------------------------------------------------------- public API

    @property
    def is_open(self) -> bool:
        """True from open()/run() until the window closes or ``close()`` is called."""
        return self._active and not self._stop.is_set()

    def open(self, timeout: float = 15.0) -> None:
        """Run the render loop in a background thread; waits for the window."""
        if self.is_open:
            return

        self._reset()

        ctx = get_context("spawn")
        self._viewer_queue = ctx.Queue(maxsize=self.queue_size)

        self._viewer_process = ctx.Process(
            target=_viewer_process,
            args=(self.model_dump(), self._viewer_queue, self._stop),
            daemon=False,
        )

        self._viewer_process.start()

    def close(self) -> None:
        """Ask the render loop to exit (it ends within one frame) and wait for it."""
        self._stop.set()
        if self._viewer_process:
            self._viewer_process.join(5)
            if self._viewer_process.is_alive():
                LOGGER.warning("Render thread still running after 5s")
            else:
                self._viewer_process = None

    def reset_view(self) -> None:
        """Reset zoom/pan of every panel on the next frame. Thread-safe."""
        self._reset_requested.set()

    def add_data(self, data: dict[str, ndarray[Any, Any]]) -> bool:
        """Queue new frames (thread-safe, non-blocking). See the module docstring.

        Returns False (without raising) once the window has been closed, so a producer
        loop can simply stop; raises only for bad input or if never started.
        """
        if not self._started:
            raise RuntimeError(f"{type(self).__name__} needs to be opened first")

        packet: dict[str, _Clip] = {}
        for name, arr in data.items():
            clip = self._normalize(name, arr)
            if clip is not None:
                packet[name] = clip

        if self._stop.is_set():
            return False

        if not packet:
            return True

        if not self._viewer_queue:
            return False

        if self._viewer_queue.full():
            self._viewer_queue.get_nowait()
            self._viewer_queue.put_nowait((monotonic_ns(), packet))

            self._dropped += 1
            now = monotonic_ns()
            if now - self._last_drop_log > 1.0:
                LOGGER.warning("Oscilloscope queue full - dropped %d packet(s)", self._dropped)
                self._last_drop_log = now
        else:
            self._viewer_queue.put_nowait((monotonic_ns(), packet))

        return True

    # ------------------------------------------------------------ input parsing

    def _normalize(self, name: str, arr: Any) -> _Clip | None:
        """-> ``_Clip`` with frames as a C-contiguous ``(T, h, w, C)`` array. Always copies."""
        a = asarray(arr)
        if a.ndim == 2:  # (H, W)
            a = a[None, :, :, None]
        elif a.ndim == 3:  # (H, W, C) if the last axis looks like channels, else (T, H, W)
            a = a[None] if a.shape[-1] in _CHANNELS else a[..., None]
        elif a.ndim != 4:
            raise ValueError(
                f"{name!r}: expected (H, W), (H, W, C), (T, H, W) or (T, H, W, C), "
                f"got shape {a.shape}"
            )
        t, h, w, c = a.shape
        if c not in _CHANNELS:
            raise ValueError(f"{name!r}: channel axis must have 1, 3 or 4 entries, got {c}")
        if a.dtype.kind not in "biuf":
            raise ValueError(f"{name!r}: unsupported dtype {a.dtype} (need bool, int or float)")
        if t == 0 or h == 0 or w == 0:
            return None

        step = max(1, ceil(sqrt(h * w / self.max_pixels)))
        dtype = "float32" if a.dtype.kind == "f" and a.dtype.itemsize > 4 else a.dtype
        a = array(a[:, ::step, ::step, :], dtype=dtype, order="C")  # the one copy
        if a.dtype.kind == "f":
            bad = ~isfinite(a)
            if bad.any():
                a[bad] = nan  # inf -> NaN, drawn black
        return _Clip(a, step)

    # ---------------------------------------------------------------- UI building

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

        if not self._viewer_queue:
            return

        while True:
            if self._viewer_queue.empty():
                break
            self._viewer_queue.get_nowait()

    def _build_ui(self) -> None:
        bind_theme(_global_theme(self._dark))
        self._registry = add_texture_registry()
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

        n = len(self._panels) + 1
        cols = min(self.columns or ceil(sqrt(n)), n)
        rows = ceil(n / cols)
        if self._subplots is None:
            if self._placeholder is not None:
                delete_item(self._placeholder)
                self._placeholder = None
            self._subplots = add_subplots(
                rows,
                cols,
                parent=self._window,
                width=-1,
                height=-1,
                link_all_x=self.link_axes,
                link_all_y=self.link_axes,
            )
        else:
            configure_item(self._subplots, rows=rows, columns=cols)  # equal cell sizes

        self._panels[name] = _Panel(name, self._subplots, self._registry, self)
        return True

    # -------------------------------------------------------------- frame logic

    def _drain_queue(self) -> dict[str, list[_Clip]]:
        if not self._viewer_queue:
            raise RuntimeError("No queue has been assinged")

        out: dict[str, list[_Clip]] = {}
        while True:
            if self._viewer_queue.empty():
                break

            ts, packet = self._viewer_queue.get_nowait()

            for key, clip in packet.items():
                out.setdefault(key, []).append(clip)

        return out

    def _ensure_panels(self, data: dict[str, list[_Clip]]) -> None:
        for key in data:
            if key not in self._panels and key not in self._rejected:
                self._add_panel(key)

    def _ingest(self, data: dict[str, list[_Clip]]) -> None:
        for key, clips in data.items():
            panel = self._panels.get(key)
            if panel is not None:
                for clip in clips:
                    panel.feed(clip)

    def _handle_input(self) -> None:
        """Keyboard/mouse shortcuts, polled once per frame (render thread)."""
        reset = self._reset_requested.is_set()

        if self.interactive:
            down = {
                "pause": is_key_down(mvKey_Spacebar),
                "reset": is_key_down(mvKey_R),
            }
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
            if not self._paused:  # resumed: jump to the newest frame
                for panel in self._panels.values():
                    panel.catch_up()

    def _frame(self) -> None:
        """One iteration of the render loop (render thread only)."""
        try:
            self._handle_input()

            data = self._drain_queue()
            if data:
                self._ensure_panels(data)
                self._ingest(data)  # keeps collecting while paused

            now = monotonic()
            for panel in self._panels.values():
                panel.tick(now)

                if not self._paused:
                    panel.update(now)

            self._frame_errors = 0
        except Exception:
            LOGGER.exception("Visualizer frame failed")
            self._frame_errors += 1
            if self._frame_errors >= _MAX_CONSECUTIVE_FRAME_ERRORS:
                LOGGER.error("Too many consecutive frame errors; stopping")
                self._stop.set()

    def _run(self) -> None:
        context_created = False
        try:
            create_context()
            context_created = True

            self._build_ui()

            create_viewport(
                title=self.title,
                width=self._size[0],
                height=self._size[1],
                clear_color=BG_DARK if self._dark else BG_WHITE,
            )
            setup_dearpygui()
            show_viewport()
            set_primary_window(self._window, True)
            self._ready.set()

            while is_dearpygui_running() and not self._stop.is_set():
                self._frame()
                render_dearpygui_frame()
        except Exception as exc:
            self._error = exc
            LOGGER.exception(exc)
        finally:
            self._stop.set()  # is_open turns False immediately; add_data starts returning False
            self._active = False
            self._panels.clear()
            self._subplots = self._window = self._placeholder = self._registry = None
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


def _viewer_process(
    config: dict[str, Any],
    queue: Queue[tuple[int, dict[str, _Clip]]],
    stop_event: TypeEvent,
) -> None:
    view = Viewer(**config)
    view._viewer_queue = queue
    view._stop = stop_event

    try:
        view._run()
    except Exception as e:
        LOGGER.exception(e)
    finally:
        pass
