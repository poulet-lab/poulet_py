try:
    from collections import deque
    from threading import Lock
    from typing import Any, Literal

    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.pyplot import close, ion, rcParams, show, subplots
    from numpy import arange, array, column_stack, ndarray
    from pandas import DataFrame
except ImportError as e:
    msg = """
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[osc]
- Module:       pip install poulet_py[utils]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class Oscilloscope:
    """A real-time data visualization tool similar to an oscilloscope.

    This class provides a thread-safe, animated plotting interface for visualizing
    streaming data with automatic downsampling and view adjustment.

    Parameters
    ----------
    max_samples : int, optional
        Maximum number of samples to keep in memory (default: 1000).
    max_points : int, optional
        Maximum number of points to display (downsampling threshold, default: 100).
    title : str, optional
        Title of the plot (default: "Real-time Data").
    xlabel : str, optional
        Label for the x-axis (default: "X").
    ylabel : str, optional
        Label for the y-axis (default: "Y").
    xlim : tuple[float, float] or "auto", optional
        Fixed x-axis limits or "auto" for automatic adjustment (default: "auto").
    ylim : tuple[float, float] or "auto", optional
        Fixed y-axis limits or "auto" for automatic adjustment (default: "auto").
    xpadding : float, optional
        Padding factor for x-axis when in auto mode (default: 0.1).
    ypadding : float, optional
        Padding factor for y-axis when in auto mode (default: 0.1).
    animation_interval : int, optional
        Refresh interval in milliseconds (default: 33 ~30fps).

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure instance containing the plot.
    ax : matplotlib.axes.Axes
        The axes instance for the plot.

    Examples
    --------
    >>> osc = Oscilloscope(max_samples=500, title="Sensor Data")
    >>> osc.start()
    >>> osc.add_data({"temp": 25.3, "humidity": 45.2})
    """

    def __init__(
        self,
        max_samples: int = 1000,
        max_points: int = 100,
        title: str = "Real-time Data",
        xlabel: str = "X",
        ylabel: str = "Y",
        xlim: tuple[float, float] | Literal["auto"] = "auto",
        ylim: tuple[float, float] | Literal["auto"] = "auto",
        xpadding: float = 0.1,
        ypadding: float = 0.1,
        animation_interval: int = 33,
    ):
        self.max_samples = max_samples
        self.max_points = max_points
        self.xlim = xlim
        self.ylim = ylim
        self.xpadding = xpadding
        self.ypadding = ypadding
        self.animation_interval = animation_interval

        # Static elements
        self._data_lock = Lock()
        self._color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
        self.fig, self.ax = subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

        # Dynamic elements
        self._x = deque(maxlen=max_samples)
        self._y = deque(maxlen=max_samples)
        self._line_collection: LineCollection | None = None
        self._legend_handles: list[Line2D] = []
        self._animation: FuncAnimation | None = None
        self._last_ymin = float("inf")
        self._last_ymax = -float("inf")
        self._last_xmin = float("inf")
        self._last_xmax = -float("inf")

    def add_data(self, y: dict, x: Any | None = None) -> None:
        """Add new data points to the oscilloscope.

        This method is thread-safe and can be called from multiple threads.

        Parameters
        ----------
        y : dict
            Dictionary of y-values where keys are series names and values are
            the data points. Each call adds one sample per series.
        x : Any, optional
            Corresponding x-value for the y samples. If None, uses an
            auto-incremented index (default: None).

        Notes
        -----
        The data is stored in internal buffers and will be displayed during the
        next animation frame.
        """
        with self._data_lock:
            self._y.append(y)
            if x is not None:
                self._x.append(x)

    def start(self) -> None:
        """Start the real-time plotting animation.

        Enables interactive mode and begins periodic updates of the display.
        The plot window will be non-blocking.

        See Also
        --------
        stop : Stop the animation and close the window.
        """
        if not self._animation:
            ion()
            self._animation = FuncAnimation(
                self.fig,
                self._update,
                interval=self.animation_interval,
                blit=False,
                cache_frame_data=False,
            )
            show(block=False)
            self.force_redraw()

    def stop(self) -> None:
        """Stop the animation and clean up resources.

        Closes the plot window and clears all internal data buffers.
        """
        if self._animation:
            self._animation.event_source.stop()
            close(self.fig)
            self._animation = None
            self._line_collection = None
            self._legend_handles = []
            self._last_ymin = float("inf")
            self._last_ymax = -float("inf")
            self._last_xmin = float("inf")
            self._last_xmax = -float("inf")
            self.ax.clear()

        with self._data_lock:
            self._x.clear()
            self._y.clear()

    def _update(self) -> list[LineCollection]:
        """Internal animation update handler.

        Parameters
        ----------
        frame : int
            Frame number (unused, required by FuncAnimation interface).

        Returns
        -------
        list
            List of artists to be redrawn (for blitting optimization).
        """
        with self._data_lock:
            x, y = self._downscale()

            segments = []
            colors = []
            new_legend_handles = []

            for i, col in enumerate(y.columns):
                segments.append(column_stack((x, y[col].values)))
                colors.append(self._color_cycle[i % len(self._color_cycle)])

                # Create proxy artist for legend
                if col not in [h.get_label() for h in self._legend_handles]:
                    proxy = Line2D([], [], color=colors[-1], linewidth=1.5, label=col)
                    new_legend_handles.append(proxy)

            # Update or create LineCollection
            if self._line_collection is None:
                self._line_collection = LineCollection(
                    segments, linewidths=1.5, colors=colors, linestyle="-"
                )
                self.ax.add_collection(self._line_collection)
            else:
                self._line_collection.set_segments(segments)
                self._line_collection.set_color(colors)

            # Update legend if needed
            if new_legend_handles:
                self._legend_handles.extend(new_legend_handles)
                self.ax.legend(handles=self._legend_handles, loc="upper right")

            self._update_view(y, x)
            return [self._line_collection]

    def _update_view(self, y: DataFrame, x: ndarray) -> None:
        """Adjust the view limits based on current data.

        Parameters
        ----------
        y : DataFrame
            Current y-axis data points.
        x : ndarray
            Current x-axis data points.

        Notes
        -----
        Applies padding when in auto-range mode and only updates the view
        when necessary to maintain smooth animation.
        """
        # Calculate ranges
        if isinstance(self.xlim, tuple):
            x_min, x_max = self.xlim
        else:  # auto
            x_min = x.min(axis=None)
            x_max = x.max(axis=None)

        if isinstance(self.ylim, tuple):
            y_min, y_max = self.ylim
        else:  # auto
            y_min = y.min(axis=None)
            y_max = y.max(axis=None)

            if y_min == y_max:
                y_min -= self.ypadding
                y_max += self.ypadding

        # Calculate padding
        x_pad = (x_max - x_min) * self.xpadding if len(x) > 1 else self.xpadding
        y_pad = (y_max - y_min) * self.ypadding if y_max != y_min else self.ypadding

        needs_update = (
            x_min - x_pad < self._last_xmin
            or x_max + x_pad > self._last_xmax
            or y_min - y_pad < self._last_ymin
            or y_max + y_pad > self._last_ymax
        )

        if needs_update:
            self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
            self.ax.set_ylim(y_min - y_pad, y_max + y_pad)

            # Store current view bounds
            self._last_xmin, self._last_xmax = x_min - x_pad, x_max + x_pad
            self._last_ymin, self._last_ymax = y_min - y_pad, y_max + y_pad

    def _downscale(self) -> tuple[ndarray, DataFrame]:
        """Downscale data for display if it exceeds max_points.

        Returns
        -------
        tuple[ndarray, DataFrame]
            Tuple containing:
            - x : ndarray
                Downscaled x-values
            - y : DataFrame
                Downscaled y-values

        Notes
        -----
        When the number of samples exceeds max_points, this method returns
        evenly spaced samples to reduce rendering overhead.
        """
        if not self._y:
            return array([]), DataFrame()

        x = array(self._x) if self._x else arange(len(self._y))
        y = DataFrame(self._y)
        if len(self._y) <= self.max_points:
            return x, y

        step = len(self._y) // self.max_points
        return x[::step], y[::step]

    def force_redraw(self) -> None:
        """Force an immediate redraw of the plot.

        Useful when you need to update the display outside the normal
        animation interval.
        """
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
