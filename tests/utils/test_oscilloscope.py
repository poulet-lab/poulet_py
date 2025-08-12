from collections import deque
from threading import Lock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# We'll mock the matplotlib imports to avoid GUI dependencies in tests
with patch.dict(
    "sys.modules",
    {
        "matplotlib.animation": MagicMock(),
        "matplotlib.collections": MagicMock(),
        "matplotlib.lines": MagicMock(),
        "matplotlib.pyplot": MagicMock(),
        "numpy": MagicMock(),
        "pandas": MagicMock(),
    },
):
    from poulet_py import Oscilloscope


@pytest.fixture
def mock_oscilloscope():
    """Fixture that provides a mocked Oscilloscope instance."""
    with (
        patch("matplotlib.pyplot.subplots"),
        patch("matplotlib.animation.FuncAnimation"),
        patch("matplotlib.pyplot.show"),
    ):
        osc = Oscilloscope(max_samples=100, max_points=50)
        osc.ax = MagicMock()
        osc.fig = MagicMock()
        osc._line_collection = MagicMock()
        return osc


def test_initialization(mock_oscilloscope):
    """Test that the Oscilloscope initializes correctly."""
    osc = mock_oscilloscope

    assert osc.max_samples == 100
    assert osc.max_points == 50
    assert isinstance(osc._data_lock, Lock)
    assert isinstance(osc._x, deque)
    assert isinstance(osc._y, deque)
    assert osc._x.maxlen == 100
    assert osc._y.maxlen == 100

    # Verify plot setup
    osc.ax.set_title.assert_called_once_with("Real-time Data")
    osc.ax.set_xlabel.assert_called_once_with("X")
    osc.ax.set_ylabel.assert_called_once_with("Y")
    osc.ax.grid.assert_called_once_with(True)


def test_add_data(mock_oscilloscope):
    """Test adding data to the oscilloscope."""
    osc = mock_oscilloscope

    # Test with auto-incremented x
    osc.add_data({"series1": 1.0, "series2": 2.0})
    with osc._data_lock:
        assert len(osc._y) == 1
        assert osc._y[0] == {"series1": 1.0, "series2": 2.0}
        assert len(osc._x) == 0  # No x provided

    # Test with explicit x
    osc.add_data({"series1": 1.5, "series2": 2.5}, x=10)
    with osc._data_lock:
        assert len(osc._y) == 2
        assert osc._x[0] == 10


def test_start_stop(mock_oscilloscope):
    """Test starting and stopping the animation."""
    osc = mock_oscilloscope

    # Test start
    osc.start()
    assert osc._animation is not None
    osc.fig.canvas.draw_idle.assert_called_once()

    # Test stop
    osc.stop()
    osc._animation.event_source.stop.assert_called_once()
    osc.fig.canvas.close.assert_called_once()
    assert osc._animation is None
    with osc._data_lock:
        assert len(osc._x) == 0
        assert len(osc._y) == 0


def test_downscale(mock_oscilloscope):
    """Test the downscaling functionality."""
    osc = mock_oscilloscope

    # Test with empty data
    x, y = osc._downscale()
    assert len(x) == 0
    assert y.empty

    # Test with small amount of data (no downscaling)
    with osc._data_lock:
        for i in range(10):
            osc._y.append({"a": i, "b": i * 2})
    x, y = osc._downscale()
    assert len(x) == 10
    assert len(y) == 10

    # Test with data that needs downscaling
    with osc._data_lock:
        osc._y = deque(maxlen=100)
        for i in range(100):
            osc._y.append({"a": i, "b": i * 2})
    x, y = osc._downscale()
    assert len(x) <= osc.max_points
    assert len(y) <= osc.max_points


def test_update_view(mock_oscilloscope):
    """Test the view update functionality."""
    osc = mock_oscilloscope
    osc.xlim = "auto"
    osc.ylim = "auto"

    # Create test data
    x = np.array([0, 1, 2, 3, 4])
    y = pd.DataFrame({"series1": [1, 2, 3, 4, 5], "series2": [5, 4, 3, 2, 1]})

    # Test auto limits
    osc._update_view(y, x)
    osc.ax.set_xlim.assert_called_once()
    osc.ax.set_ylim.assert_called_once()

    # Test fixed limits
    osc.xlim = (0, 10)
    osc.ylim = (-5, 5)
    osc._update_view(y, x)
    osc.ax.set_xlim.assert_called_with(0, 10)
    osc.ax.set_ylim.assert_called_with(-5, 5)


def test_update(mock_oscilloscope):
    """Test the animation update function."""
    osc = mock_oscilloscope

    # Setup some test data
    with osc._data_lock:
        osc._y.append({"series1": 1.0, "series2": 2.0})
        osc._y.append({"series1": 1.5, "series2": 2.5})

    # Mock the downscale method to return known data
    test_x = np.array([0, 1])
    test_y = pd.DataFrame({"series1": [1.0, 1.5], "series2": [2.0, 2.5]})
    osc._downscale = MagicMock(return_value=(test_x, test_y))

    # Call update
    result = osc._update(frame=0)

    # Verify calls
    osc._downscale.assert_called_once()
    osc._line_collection.set_segments.assert_called_once()
    osc._line_collection.set_color.assert_called_once()
    osc.ax.legend.assert_called_once()
    assert result == [osc._line_collection]


def test_force_redraw(mock_oscilloscope):
    """Test the force redraw functionality."""
    osc = mock_oscilloscope
    osc.force_redraw()
    osc.fig.canvas.draw_idle.assert_called_once()
    osc.fig.canvas.flush_events.assert_called_once()


def test_import_error():
    """Test that the import error is raised when dependencies are missing."""
    with patch.dict("sys.modules", {"matplotlib.animation": None}):
        with pytest.raises(ImportError, match="Missing 'osc' module"):
            pass  # Replace with your actual module name
