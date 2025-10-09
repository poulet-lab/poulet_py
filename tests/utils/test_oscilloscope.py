# test_oscilloscope.py
import matplotlib
import pytest

matplotlib.use("Agg")  # prevent GUI during tests

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy.testing import assert_array_almost_equal
from pandas import DataFrame

from poulet_py import Oscilloscope


@pytest.fixture
def osc():
    """Create a fresh oscilloscope instance for tests."""
    osc = Oscilloscope(max_samples=10, max_points=5)
    return osc


def test_add_data_and_downscale_with_x(osc: Oscilloscope):
    osc.add_data({"a": 1}, x=0)
    osc.add_data({"a": 2}, x=1)
    x, y = osc._downscale()
    assert np.array_equal(x, np.array([0, 1]))
    assert isinstance(y, DataFrame)
    assert list(y.columns) == ["a"]
    assert y.iloc[-1, 0] == 2


def test_add_data_and_downscale_auto_x(osc: Oscilloscope):
    osc.add_data({"a": 10})
    osc.add_data({"a": 20})
    x, y = osc._downscale()
    assert np.array_equal(x, np.array([0, 1]))
    assert y["a"].tolist() == [10, 20]


def test_downscale_large_dataset(osc: Oscilloscope):
    for i in range(50):
        osc.add_data({"a": i}, x=i)
    x, y = osc._downscale()
    # Should be reduced to <= max_points
    assert len(x) <= osc.max_points
    assert len(y) <= osc.max_points


def test_update_creates_linecollection(osc: Oscilloscope):
    for i in range(5):
        osc.add_data({"a": i, "b": i + 1}, x=i)

    artists = osc._update(0)
    assert len(artists) == 1
    assert isinstance(artists[0], LineCollection)
    # Legend should contain 2 labels
    labels = [h.get_label() for h in osc._legend_handles]
    assert "a" in labels and "b" in labels


def test_update_view_auto_and_fixed(osc: Oscilloscope):
    # Auto mode
    for i in range(3):
        osc.add_data({"a": i}, x=i)
    x, y = osc._downscale()
    osc._update_view(y, x)
    xlim = osc.ax.get_xlim()
    ylim = osc.ax.get_ylim()

    assert xlim[0] < 0
    assert ylim[1] > 2

    # Fixed mode
    osc.xlim = (0, 1)
    osc.ylim = (-1, 1)
    osc._update_view(y, x)
    xlim = osc.ax.get_xlim()
    ylim = osc.ax.get_ylim()

    assert_array_almost_equal(xlim, (0, 1), 1, "Error in fixed mode")


def test_start_and_stop(osc: Oscilloscope):
    osc.start()
    assert isinstance(osc._animation, FuncAnimation)
    osc.stop()
    assert osc._animation is None
    assert osc._line_collection is None
    assert osc._legend_handles == []
    assert len(osc._x) == 0 and len(osc._y) == 0


def test_force_redraw_does_not_crash(osc: Oscilloscope):
    osc.force_redraw()  # just ensure it runs without error
