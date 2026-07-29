"""Legacy widefield analysis methods waiting to be integrated with WidefieldData."""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

try:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from numpy import ceil, ndarray, pad, save, savez_compressed, uint16

    from poulet_py import LOGGER

except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

Also ensure: h5py, numpy, pandas, scikit-image, imageio, matplotlib are installed
"""
    raise ImportError(msg) from e


class WidefieldAnalysisToIntegrate:
    

    


    def set_roi(self, roi: tuple[int, int] | dict[str, Any]) -> None:
        """
        Set the region of interest for trace extraction.

        Args:
            roi: Either a tuple (x, y) or a dict with 'center' key.

        Raises:
            ValueError: If tuple doesn't have 2 elements or dict
                missing 'center' key.
            TypeError: If roi is neither tuple nor dict.
        """
        if isinstance(roi, tuple):
            if len(roi) != 2:
                raise ValueError("ROI tuple must have 2 elements (x, y)")
            self.roi = {"center": roi}
        elif isinstance(roi, dict):
            if "center" not in roi:
                raise ValueError("ROI dictionary must contain 'center' key")
            self.roi = roi.copy()
        else:
            raise TypeError(f"ROI must be tuple (x, y) or dict, got {type(roi)}")

        center = self.roi["center"]
        if self.imaging_data is not None:
            _, H, W = self.imaging_data.shape
            if center[0] < 0 or center[0] >= W or center[1] < 0 or center[1] >= H:
                LOGGER.warning(
                    f"ROI center ({center[0]}, {center[1]}) is outside image bounds ({W}, {H})"
                )

        LOGGER.info(f"ROI set: center=({center[0]}, {center[1]})")

    def calculate_percentile_centroid_roi(
        self, data: ndarray[Any, Any], percentile: float = 95.0
    ) -> tuple[int, int]:
        """
        Find ROI center from high-intensity pixels.

        Computes the centroid of pixels above the specified
        percentile threshold.

        Args:
            data: 2D image array.
            percentile: Threshold percentile (0-100). Default 95.

        Returns:
            Tuple (x, y) of centroid coordinates.
        """
        return wf_roi.centroid_from_percentile(data, percentile)

    def calculate_trace_within_roi(
        self,
        data: ndarray[Any, Any] | None = None,
        roi: tuple[int, int] | dict[str, Any] | None = None,
        diameter: float = 50.0,
    ) -> ndarray[Any, Any] | None:
        """
        Extract mean fluorescence trace from circular ROI.

        Computes the mean pixel value within a circular region
        for each frame.

        Args:
            data: 3D array. Defaults to imaging_data.
            roi: ROI center as tuple (x, y) or dict with 'center'.
                Defaults to self.roi.
            diameter: ROI diameter in pixels. Default 50.

        Returns:
            1D array of mean values per frame, or None on error.
        """
        if data is None:
            if self.imaging_data is None:
                LOGGER.warning("No data provided and imaging_data not loaded")
                return None
            data = self.imaging_data

        if roi is None:
            if self.roi is None:
                LOGGER.error("No ROI provided and self.roi not set")
                return None
            roi = self.roi

        if isinstance(roi, dict):
            center = roi.get("center")
            if center is None:
                LOGGER.error("ROI dictionary must contain 'center' key")
                return None
        elif isinstance(roi, tuple):
            center = roi
        else:
            LOGGER.error(f"ROI must be tuple (x, y) or dict, got {type(roi)}")
            return None

        return wf_roi.trace_within_circular_roi(data, center, diameter)
