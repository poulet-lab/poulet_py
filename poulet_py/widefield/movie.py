"""
Movie creation functions for widefield imaging data.

This module provides functions for generating video files
from widefield imaging data arrays.
"""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio

from poulet_py import LOGGER

if TYPE_CHECKING:
    from .analysis import WidefieldAnalysis


def create_movie_from_array(
    data: np.ndarray,
    output_path: Path,
    fps: int = 10,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_callback: Callable[..., None] | None = None,
    wf_analysis: "WidefieldAnalysis | None" = None,
) -> Path | None:
    """
    Create an MP4 movie from a 3D numpy array.

    Generates a video file by rendering each frame of the input
    array as an image with matplotlib and encoding them into
    an MP4 video using FFMPEG.

    Args:
        data: 3D numpy array with shape (frames, height, width).
        output_path: Path where the MP4 file will be saved.
        fps: Frames per second for the output video. Default is 10.
        cmap: Matplotlib colormap name. Default is "gray".
        vmin: Minimum value for colormap scaling. If None, uses
            the minimum value in the data.
        vmax: Maximum value for colormap scaling. If None, uses
            the maximum value in the data.
        frame_callback: Optional callback function called for each
            frame with signature:
            callback(fig, ax, frame_idx, frame_data, wf_analysis)
            Can be used to add custom annotations or overlays.
        wf_analysis: Optional WidefieldAnalysis instance passed
            to the frame_callback for accessing trial metadata.

    Returns:
        Path to the created video file, or None on error.
    """
    if data.ndim != 3:
        LOGGER.error(f"Expected 3D array (T, H, W), got: {data.shape}")
        return None

    T, H, W = data.shape
    LOGGER.info(f"Creating movie from {T} frames ({H}x{W})")

    output_path = Path(output_path)

    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())

    try:
        frames: list[Any] = []
        for frame_idx in range(T):
            frame = data[frame_idx]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Frame {frame_idx + 1}/{T}", fontsize=14)
            ax.axis("off")

            if frame_callback is not None:
                frame_callback(fig, ax, frame_idx, frame, wf_analysis)
            else:
                plt.tight_layout()

            buf = BytesIO()
            if frame_callback is not None:
                fig.savefig(buf, format="png", dpi=100, bbox_inches=None, pad_inches=0.0)
            else:
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            frame_img = skio.imread(buf)
            frames.append(frame_img)
            buf.close()
            plt.close(fig)

        LOGGER.info(f"Saving movie to: {output_path}")
        writer = imageio.get_writer(str(output_path), format="FFMPEG", fps=fps, codec="libx264")
        for frame in frames:
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            writer.append_data(frame)
        writer.close()
        LOGGER.info(f"Movie saved: {output_path}")

        return output_path

    except Exception:
        LOGGER.exception("Error creating movie")
        return None
