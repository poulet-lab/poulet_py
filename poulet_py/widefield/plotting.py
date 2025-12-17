"""
Plotting and visualization functions for widefield imaging data.

This module provides functions for displaying images and
creating interactive visualizations for mask definition.
"""

import matplotlib.pyplot as plt
import numpy as np

from poulet_py import LOGGER


def show_frame(
    frame: np.ndarray,
    title: str = "Frame",
    cmap: str = "gray",
) -> None:
    """
    Display a single frame with matplotlib.

    Opens a matplotlib figure window showing the provided
    2D array as an image.

    Args:
        frame: 2D numpy array containing the image to display.
        title: Title to show above the image. Default is "Frame".
        cmap: Matplotlib colormap name. Default is "gray".
    """
    try:
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(frame, cmap=cmap)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        LOGGER.info(f"Displayed frame: {title}, shape: {frame.shape}")
    except Exception:
        LOGGER.exception(f"Error displaying frame: {title}")


def create_mask_interactive(
    green_reference: np.ndarray,
    initial_radius: float = 100.0,
) -> dict[str, float] | None:
    """
    Create a circular mask interactively using mouse and keyboard.

    Opens an interactive matplotlib window displaying the reference
    image. The user can click to set the mask center, adjust the
    radius using keyboard shortcuts, and confirm the selection.

    Controls:
        - Left click: Set mask center position
        - B key: Increase radius by 10 pixels
        - S key: Decrease radius by 10 pixels (minimum 10)
        - Enter key: Confirm mask and close window

    Args:
        green_reference: 2D numpy array containing the reference
            image to display during mask creation.
        initial_radius: Starting radius for the circular mask
            in pixels. Default is 100.0.

    Returns:
        Dictionary containing mask parameters:
        - center_x: X coordinate of mask center
        - center_y: Y coordinate of mask center
        - radius: Radius of the circular mask
        Returns None if no mask was created (window closed
        without confirming).
    """
    center: list[int | None] = [None, None]
    radius = initial_radius

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(green_reference, cmap="gray")
    ax.set_title("Click to set center | B=bigger, S=smaller | Enter=confirm", fontsize=12)
    ax.axis("off")

    circle = None

    def update_circle() -> None:
        nonlocal circle
        if circle:
            circle.remove()
        if center[0] is not None and center[1] is not None:
            circle = plt.Circle(
                (center[1], center[0]), radius, fill=False, color="red", linewidth=2
            )
            ax.add_patch(circle)
            fig.canvas.draw()

    def on_click(event: plt.matplotlib.backend_bases.MouseEvent) -> None:
        if event.inaxes != ax:
            return
        if event.button == 1:
            center[0] = int(event.ydata)
            center[1] = int(event.xdata)
            LOGGER.info(f"Center set to: ({center[1]}, {center[0]})")
            update_circle()

    def on_key(event: plt.matplotlib.backend_bases.KeyEvent) -> None:
        nonlocal radius
        if event.key == "b" or event.key == "B":
            radius += 10
            LOGGER.info(f"Radius increased to: {radius:.1f}")
            update_circle()
        elif event.key == "s" or event.key == "S":
            radius = max(10, radius - 10)
            LOGGER.info(f"Radius decreased to: {radius:.1f}")
            update_circle()
        elif event.key == "enter":
            if center[0] is not None and center[1] is not None:
                plt.close(fig)
            else:
                LOGGER.warning("Please click to set center first")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()

    if center[0] is not None and center[1] is not None:
        mask_data = {
            "center_x": float(center[1]),
            "center_y": float(center[0]),
            "radius": float(radius),
        }
        LOGGER.info(
            f"Mask created: center=({mask_data['center_x']}, "
            f"{mask_data['center_y']}), radius={mask_data['radius']}"
        )
        return mask_data

    LOGGER.warning("No mask created")
    return None
