import logging
import platform
from threading import Thread

import cv2
import matplotlib.pyplot as plt


class UVCCameraStream:
    def __init__(
        self, src=0, width=None, height=None, fps=None, api_preference=None
    ):
        """
        Initialize the UVC camera stream with cross-platform support.

        Parameters:
        - src: Camera index or path (default 0)
        - width: Desired frame width (optional)
        - height: Desired frame height (optional)
        - fps: Desired frames per second (optional)
        - api_preference: Optional OpenCV API preference for video capture
        """
        self.system = platform.system().lower()
        self.logger = logging.getLogger("UVCCameraStream")
        self.logger.setLevel(logging.INFO)

        # Set up console handler if no handlers are configured
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Platform-specific initialization
        self.logger.info(f"Initializing camera on {self.system}")

        # Try different API preferences based on platform
        if api_preference is not None:
            self.stream = cv2.VideoCapture(src, api_preference)
        else:
            self.stream = cv2.VideoCapture(src)

            # If default fails, try platform-specific backends
            if not self.stream.isOpened():
                self.logger.warning(
                    "Default backend failed, trying platform-specific alternatives"
                )
                backends = self._get_platform_backends()
                for backend in backends:
                    self.stream = cv2.VideoCapture(src, backend)
                    if self.stream.isOpened():
                        self.logger.info(f"Success with backend: {backend}")
                        break

        if not self.stream.isOpened():
            raise RuntimeError(
                f"Could not open camera with source {src} on {self.system}"
            )

        # Set camera properties if specified
        if width is not None:
            self._set_property(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._set_property(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self._set_property(cv2.CAP_PROP_FPS, fps)

        # Read first frame to initialize
        (self.grabbed, self.frame) = self.stream.read()

        # Initialize thread control variables
        self.stopped = False
        self.paused = False

    def _get_platform_backends(self):
        """Return preferred backends based on platform"""
        if self.system == "linux":
            return [
                cv2.CAP_V4L2,  # V4L2 for modern Linux
                cv2.CAP_V4L,  # Legacy V4L
            ]
        elif self.system == "windows":
            return [
                cv2.CAP_DSHOW,  # DirectShow
                cv2.CAP_MSMF,  # Microsoft Media Foundation
            ]
        elif self.system == "darwin":  # macOS
            return [
                cv2.CAP_AVFOUNDATION,  # AVFoundation
                cv2.CAP_QT,  # QuickTime (legacy)
            ]
        else:
            return []

    def _set_property(self, prop, value):
        """Attempt to set property with error handling"""
        if not self.stream.set(prop, value):
            self.logger.warning(f"Failed to set property {prop} to {value}")

    def start(self):
        """Start the thread to read frames from the video stream"""
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Continuously grab frames from the stream"""
        while not self.stopped:
            if not self.paused:
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.logger.error("Frame grab failed, stopping stream")
                    self.stop()
                    break
                self.frame = frame
            else:
                # Small sleep when paused to reduce CPU usage
                import time

                time.sleep(0.1)

    def read(self):
        """Return the most recent frame"""
        return self.frame

    def stop(self):
        """Stop the stream and release resources"""
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)

    def pause(self):
        """Pause the stream"""
        self.paused = True

    def resume(self):
        """Resume the stream"""
        self.paused = False

    def release(self):
        """Release the camera resource"""
        self.stop()
        if hasattr(self, "stream") and self.stream.isOpened():
            self.stream.release()

    def plot_stream(self, figsize=(10, 8), cmap=None):
        """
        Plot the camera stream using matplotlib in real-time.

        Parameters:
        - figsize: Size of the matplotlib figure
        - cmap: Color map to use (None for RGB, 'gray' for grayscale)
        """
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=figsize)

        try:
            while not self.stopped:
                frame = self.read()
                if frame is None:
                    continue

                # Convert BGR to RGB for matplotlib
                if len(frame.shape) == 3:  # Color image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Clear previous image and plot new one
                ax.clear()
                if cmap:
                    ax.imshow(frame, cmap=cmap)
                else:
                    ax.imshow(frame)
                ax.axis("off")
                plt.title("UVC Camera Stream")

                # Pause to allow the plot to update
                plt.pause(0.001)

                # Break if figure is closed
                if not plt.fignum_exists(fig.number):
                    break

        except KeyboardInterrupt:
            self.logger.info("Stream stopped by user")
        except Exception as e:
            self.logger.error(f"Error in plot_stream: {e!s}")
        finally:
            plt.ioff()
            self.release()

    def show_opencv(self, window_name="UVC Camera"):
        """
        Display stream using OpenCV's imshow (faster but less flexible than matplotlib)
        Press 'q' to quit.
        """
        try:
            while not self.stopped:
                frame = self.read()
                if frame is None:
                    continue

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Small delay to prevent high CPU usage
                import time

                time.sleep(1 / 30)  # ~30 fps

        except KeyboardInterrupt:
            self.logger.info("Stream stopped by user")
        except Exception as e:
            self.logger.error(f"Error in show_opencv: {e!s}")
        finally:
            cv2.destroyAllWindows()
            self.release()

    def get_properties(self):
        """Print current camera properties"""
        props = {
            "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
            "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
            "FPS": cv2.CAP_PROP_FPS,
            "Brightness": cv2.CAP_PROP_BRIGHTNESS,
            "Contrast": cv2.CAP_PROP_CONTRAST,
            "Saturation": cv2.CAP_PROP_SATURATION,
            "Hue": cv2.CAP_PROP_HUE,
            "Gain": cv2.CAP_PROP_GAIN,
            "Exposure": cv2.CAP_PROP_EXPOSURE,
        }

        self.logger.info("Camera Properties:")
        for name, prop in props.items():
            value = self.stream.get(prop)
            self.logger.info(f"{name}: {value}")
