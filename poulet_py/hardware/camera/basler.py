try:
    import csv
    import datetime
    import json
    import os
    import time
    from typing import Literal

    import cv2
    from pypylon import pylon

    from poulet_py.config.logging import LOGGER, setup_logging
except ImportError as e:
    msg = """
Missing 'camera' module. Install options:
- Dedicated:    pip install poulet_py[camera]
- Module:       pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BaslerCamera:
    """
    A class to interact with multiple Basler cameras using pypylon and OpenCV.
    Each camera will record to its own video file and log timestamps to a CSV.
    """

    def __init__(self, max_cameras=2):
        """
        Initializes the BaslerCamera object by enumerating devices and
        attaching up to max_cameras.

        Args:
            max_cameras (int): The maximum number of cameras to use.
        """
        tlFactory = pylon.TlFactory.GetInstance()

        self.devices = tlFactory.EnumerateDevices()
        if len(self.devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        self.max_cameras = min(len(self.devices), max_cameras)

        self.cameras = pylon.InstantCameraArray(self.max_cameras)
        for i in range(self.max_cameras):
            self.cameras[i].Attach(tlFactory.CreateDevice(self.devices[i]))
            LOGGER.info(f"Using device {self.cameras[i].GetDeviceInfo().GetModelName()}")

        self.frames_per_second = None
        self.outs = {}  # VideoWriter objects keyed by camera index
        self.timestamps_files = {}  # Timestamps CSV file path per camera
        self.frame_numbers = {}  # Frame count for each camera
        self.start_time = None
        self.error_log_file = None

    def set_frames_per_second(self, frames_per_second):
        """
        Sets the frame rate for each camera.

        Args:
            frames_per_second (float): Desired frame rate in frames per second.
        """
        self.frames_per_second = frames_per_second
        for cam in self.cameras:
            if not cam.IsOpen():
                cam.Open()
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(frames_per_second)

    def set_error_log_path(self, path, file_name):
        """
        Sets the error log file.

        Args:
            path (str): Directory for the error log.
            file_name (str): Name of the error log file.
        """
        self.error_log_file = os.path.join(path, file_name)
        self._file_logger = LOGGER.getChild(f"hardware.camera.basler.{id(self)}")
        setup_logging(self._file_logger, level="error", file=self.error_log_file)

    def set_output_file(self, path, extra_name, base_file_name="basler-camera"):
        """
        Sets up output video files and timestamp CSV files for all cameras.

        Args:
            path (str): Directory to save the output files.
            extra_name (str): Extra name to add to the file names.
            base_file_name (str): Base name for the files.
        """
        os.makedirs(path, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"MP4V")

        for i, cam in enumerate(self.cameras):
            if not cam.IsOpen():
                cam.Open()

            frame_width = int(cam.Width.Value)
            frame_height = int(cam.Height.Value)

            self.output_file_name = f"{base_file_name}_{extra_name}_cam{i}.mp4"
            self.output_path = os.path.join(path)
            self.output_file = os.path.join(self.output_path, self.output_file_name)
            self.outs[i] = cv2.VideoWriter(
                self.output_file,
                fourcc,
                self.frames_per_second,
                (frame_width, frame_height),
            )

            timestamps_file = os.path.join(
                self.output_path, f"{base_file_name}_{extra_name}_cam{i}_timestamps.csv"
            )
            self.timestamps_files[i] = timestamps_file

            if not os.path.isfile(timestamps_file):
                with open(timestamps_file, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["timestamp"])

            self.frame_numbers[i] = 1
            cam.Close()

    def save_timestamp(self, camera_index, timestamp):
        """
        Save a timestamp to the CSV file for the specified camera.

        Args:
            camera_index (int): Index of the camera.
            timestamp (float): Timestamp to record.
        """
        try:
            with open(self.timestamps_files[camera_index], mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp])
        except Exception as e:
            self.log_error(e)

    def start_streaming(self):
        """
        Starts the grabbing (streaming) for all cameras.
        """
        self.start_time = time.time()
        for cam in self.cameras:
            if not cam.IsOpen():
                cam.Open()
        self.cameras.StartGrabbing()
        LOGGER.info("Started streaming on all cameras.")

    def stop_streaming(self):
        """
        Stops the streaming and closes all cameras and video writers.
        """
        self.cameras.StopGrabbing()
        for i, cam in enumerate(self.cameras):
            if cam.IsOpen():
                cam.Close()
            if i in self.outs and self.outs[i] is not None:
                self.outs[i].release()
        LOGGER.info("Stopped streaming and closed all cameras.")

    def capture_frame(self):
        """
        Captures a single frame from whichever camera has a frame ready.
        The frame is written to its corresponding video file and timestamp logged.
        """
        try:
            if not self.cameras.IsGrabbing():
                return

            grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            camera_index = grabResult.GetCameraContext()

            if grabResult.GrabSucceeded():
                img = grabResult.Array
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.outs[camera_index].write(img_bgr)

                timestamp = time.time() - self.start_time
                self.save_timestamp(camera_index, timestamp)

                self.frame_numbers[camera_index] += 1

            grabResult.Release()
        except Exception as e:
            self.log_error(e)

    def stream_video(self, window_width=None, window_height=None):
        """
        Streams the live video feed from all cameras. Each camera is shown in its own window.

        Args:
            window_width (int, optional): Width to resize the window.
            window_height (int, optional): Height to resize the window.
        """
        LOGGER.info("Press 'e' to quit the video stream.")

        while self.cameras.IsGrabbing():
            try:
                grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                camera_index = grabResult.GetCameraContext()

                if grabResult.GrabSucceeded():
                    img = grabResult.Array
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    if window_width is not None and window_height is not None:
                        img_bgr = cv2.resize(img_bgr, (round(window_width), round(window_height)))

                    window_name = f"Camera {camera_index}"
                    cv2.imshow(window_name, img_bgr)

                grabResult.Release()

                if cv2.waitKey(1) & 0xFF == ord("e"):
                    break

            except Exception as e:
                self.log_error(e)
                break

        cv2.destroyAllWindows()

    def save_metadata(self, base_file_name="basler-camera", extra_name=""):
        """
        Saves metadata about the recording for each camera to a JSON file.

        Args:
            path (str): Directory to save the metadata files.
            base_file_name (str, optional): Base name for the metadata files.
            extra_name (str, optional): Extra name to add to the file names.
        """
        for i, cam in enumerate(self.cameras):
            metadata_file_name = f"{base_file_name}_{extra_name}_cam{i}.json"
            metadata_path = os.path.join(self.output_path, metadata_file_name)

            if not cam.IsOpen():
                cam.Open()
            data = {
                "camera": cam.GetDeviceInfo().GetModelName(),
                "width": cam.Width.Value,
                "height": cam.Height.Value,
                "frame_rate_fps": self.frames_per_second,
                "output_file": f"{base_file_name}_{extra_name}_cam{i}.mp4",
                "number_of_frames": self.frame_numbers.get(i, 0),
            }
            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=4)
            cam.Close()

    def recording(
        self,
        data_save_folder: str,
        cage_id: str,
        n_mouse: int,
        condition: str,
        mouse_ids: list = [],
        duration_s: int = 10,
        buffer_s=10,
        total_rec=4,
        fps: int = 30,
        video_format: Literal["mp4", "avi"] = "mp4",
    ):
        self.set_frames_per_second(30)
        self.start_streaming()

        try:
            LOGGER.info("Stream preview started...")
            time.sleep(5)

            for rec_count in range(total_rec):
                start_time = time.time()
                LOGGER.info("Recording started....")

                current_time = datetime.datetime.now().strftime("%H%M%S")
                self.set_output_file(
                    data_save_folder,
                    f"recording_{rec_count + 1}_{current_time}",
                )

                try:
                    LOGGER.info("Starting capture...")
                    self.set_timer(start_time)
                    LOGGER.info("Recording finished")

                except Exception:
                    LOGGER.exception("Error during capture")

                finally:
                    LOGGER.info(f"Frames captured: {self.frame_number}")
                    self.save_metadata()

                    if rec_count < total_rec - 1:
                        LOGGER.info("Buffer period")
                        time.sleep(buffer_s)

        finally:
            self.stop_streaming()

    def set_timer(self, start_time):
        """
        Sets the timer for the camera.

        Args:
            start_time (float): The time at which the camera recording started.
        """
        self.start_time = start_time

    def log_error(self, error_message):
        LOGGER.error(error_message)
        file_logger = getattr(self, "_file_logger", None)
        if file_logger is not None:
            file_logger.error(error_message)
