try:
    import csv
    import datetime
    import json
    import os
    import time
    from typing import Literal

    import cv2
    from pypylon import pylon

    from poulet_py import LOGGER, setup_logging
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
        self.output_files = {}  # Output video file name keyed by camera index
        self.timestamps_files = {}  # Timestamps CSV file path per camera
        self.frame_numbers = {}  # Frame count for each camera
        self.last_block_ids = {}  # Last seen block id by camera index
        self.dropped_frames = {}  # Dropped frame count by camera index
        self.start_time = None
        self.error_log_file = None
        self.output_path = ""

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

    def set_output_file(
        self,
        path,
        extra_name,
        base_file_name="basler-camera",
        video_format: Literal["mp4", "avi"] = "mp4",
    ):
        """
        Sets up output video files and timestamp CSV files for all cameras.

        Args:
            path (str): Directory to save the output files.
            extra_name (str): Extra name to add to the file names.
            base_file_name (str): Base name for the files.
        """
        os.makedirs(path, exist_ok=True)

        output_ext = video_format.lower()
        if output_ext == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        elif output_ext == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            raise ValueError(f"Unsupported video format: {video_format}")

        for i, cam in enumerate(self.cameras):
            if not cam.IsOpen():
                cam.Open()

            frame_width = int(cam.Width.Value)
            frame_height = int(cam.Height.Value)

            self.output_file_name = f"{base_file_name}_{extra_name}_cam{i}.{output_ext}"
            self.output_path = os.path.join(path)
            self.output_file = os.path.join(self.output_path, self.output_file_name)
            self.outs[i] = cv2.VideoWriter(
                self.output_file,
                fourcc,
                self.frames_per_second,
                (frame_width, frame_height),
            )
            self.output_files[i] = self.output_file_name

            timestamps_file = os.path.join(
                self.output_path, f"{base_file_name}_{extra_name}_cam{i}_timestamps.csv"
            )
            self.timestamps_files[i] = timestamps_file

            if not os.path.isfile(timestamps_file):
                with open(timestamps_file, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            "frame_number",
                            "timestamp_iso",
                            "time_since_start_s",
                            "camera_index",
                            "block_id",
                            "camera_timestamp_ns",
                        ]
                    )

            self.frame_numbers[i] = 0
            self.last_block_ids[i] = None
            self.dropped_frames[i] = 0
            cam.Close()

    def save_timestamp(
        self,
        camera_index,
        timestamp,
        frame_number=None,
        timestamp_iso=None,
        block_id=None,
        camera_timestamp_ns=None,
    ):
        """
        Save a timestamp to the CSV file for the specified camera.

        Args:
            camera_index (int): Index of the camera.
            timestamp (float): Timestamp to record.
        """
        try:
            if frame_number is None:
                frame_number = self.frame_numbers.get(camera_index, 0)
            if timestamp_iso is None:
                timestamp_iso = datetime.datetime.fromtimestamp(time.time()).isoformat(
                    timespec="milliseconds"
                )
            with open(self.timestamps_files[camera_index], mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        frame_number,
                        timestamp_iso,
                        round(float(timestamp), 6),
                        camera_index,
                        block_id if block_id is not None else "",
                        camera_timestamp_ns if camera_timestamp_ns is not None else "",
                    ]
                )
        except Exception as e:
            self.log_error(e)

    def start_streaming(self, grab_strategy=pylon.GrabStrategy_LatestImageOnly):
        """
        Starts the grabbing (streaming) for all cameras.
        """
        self.start_time = time.time()
        for cam in self.cameras:
            if not cam.IsOpen():
                cam.Open()
        self.cameras.StartGrabbing(grab_strategy)
        LOGGER.info("Started streaming on all cameras.")

    def stop_streaming(self):
        """
        Stops the streaming and closes all cameras and video writers.
        """
        if self.cameras.IsGrabbing():
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
                if camera_index in self.outs:
                    self.outs[camera_index].write(img_bgr)

                timestamp = time.time() - self.start_time
                self.frame_numbers[camera_index] = self.frame_numbers.get(camera_index, 0) + 1
                block_id = int(getattr(grabResult, "BlockID", -1))
                previous_block_id = self.last_block_ids.get(camera_index)
                if previous_block_id is not None and block_id > previous_block_id + 1:
                    self.dropped_frames[camera_index] += block_id - previous_block_id - 1
                self.last_block_ids[camera_index] = block_id
                self.save_timestamp(
                    camera_index,
                    timestamp,
                    frame_number=self.frame_numbers[camera_index],
                    timestamp_iso=datetime.datetime.fromtimestamp(time.time()).isoformat(
                        timespec="milliseconds"
                    ),
                    block_id=block_id,
                    camera_timestamp_ns=getattr(grabResult, "TimeStamp", None),
                )

            grabResult.Release()
        except Exception as e:
            self.log_error(e)

    def run_capture_loop(
        self,
        duration_s=None,
        show_preview=False,
        preview_key="e",
        window_width=None,
        window_height=None,
        record=True,
        max_consecutive_errors=20,
    ):
        """
        Runs one acquisition loop that can preview and record simultaneously.

        Args:
            duration_s (float | None): Recording duration in seconds.
            show_preview (bool): If True, shows one preview window per camera.
            preview_key (str): Keyboard key used to stop preview loop.
            window_width (int | None): Optional preview width.
            window_height (int | None): Optional preview height.
            record (bool): If True, writes video frames and timestamps.

        Returns:
            dict: Run diagnostics summary.
        """
        run_start = time.time()
        stop_requested_by_key = False
        consecutive_errors = 0

        while self.cameras.IsGrabbing():
            if duration_s is not None and (time.time() - run_start) >= duration_s:
                break

            try:
                grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                camera_index = grabResult.GetCameraContext()

                if grabResult.GrabSucceeded():
                    img = grabResult.Array
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    if window_width is not None and window_height is not None:
                        img_bgr = cv2.resize(img_bgr, (round(window_width), round(window_height)))

                    block_id = int(getattr(grabResult, "BlockID", -1))
                    previous_block_id = self.last_block_ids.get(camera_index)
                    if previous_block_id is not None and block_id > previous_block_id + 1:
                        self.dropped_frames[camera_index] += block_id - previous_block_id - 1
                    self.last_block_ids[camera_index] = block_id

                    if record and camera_index in self.outs:
                        self.outs[camera_index].write(img_bgr)
                        elapsed_s = time.time() - self.start_time
                        self.frame_numbers[camera_index] = (
                            self.frame_numbers.get(camera_index, 0) + 1
                        )
                        self.save_timestamp(
                            camera_index,
                            elapsed_s,
                            frame_number=self.frame_numbers[camera_index],
                            timestamp_iso=datetime.datetime.fromtimestamp(time.time()).isoformat(
                                timespec="milliseconds"
                            ),
                            block_id=block_id,
                            camera_timestamp_ns=getattr(grabResult, "TimeStamp", None),
                        )

                    if show_preview:
                        cv2.imshow(f"Camera {camera_index}", img_bgr)

                    consecutive_errors = 0

                grabResult.Release()

                if show_preview and cv2.waitKey(1) & 0xFF == ord(preview_key):
                    stop_requested_by_key = True
                    break

            except Exception as e:
                consecutive_errors += 1
                self.log_error(
                    f"Capture loop error ({consecutive_errors}/{max_consecutive_errors}): {type(e).__name__}: {e!r}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    self.log_error("Stopping capture loop after repeated camera errors.")
                    break
                continue

        duration_actual_s = max(time.time() - run_start, 1e-9)
        per_camera = {}
        for i in range(self.max_cameras):
            frames_captured = self.frame_numbers.get(i, 0)
            dropped = self.dropped_frames.get(i, 0)
            expected_frames = None
            if self.frames_per_second is not None and duration_s is not None:
                expected_frames = int(round(self.frames_per_second * duration_s))
            per_camera[i] = {
                "frames_captured": frames_captured,
                "dropped_frames": dropped,
                "expected_frames": expected_frames,
                "effective_fps": round(frames_captured / duration_actual_s, 3),
            }

        diagnostics = {
            "run_start_iso": datetime.datetime.fromtimestamp(run_start).isoformat(
                timespec="seconds"
            ),
            "run_end_iso": datetime.datetime.fromtimestamp(time.time()).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(duration_actual_s, 3),
            "configured_fps": self.frames_per_second,
            "stop_requested_by_key": stop_requested_by_key,
            "per_camera": per_camera,
        }

        if show_preview:
            cv2.destroyAllWindows()

        return diagnostics

    def stream_video(self, window_width=None, window_height=None):
        """
        Streams the live video feed from all cameras. Each camera is shown in its own window.

        Args:
            window_width (int, optional): Width to resize the window.
            window_height (int, optional): Height to resize the window.
        """
        LOGGER.info("Press 'e' to quit the video stream.")

        self.run_capture_loop(
            duration_s=None,
            show_preview=True,
            preview_key="e",
            window_width=window_width,
            window_height=window_height,
            record=False,
        )

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
                "output_file": self.output_files.get(
                    i, f"{base_file_name}_{extra_name}_cam{i}.mp4"
                ),
                "number_of_frames": self.frame_numbers.get(i, 0),
            }
            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=4)
            cam.Close()

    def save_diagnostics(self, diagnostics, base_file_name="basler-camera", extra_name=""):
        diagnostics_file_name = f"{base_file_name}_{extra_name}_diagnostics.json"
        diagnostics_path = os.path.join(self.output_path, diagnostics_file_name)
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=4)
        return diagnostics_path

    def recording(
        self,
        data_save_folder: str,
        cage_id: str,
        n_mouse: int,
        condition: str,
        mouse_ids: list | None = None,
        duration_s: float = 10,
        buffer_s=10,
        total_rec=4,
        fps: int = 30,
        video_format: Literal["mp4", "avi"] = "mp4",
        show_preview: bool = False,
        preview_key: str = "e",
        window_width=None,
        window_height=None,
        grab_strategy=pylon.GrabStrategy_LatestImageOnly,
    ):
        session_metadata = {
            "cage_id": cage_id,
            "n_mouse": n_mouse,
            "condition": condition,
            "mouse_ids": mouse_ids or [],
        }
        self.set_frames_per_second(fps)
        saved_diagnostics_paths = []

        try:
            LOGGER.info("Stream preview started...")
            time.sleep(5)

            for rec_count in range(total_rec):
                start_time = time.time()
                LOGGER.info("Recording started....")

                current_time = datetime.datetime.now().strftime("%H%M%S")
                extra_name = f"recording_{rec_count + 1}_{current_time}"
                self.set_output_file(
                    data_save_folder,
                    extra_name,
                    video_format=video_format,
                )
                self.start_streaming(grab_strategy=grab_strategy)

                try:
                    LOGGER.info("Starting capture...")
                    self.set_timer(start_time)
                    diagnostics = self.run_capture_loop(
                        duration_s=duration_s,
                        show_preview=show_preview,
                        preview_key=preview_key,
                        window_width=window_width,
                        window_height=window_height,
                        record=True,
                    )
                    diagnostics["session_metadata"] = session_metadata
                    diagnostics["recording_index"] = rec_count + 1
                    diagnostics_path = self.save_diagnostics(diagnostics, extra_name=extra_name)
                    saved_diagnostics_paths.append(diagnostics_path)
                    LOGGER.info(f"Recording finished. Diagnostics: {diagnostics_path}")

                except Exception:
                    LOGGER.exception("Error during capture")

                finally:
                    LOGGER.info(f"Frames captured: {self.frame_numbers}")
                    self.save_metadata(extra_name=extra_name)
                    self.stop_streaming()

                    if rec_count < total_rec - 1:
                        LOGGER.info("Buffer period")
                        time.sleep(buffer_s)

        finally:
            self.stop_streaming()
        return saved_diagnostics_paths

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
