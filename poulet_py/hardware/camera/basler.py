from pypylon import pylon
import cv2
import os
import time
import csv
import json
import logging


class BaslerCamera:
    """
    A class to interact with a Basler camera using pypylon and OpenCV.
    """

    def __init__(self):
        """
        Initializes the BaslerCamera object and opens a connection to the first available camera.
        """
        self.basler_camera = None
        self.out = None

        while self.basler_camera is None:
            try:
                # Try to create and open the camera
                self.basler_camera = pylon.InstantCamera(
                    pylon.TlFactory.GetInstance().CreateFirstDevice()
                )
                self.basler_camera.Open()
                print("Camera opened successfully.")
            except pylon.RuntimeException as e:
                # Handle the case where the camera is busy or not available
                print(f"Failed to open camera: {e}")
                input(
                    "Please make the camera available and press Enter to try again..."
                )

    def set_frames_per_second(self, frames_per_second):
        """
        Sets the frame rate for the camera.

        Args:
            frames_per_second (float): The desired frame rate in frames per second.
        """
        self.frames_per_second = frames_per_second
        self.basler_camera.AcquisitionFrameRateEnable.SetValue(True)
        self.basler_camera.AcquisitionFrameRate.SetValue(self.frames_per_second)

    def set_error_log_path(self, path, file_name):
        """
        Sets the path for the error log file.

        Args:
            path (str): The directory where the error log file will be saved.
        """
        self.error_log_file = os.path.join(path, file_name)

    def set_output_file(self, path, extra_name, base_file_name="basler-camera"):
        """
        Sets the output file for recording the video.

        Args:
            path (str): The directory where the output file will be saved.
            extra_name (str): An additional name to be added to the base file name.
            base_file_name (str, optional): The base name of the output file. Defaults to 'basler-camera'.
        """
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")

        frame_width = int(self.basler_camera.Width.Value)
        frame_height = int(self.basler_camera.Height.Value)

        # Construct the full output file name and path
        self.output_file_name = f"{base_file_name}_{extra_name}.mp4"
        self.output_path = os.path.join(path, self.output_file_name)

        # Create the VideoWriter object for recording
        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.frames_per_second,
            (frame_width, frame_height),
        )

        self.timestamps_file = os.path.join(
            path, f"{base_file_name}_{extra_name}_timestamps.csv"
        )

        # Create the CSV file and write the header if it doesn't exist
        if not os.path.isfile(self.timestamps_file):
            with open(self.timestamps_file, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp"])

    def save_timestamp(self, timestamp):
        """
        Save the timestamp to a CSV file.

        Args:
            timestamp: The timestamp to be saved.
        """
        try:
            with open(self.timestamps_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp])
        except Exception as e:
            print(f"Error saving timestamp: {e}")

    def set_timer(self, start_time):
        """
        Sets the timer for the camera.

        Args:
            start_time (float): The time at which the camera recording started.
        """
        self.start_time = start_time

    def start_streaming(self):
        """
        Starts the camera recording.
        """
        self.frame_number = 1
        self.basler_camera.StartGrabbing()

    def stop_streaming(self):
        """
        Stops the camera recording.
        """
        self.basler_camera.StopGrabbing()
        self.basler_camera.Close()

        if self.out is not None:
            self.out.release()

    def capture_frame(self):
        """
        Captures a single frame from the Basler camera, converts it to BGR color format,
        and writes it to the output file.
        """
        try:
            grab_result = self.basler_camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.out.write(img_bgr)

                timestamp = time.time() - self.start_time
                self.save_timestamp(timestamp)

                self.frame_number += 1
            grab_result.Release()
        except Exception as e:
            self.log_error(e)

    def save_metadata(self):
        """
        Saves metadata about the recording to a JSON file in the output directory.
        """
        metadata_file_name = f"{self.output_file_name.split('.')[0]}.json"
        metadata_path = os.path.join(
            os.path.dirname(self.output_path), metadata_file_name
        )

        data = {
            "camera": "basler",
            "width": self.basler_camera.Width.Value,
            "height": self.basler_camera.Height.Value,
            "frame_rate_fps": self.frames_per_second,
            "output_file": self.output_file_name,
            "number_of_frames": self.frame_number,
        }

        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=4)

    def stream_video(self, window_width=None, window_height=None):
        """
        Streams the live video feed from the Basler camera.
        """
        print("Press 'e' to quit the video stream.")

        window_name = "Basler camera"

        while True:
            grab_result = self.basler_camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Resize the image if window size is specified
                if window_width is not None and window_height is not None:
                    img_bgr = cv2.resize(
                        img_bgr, (round(window_width), round(window_height))
                    )

                cv2.imshow(window_name, img_bgr)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("e"):
                    break

            grab_result.Release()

        cv2.destroyAllWindows()

    @staticmethod
    def log_error(self, error_message):
        """
        Logs an error message to the error log file.
        """
        if self.error_log_file is not None:
            logging.error(error_message)
        else:
            print(f"An error occurred: {error_message}")
            print("Set the error log file path to log the error with set_error_log_path().")
        
