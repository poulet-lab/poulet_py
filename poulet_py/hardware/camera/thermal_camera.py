import os
import platform
import time
from datetime import datetime

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import keyboard
except ImportError:
    pass
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import json
import logging
import signal
import sys

import clr
from scipy import ndimage


def py_frame_callback(frame, userptr):
    """
    Callback function to handle frames from the camera.

    Args:
        frame: The frame data from the camera.
        userptr: User pointer.
    """
    array_pointer = cast(
        frame.contents.data,
        POINTER(c_uint16 * (frame.contents.width * frame.contents.height)),
    )
    data = np.frombuffer(array_pointer.contents, dtype=np.uint16).reshape(
        frame.contents.height, frame.contents.width
    )

    # Ensure frame size is correct
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    # Add frame data to queue if not full
    if not q.full():
        q.put(data)


# Check whether we are in Windows
if not platform.system() == "Windows":
    from .uvctypes import *

    BUF_SIZE = 2
    q = Queue(BUF_SIZE)
    PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
    tiff_frame = 1
    colorMapType = 0
else:
    import pythoncom


class ThermalCamera:
    """
    A class to interact with the Lepton 3.5 thermal camera.
    """

    def __init__(self, vminT=30, vmaxT=34):
        """
        Initializes the ThermalCamera object.

        Args:
            vminT (int, optional): Minimum temperature threshold. Defaults to 30.
            vmaxT (int, optional): Maximum temperature threshold. Defaults to 34.
        """
        self.vminT = int(vminT)
        self.vmaxT = int(vmaxT)
        self.frames_per_second = 8.7
        self.width = 160
        self.height = 120
        self.video_format = None
        self.windows = False
        self.shutter_manual = False

        # Check whether we are in Windows
        if platform.system() == "Windows":
            self.windows = True
            self.windows_camera = CameraWindows()

        print("Object thermal camera initialized")
        print(f"vminT = {self.vminT} and vmaxT = {self.vmaxT}")

    def start_streaming(self):
        global devh
        global dev
        """
        Method to start streaming. This method needs to be called always
        before you can extract the data from the camera.
        """
        if self.windows:
            self.windows_camera.initialise_camera()
            time.sleep(1)
            self.windows_camera.start_streaming()
        else:
            ctx = POINTER(uvc_context)()
            dev = POINTER(uvc_device)()
            devh = POINTER(uvc_device_handle)()
            ctrl = uvc_stream_ctrl()
            print(ctrl.__dict__)

            res = libuvc.uvc_init(byref(ctx), 0)
            if res < 0:
                print("uvc_init error")
                exit(1)

            try:
                res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
                print(res)
                if res < 0:
                    print("uvc_find_device error")
                    exit(1)

                try:
                    res = libuvc.uvc_open(dev, byref(devh))
                    print(res)
                    if res < 0:
                        print("uvc_open error")
                        exit(1)

                    print("device opened!")

                    frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
                    if len(frame_formats) == 0:
                        print("device does not support Y16")
                        exit(1)

                    libuvc.uvc_get_stream_ctrl_format_size(
                        devh,
                        byref(ctrl),
                        UVC_FRAME_FORMAT_Y16,
                        frame_formats[0].wWidth,
                        frame_formats[0].wHeight,
                        int(1e7 / frame_formats[0].dwDefaultFrameInterval),
                    )

                    res = libuvc.uvc_start_streaming(
                        devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0
                    )
                    if res < 0:
                        print(f"uvc_start_streaming failed: {res}")
                        exit(1)

                    print("done starting stream, displaying settings")
                    print_shutter_info(devh)
                    print("resetting settings to default")
                    set_auto_ffc(devh)
                    set_gain_high(devh)
                    print("current settings")
                    print_shutter_info(devh)

                except:
                    libuvc.uvc_unref_device(dev)
                    print("Failed to Open Device")
                    exit(1)
            except:
                libuvc.uvc_exit(ctx)
                print("Failed to Find Device")
                exit(1)

    def set_timer(self, start_time):
        """
        Sets the timer for the camera.

        Args:
            start_time (float): The time at which the camera recording started.
        """
        self.start_time = start_time

    def set_error_log_path(self, path, file_name):
        """
        Sets the path for the error log file.

        Args:
            path (str): The path to the error log file.
        """
        self.error_log_file = os.path.join(path, file_name)

    def set_output_file(
        self,
        path,
        extra_name,
        base_file_name="thermal-camera",
        video_format="hdf5",
        png=False,
    ):
        """
        Sets the output file for recording the video.

        Args:
            path (str): The directory where the output file will be saved.
            extra_name (str): An additional name to be added to the base file name.
            base_file_name (str, optional): The base name of the output file. Defaults to 'thermal-camera'.
            video_format (str, optional): The format of the output video file. Defaults to 'hdf5'.
            png (bool, optional): Whether to save frames as PNG images. Defaults to False.
        """
        self.video_format = video_format
        self.output_file_name = f"{base_file_name}_{extra_name}.{video_format}"
        self.output_path = os.path.join(path, self.output_file_name)
        self.png = png

    def set_shutter_manual(self):
        """
        Sets the camera shutter to manual mode.
        """
        global devh

        print("Shutter is now manual.")
        try:
            if self.windows:
                self.windows_camera.set_shutter_manual()
            else:
                set_manual_ffc(devh)
        except:
            print("Failed to set shutter to manual.")
        finally:
            self.shutter_manual = True

    def perform_manual_ffc(self):
        """
        Performs a manual Flat Field Correction (FFC).
        """
        global devh

        print("Manual FFC")
        if self.windows:
            self.windows_camera.perform_manual_ffc()
        else:
            perform_manual_ffc(devh)
            print_shutter_info(devh)

    def stop_streaming(self):
        """
        Stops the camera stream.
        """
        global devh

        # check if there's a file open
        if self.video_format == "hdf5" and self.create_hdf5_file:
            self.hpy_file.close()

        print("Stop streaming")
        if self.windows:
            self.windows_camera.stop_streaming()
        else:
            libuvc.uvc_stop_streaminging(devh)

    def create_hdf5_file(self):
        """
        Creates an HDF5 file to store the thermal image data.
        """
        self.frame_number = 1
        if self.video_format == "hdf5":
            self.hpy_file = h5py.File(self.output_path, "w")
        else:
            assert False, "Invalid video format. Please set the video format to 'hdf5'."

    def capture_frame(self):
        """
        Captures a single frame from the thermal camera, converts it to Celsius,
        and writes it to the output file.
        """

        # Warning if hdf5 file is not created
        if self.video_format != "hdf5":
            assert False, "Invalid video format. Please set the video format to 'hdf5'."

        if self.windows:
            thermal_image_kelvin_data = self.windows_camera.get_frame()
        else:
            thermal_image_kelvin_data = q.get(True, 500)

        if thermal_image_kelvin_data is not None:
            thermal_image_celsius_data = (thermal_image_kelvin_data - 27315) / 100

            self.hpy_file.create_dataset(
                (f"frame{self.frame_number}"), data=thermal_image_celsius_data
            )

            self.frame_number += 1
            print(f"Frame captured! at {self.output_path}")
        else:
            print("Thermal data is none")


    def export_frame_to_png(self, path, file_name, colormap="coolwarm"):
        """Save all frames from the current HDF5 recording as PNG images."""

        with h5py.File(self.output_path, "r") as f:
            frame_keys = list(f.keys())
            frame_keys.sort(key=lambda x: int(x.replace("frame", "")))

            for frame_name in frame_keys:
                frame_data = f[frame_name][()]
                png_filename = os.path.join(
                    path, f"{file_name}_{frame_name}.png"
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(
                    frame_data,
                    cmap=colormap,
                    vmin=self.vminT,
                    vmax=self.vmaxT,
                )
                fig.colorbar(im, ax=ax, label="Temperature (°C)")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(png_filename, bbox_inches="tight")
                plt.close(fig)


    def grab_data_func(self, func, **kwargs):
        """
        Grabs data from the thermal camera and processes it using the provided function.

        Args:
            func (function): A function to process the thermal image data.
            **kwargs: Additional keyword arguments to pass to the processing function.

        Raises:
            AssertionError: If the output path is not set.
            Exception: If an error occurs during data capture and processing.
        """
        end = False

        # Warning if hdf5 file is not created
        if self.video_format != "hdf5":
            assert False, "Invalid video format. Please set the video format to 'hdf5'."

        print("Starting to grab data")
        try:
            while not end:
                if self.windows:
                    thermal_image_kelvin_data = self.windows_camera.get_frame()
                else:
                    thermal_image_kelvin_data = q.get(True, 500)
                if thermal_image_kelvin_data is None:
                    print("Data is none")
                    # make an empty frame
                    thermal_image_celsius_data = np.zeros([120, 160])

                thermal_image_celsius_data = (thermal_image_kelvin_data - 27315) / 100

                end = func(
                    thermal_image_data=thermal_image_celsius_data,
                    hpy_file=self.hpy_file,
                    frame_number=self.frame_number,
                    cam=self,
                    **kwargs,
                )

                self.frame_number += 1

        except Exception as e:
            self.log_error(e)
            self.stop_streaming()

    def plot_live(self, overlay_circle=None):
        """
        Continuously updates a live plot with thermal camera data.
        Keyboard commands:
          - Press "r" to refresh the shutter.
          - Press "t" to take a thermal pic.
          - Press "e" to exit.
        """
        print('Press "r" to refresh the shutter.')
        print('Press "t" to take a thermal pic.')
        print('Press "e" to exit.')
        print("Starting live plot...")

        mpl.rc("image", cmap="coolwarm")
        if self.windows:
            print("Windows detected")
            plt.ion()

        fig = plt.figure()
        ax = plt.axes()
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", "5%", "5%")

        dummy = np.zeros((120, 160))
        img = ax.imshow(
            dummy, interpolation="nearest", vmin=self.vminT, vmax=self.vmaxT, animated=True
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(img, cax=cax)

        circle_artist = None
        mean_text_artist = None

        def valid_circle(circle):
            if not isinstance(circle, dict):
                return False
            if "centre" not in circle or "radius" not in circle:
                return False
            centre = circle["centre"]
            radius = circle["radius"]
            if not (isinstance(centre, (tuple, list)) and len(centre) == 2):
                return False
            if not (isinstance(radius, (int, float)) and radius > 0):
                return False
            return True

        pressed = False
        try:
            while True:
                if self.windows:
                    data = self.windows_camera.get_frame()
                else:
                    data = q.get(True, 500)
                if data is None:
                    print("Data is none")
                    data = np.zeros((120, 160))

                data = (data - 27315) / 100

                img.set_data(data)

                if circle_artist is not None:
                    circle_artist.remove()
                    circle_artist = None
                if mean_text_artist is not None:
                    mean_text_artist.remove()
                    mean_text_artist = None

                if valid_circle(overlay_circle):
                    centre = overlay_circle["centre"]
                    radius = overlay_circle["radius"]
                    circle_artist = Circle(
                        (centre[0], centre[1]),
                        radius,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=2,
                    )
                    ax.add_patch(circle_artist)

                    yy, xx = np.ogrid[: data.shape[0], : data.shape[1]]
                    dist = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2)
                    mask = dist <= radius
                    if np.any(mask):
                        mean_temp = np.mean(data[mask])
                        text_x = centre[0] + radius + 10
                        text_y = centre[1]
                        mean_text_artist = ax.text(
                            text_x,
                            text_y,
                            f"{mean_temp:.2f}",
                            color="black",
                            ha="left",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                        )

                fig.canvas.draw_idle()
                plt.pause(0.2)

                if keyboard.is_pressed("r"):
                    if not pressed:
                        print("Manual FFC")
                        self.perform_manual_ffc()
                        pressed = True
                elif keyboard.is_pressed("t"):
                    if not pressed:
                        now = datetime.now()
                        dt_string = now.strftime("day_%d_%m_%Y_time_%H_%M_%S")
                        try:
                            self.capture_frame()
                        except Exception as e:
                            self.log_error(e)
                            print("There isn't a set path!")
                        pressed = True
                elif keyboard.is_pressed("e"):
                    if not pressed:
                        print("Exiting live plot")
                        break
                else:
                    pressed = False
        except Exception as e:
            self.log_error(e)
            if self.windows:
                plt.ioff()
                plt.close(fig)
            self.stop_streaming()
        finally:
            if self.windows:
                plt.ioff()
                plt.close(fig)

    def save_metadata(self):
        """
        Saves metadata about the recording to a JSON file in the output directory.
        """
        metadata_file_name = f"{self.output_file_name.split('.')[0]}.json"
        metadata_path = os.path.join(os.path.dirname(self.output_path), metadata_file_name)

        data = {
            "camera": "thermal",
            "resolution_width": self.width,
            "resolution_height": self.height,
            "frame_rate_fps": self.frames_per_second,
            "output_file": self.output_file_name,
            "temperature_min": self.vminT,
            "temperature_max": self.vmaxT,
            "video_format": self.video_format,
            "png_frames": self.png,
            "shutter_manual": self.shutter_manual,
        }

        if self.video_format == "hdf5":
            data["number_of_frames"] = self.frame_number

        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=4)

    def log_error(self, error_message):
        """
        Logs an error message to the error log file.
        """
        print("An error occurred:", error_message)
        if self.error_log_file is not None:
            logging.error(error_message)
        else:
            print(f"An error occurred: {error_message}")
            print("Set the error log file path to log the error with set_error_log_path().")


folder = "x64" if platform.architecture()[0] == "64bit" else "x86"
path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
sys.path.append(os.path.sep.join([path, folder]))
clr.AddReference("LeptonUVC")
clr.AddReference("ManagedIR16Filters")

from IR16Filters import IR16Capture, NewBytesFrameEvent
from Lepton import CCI


def handle_exit(sig, frame):
    print("Exiting and cleaning up...")
    pythoncom.CoUninitialize()


# Register signal handlers for clean exit
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


class CameraWindows:
    def __init__(self):
        self.latest_frame = None
        self.CCI = CCI
        self.IR16Capture = IR16Capture
        self.NewBytesFrameEvent = NewBytesFrameEvent
        self.device = None
        self.reader = None

    def add_frame(self, array, width, height):
        """
        Add a new frame to the buffer of read data.
        """
        img = np.fromiter(array, dtype="uint16").reshape(height, width)  # parse
        img = ndimage.rotate(img, angle=0, reshape=True)  # rotation
        self.latest_frame = img.astype(np.float16)  # update the last reading

    def initialise_camera(self):
        """
        Initialize the camera and start capturing frames.
        """
        devices = []

        # initialize COM on this thread
        pythoncom.CoInitialize()
        time.sleep(1)

        for i in self.CCI.GetDevices():
            if i.Name.startswith("PureThermal"):
                devices.append(i)

            if len(devices) > 1:
                print("Multiple Pure Thermal devices have been found.\n")
                for i, d in enumerate(devices):
                    print(f"{i}. {d}")
                while True:
                    idx = input("Select the index of the required device: ")
                    try:
                        idx = int(idx)
                        if idx in range(len(devices)):
                            self.device = devices[idx]
                            break
                    except ValueError:
                        print("Unrecognized input value.\n")

            elif len(devices) == 1:
                self.device = devices[0]
            else:
                self.device = None

            txt = "No devices called 'PureThermal' have been found."
            assert self.device is not None, txt
            self.device = self.device.Open()
            self.device.sys.RunFFCNormalization()

            self.device.sys.SetGainMode(self.CCI.Sys.GainMode.HIGH)

            self.reader = self.IR16Capture()
            callback = self.NewBytesFrameEvent(self.add_frame)
            self.reader.SetupGraphWithBytesCallback(callback)

    def start_streaming(self):
        """
        Start capturing frames.
        """
        self.reader.RunGraph()

    def set_shutter_manual(self):
        """
        Set the shutter mode to manual.
        """
        new_shutter_mode_obj = self.device.sys.GetFfcShutterModeObj()
        new_shutter_mode_obj.shutterMode = self.CCI.Sys.FfcShutterMode.AUTO

        self.device.sys.SetFfcShutterModeObj(new_shutter_mode_obj)

    def perform_manualff(self):
        """
        Perform a manual flat field correction.
        """
        self.device.sys.RunFFCNormalization()

    def stop_streaming(self):
        """
        Stop capturing frames.
        """
        self.reader.StopGraph()
        pythoncom.CoUninitialize()
        handle_exit(None, None)

    def get_frame(self):
        """
        Retrieve the latest frame captured by the camera.
        """
        return self.latest_frame
