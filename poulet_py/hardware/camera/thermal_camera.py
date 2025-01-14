import numpy as np
import h5py
import platform
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import time
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
from scipy import ndimage
import clr
import sys
import os
import platform
import numpy as np
import pythoncom
import signal
import logging


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
    from uvctypes import *

    BUF_SIZE = 2
    q = Queue(BUF_SIZE)
    PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(
        py_frame_callback
    )
    tiff_frame = 1
    colorMapType = 0


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
                        print("uvc_start_streaming failed: {0}".format(res))
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

            # get current time
            timestamp = time.time() - self.start_time
            self.hpy_file.create_dataset((f"time{self.frame_number}"), data=[timestamp])

            self.frame_number += 1
        else:
            print("Thermal data is none")

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

    def plot_live(self):
        """
        Method to plot the thermal camera as a 2-D raster (imshow, heatmap).
        The min and max values of the heatmap are specified.
        You can take a pic too.
        """
        print('Press "r" to refresh the shutter.')
        print('Press "t" to take a thermal pic.')
        print('Press "e" to exit.')

        mpl.rc("image", cmap="coolwarm")

        pressed = False

        if platform.system() == "Windows":
            plt.ion()  # Enable interactive mode

        fig = plt.figure()
        ax = plt.axes()
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", "5%", "5%")

        dummy = np.zeros([120, 160])

        img = ax.imshow(
            dummy,
            interpolation="nearest",
            vmin=self.vminT,
            vmax=self.vmaxT,
            animated=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        fig.colorbar(img, cax=cax)

        try:
            while True:
                if platform.system() == "Windows":
                    data = self.windows_camera.get_frame()
                else:
                    data = q.get(True, 500)
                if data is None:
                    print("Data is none")
                    # make an empty frame
                    data = np.zeros([120, 160])

                data = (data - 27315) / 100

                if platform.system() == "Windows":
                    img.set_data(data)  # Update image data
                    fig.canvas.draw()  # Redraw the figure
                    fig.canvas.flush_events()  # Flush the GUI events for real-time updates
                else:
                    ax.clear()
                    img = ax.imshow(data, vmin=self.vminT, vmax=self.vmaxT)
                    fig.colorbar(img, cax=cax)

                plt.pause(0.0005)

                if keyboard.is_pressed("r"):
                    if not pressed:
                        print("Manual FFC")
                        self.performManualff()
                        pressed = True

                elif keyboard.is_pressed("t"):
                    if not pressed:
                        try:
                            now = datetime.now()
                            dt_string = now.strftime("day_%d_%m_%Y_time_%H_%M_%S")
                            print(dt_string)
                            f = h5py.File(f"{self.pathset}/{dt_string}.hdf5", "w")
                            f.create_dataset("image", data=data)
                            f = None
                            print("Thermal pic saved as hdf5")
                            if self.png:
                                plt.imsave(
                                    f"{self.pathset}/{dt_string}.png",
                                    data,
                                    vmin=self.vminT,
                                    vmax=self.vmaxT,
                                )

                        except Exception as e:
                            self.log_error(e)
                            print("There isn't a set path!")

                    pressed = True

                elif keyboard.is_pressed("e"):
                    if not pressed:
                        print("We are done")
                        break

                else:
                    pressed = False

        except Exception as e:
            self.log_error(e)
            if platform.system() == "Windows":
                plt.ioff()
                plt.close(fig)

            self.stop_streaming()

        finally:
            if platform.system() == "Windows":
                plt.ioff()
                plt.close(fig)

    def save_metadata(self):
        """
        Saves metadata about the recording to a JSON file in the output directory.
        """
        metadata_file_name = f"{self.output_file_name.split('.')[0]}.json"
        metadata_path = os.path.join(
            os.path.dirname(self.output_path), metadata_file_name
        )

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
        

# imports


# Initialize COM
pythoncom.CoInitialize()

folder = "x64" if platform.architecture()[0] == "64bit" else "x86"
path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
sys.path.append(os.path.sep.join([path, folder]))
clr.AddReference("LeptonUVC")
clr.AddReference("ManagedIR16Filters")

from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent


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
        for i in self.CCI.GetDevices():
            if i.Name.startswith("PureThermal"):
                devices.append(i)

        if len(devices) > 1:
            print("Multiple Pure Thermal devices have been found.\n")
            for i, d in enumerate(devices):
                print("{}. {}".format(i, d))
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
