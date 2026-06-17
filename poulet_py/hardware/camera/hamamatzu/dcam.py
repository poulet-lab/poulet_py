from ctypes import byref
from threading import Condition, Event, Thread
from typing import Any

from numpy import ndarray
from pydantic import BaseModel, Field, PrivateAttr

from ._api import (
    DCAM_IDPROP,
    DCAM_PIXELTYPE,
    DCAMAPI_INIT,
    DCAMDEV_OPEN,
    DCAMERR,
    DCAMPROP,
    dcamapi_init,
    dcamapi_uninit,
    dcamdev_close,
    dcamdev_open,
    dcamwait_close,
)


class DCAM(BaseModel):
    device_index: int = Field(default=0, description="")
    pixel_type: DCAM_PIXELTYPE = Field(
        default=DCAM_PIXELTYPE.MONO16, description="The pixel type of the camera."
    )
    trigger_source: DCAMPROP.TRIGGERSOURCE = Field(
        default=DCAMPROP.TRIGGERSOURCE.INTERNAL, description="The trigger source of the camera."
    )

    _device_count: int = PrivateAttr(default=0)
    _hdcam: Any = PrivateAttr()
    _is_open: bool = PrivateAttr(default=False)

    _buffer_idx: int = PrivateAttr(0)
    _buffer: ndarray = PrivateAttr()
    _acquisition_thread: Thread = PrivateAttr()
    _stop_acquisition_event: Event = PrivateAttr(default_factory=Event)
    _sampling_cond: Condition = PrivateAttr(default_factory=Condition)
    _stimulus_running: bool = PrivateAttr(default=False)

    @property
    def is_open(self):
        return self._is_open

    @property
    def device_count(self):
        self._ensure_open()

        return self._device_count

    def open(self):
        if self._is_open:
            return False

        paraminit = DCAMAPI_INIT()
        err = dcamapi_init(byref(paraminit))
        if err != DCAMERR.SUCCESS:
            raise RuntimeError(f"Failed to initialize DCAM-API: {DCAMERR(err).name}")
        self._device_count = paraminit.iDeviceCount

        paramopen = DCAMDEV_OPEN()
        paramopen.index = self.device_index
        err = dcamdev_open(byref(paramopen))
        if err != DCAMERR.SUCCESS:
            raise RuntimeError(f"Failed to initialize DCAM device: {DCAMERR(err).name}")
        self._hdcam = paramopen.hdcam

        self._is_open = True

        return True

    def close(self):
        if not self._is_open:
            return

        # self.__close_hdcamwait() TODO
        self._is_open = False

        dcamdev_close(self.__hdcam)
        self.__hdcam = 0

        self._device_count = 0
        dcamapi_uninit()

        return True

    def _ensure_open(self):
        """
        Check if serial connection is open.

        Raises
        ------
        RuntimeError
            If serial connection is not open.
        """
        if not self._is_open:
            raise RuntimeError("DCAM is not open")

    #### TODO ####
    def __close_hdcamwait(self):

        if self.__hdcamwait == 0:
            return True

        ret = self.__result(dcamwait_close(self.__hdcamwait))
        if ret is False:
            return False

        self.__hdcamwait = 0
        return True

    def wait_event(self, eventmask: DCAMWAIT_CAPEVENT, timeout_millisec):
        """Wait specified event.

        Wait specified event.

        Arg:
            eventmask (DCAMWAIT_CAPEVENT): Event mask to wait.
            timeout_millisec (int): Timeout by milliseconds.

        Returns:
            DCAMWAIT_CAPEVENT: Happened event.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        ret = self.__open_hdcamwait()
        if ret is False:
            return False

        paramwaitstart = DCAMWAIT_START()
        paramwaitstart.eventmask = eventmask
        paramwaitstart.timeout = timeout_millisec
        ret = self.__result(dcamwait_start(self.__hdcamwait, byref(paramwaitstart)))
        if ret is False:
            return False

        return paramwaitstart.eventhappened

    def dev_getstring(self, idstr: DCAM_IDSTR):
        """Get string of device.

        Get string of device.

        Args:
            idstr (DCAM_IDSTR): String id.

        Returns:
            string: Device information specified by DCAM_IDSTR.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if self.is_opened():
            hdcam = self.__hdcam
        else:
            hdcam = self.__iDevice

        paramdevstr = DCAMDEV_STRING()
        paramdevstr.iString = idstr
        paramdevstr.alloctext(256)

        ret = self.__result(dcamdev_getstring(hdcam, byref(paramdevstr)))
        if ret is False:
            return False

        return paramdevstr.text.decode()

    def dev_getcapability(self):
        """Get capability of function

        Get capability of function

        Returns:
            DCAMDEV_CAPABILITY: Capability of function
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        capability = DCAMDEV_CAPABILITY()
        ret = self.__result(dcamdev_getcapability(self.__hdcam, byref(capability)))
        if ret is False:
            return False

        return capability

    # dcamprop functions

    def prop_getattr(self, idprop: DCAM_IDPROP):
        """Get property attribute.

        Get property attribute.

        args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            DCAMPROP_ATTR: Attribute information of the property.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        propattr = DCAMPROP_ATTR()
        propattr.iProp = idprop
        ret = self.__result(dcamprop_getattr(self.__hdcam, byref(propattr)))
        if ret is False:
            return False

        return propattr

    def prop_getvalue(self, idprop: DCAM_IDPROP):
        """Get property value.

        Get property value.

        args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            float: Property value of property id.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cDouble = c_double()
        ret = self.__result(dcamprop_getvalue(self.__hdcam, idprop, byref(cDouble)))
        if ret is False:
            return False

        return cDouble.value

    def prop_setvalue(self, idprop: DCAM_IDPROP, fValue):
        """Set property value.

        Set property value.

        args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Setting value.

        Returns:
            bool: True if set property value was succeeded. False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        ret = self.__result(dcamprop_setvalue(self.__hdcam, idprop, fValue))
        if ret is False:
            return False

        return True

    def prop_setgetvalue(self, idprop: DCAM_IDPROP, fValue, option=0):
        """Set and get property value.

        Set and get property value.

        args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Input value for setting and receive actual set value by ref.

        Returns:
            float: Accurate value set in device.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cDouble = c_double(fValue)
        cOption = c_int32(option)
        ret = self.__result(dcamprop_setgetvalue(self.__hdcam, idprop, byref(cDouble), cOption))
        if ret is False:
            return False

        return cDouble.value

    def prop_queryvalue(self, idprop: DCAM_IDPROP, fValue, option=0):
        """Query property value.

        Query property value.

        Args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Value of property.

        Returns:
            float: Property value specified by option.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cDouble = c_double(fValue)
        cOption = c_int32(option)
        ret = self.__result(dcamprop_queryvalue(self.__hdcam, idprop, byref(cDouble), cOption))
        if ret is False:
            return False

        return cDouble.value

    def prop_getnextid(self, idprop: DCAM_IDPROP):
        """Get next property id.

        Get next property id.

        Args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            DCAM_IDPROP: Next property id.
            bool: False if no more property or error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cIdprop = c_int32(idprop)
        cOption = c_int32(0)  # search next ID

        ret = self.__result(dcamprop_getnextid(self.__hdcam, byref(cIdprop), cOption))
        if ret is False:
            return False

        return cIdprop.value

    def prop_getname(self, idprop: DCAM_IDPROP):
        """Get name of property.

        Get name of property.

        Args:
            idprop (DCAM_IDPROP): Property id.

        Returns:
            string: Caracter string of property id.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        textbuf = create_string_buffer(256)
        ret = self.__result(dcamprop_getname(self.__hdcam, idprop, textbuf, sizeof(textbuf)))
        if ret is False:
            return False

        return textbuf.value.decode()

    def prop_getvaluetext(self, idprop: DCAM_IDPROP, fValue):
        """Get text of property value.

        Get text of property value.

        Args:
            idprop (DCAM_IDPROP): Property id.
            fValue (float): Setting value.

        Returns:
            string: Caracter string of property value.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        paramvaluetext = DCAMPROP_VALUETEXT()
        paramvaluetext.iProp = idprop
        paramvaluetext.value = fValue
        paramvaluetext.alloctext(256)

        ret = self.__result(dcamprop_getvaluetext(self.__hdcam, byref(paramvaluetext)))
        if ret is False:
            return False

        return paramvaluetext.text.decode()

    # dcambuf functions

    def buf_alloc(self, nFrame):
        """Alloc DCAM internal buffer.

        Alloc DCAM internal buffer.

        Arg:
            nFrame (int): Number of frames.

        Returns:
            bool: True if buffer is prepared. False if buffer is not prepared. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cFrame = c_int32(nFrame)
        ret = self.__result(dcambuf_alloc(self.__hdcam, cFrame))
        if ret is False:
            return False

        return self.__result(dcammisc_setupframe(self.__hdcam, self.__bufframe))

    def buf_release(self):
        """Release DCAM internal buffer.

        Release DCAM internal buffer.

        Returns:
            bool: True if release DCAM internal buffser was succeeded. False if error happens during releasing buffer. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cOption = c_int32(0)
        return self.__result(dcambuf_release(self.__hdcam, cOption))

    def buf_getframe(self, iFrame):
        """Return DCAMBUF_FRAME instance.

        Return DCAMBUF_FRAME instance with image data specified by iFrame.

        Arg:
            iFrame (int): Index of target frame.

        Returns:
            (aFrame, npBuf): aFrame is DCAMBUF_FRAME, npBuf is NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        framebundlenum = 1

        fValue = c_double()
        err = dcamprop_getvalue(self.__hdcam, DCAM_IDPROP.FRAMEBUNDLE_MODE, byref(fValue))
        if not err.is_failed() and int(fValue.value) == DCAMPROP.MODE.ON:
            err = dcamprop_getvalue(self.__hdcam, DCAM_IDPROP.FRAMEBUNDLE_NUMBER, byref(fValue))
            if not err.is_failed():
                framebundlenum = int(fValue.value)
            else:
                return False

        viewnum = 1

        err = dcamprop_getvalue(self.__hdcam, DCAM_IDPROP.NUMBEROF_VIEW, byref(fValue))
        if not err.is_failed():
            viewnum = int(fValue.value)

        npBuf = dcammisc_alloc_ndarray(self.__bufframe, framebundlenum, viewnum)

        if npBuf is False:
            return self.__result(DCAMERR.INVALIDPIXELTYPE)

        aFrame = DCAMBUF_FRAME()
        aFrame.iFrame = iFrame

        aFrame.buf = npBuf.ctypes.data_as(c_void_p)
        aFrame.rowbytes = self.__bufframe.rowbytes
        aFrame.type = self.__bufframe.type
        aFrame.width = self.__bufframe.width
        aFrame.height = self.__bufframe.height

        ret = self.__result(dcambuf_copyframe(self.__hdcam, byref(aFrame)))
        if ret is False:
            return False

        return (aFrame, npBuf)

    def buf_getframedata(self, iFrame):
        """Return NumPy buffer.

        Return NumPy buffer of image data specified by iFrame.

        Arg:
            iFrame (int): Index of target frame.

        Returns:
            npBuf: NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        """
        ret = self.buf_getframe(iFrame)
        if ret is False:
            return False

        return ret[1]

    def buf_getlastframedata(self):
        """Return NumPy buffer of last updated.

        Return NumPy buffer of image data of last updated frame.

        Returns:
            npBuf: NumPy buffer.
            bool: False if error happens. lasterr() returns the DCAMERR value.
        """
        return self.buf_getframedata(-1)

    # dcamcap functions

    def cap_start(self, bSequence=True):
        """Start capturing.

        Start capturing.

        Arg:
            bSequence (bool): False means SNAPSHOT, others means SEQUENCE.

        Returns:
            bool: True if start capture. False if error happened.  lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        if bSequence:
            mode = DCAMCAP_START.SEQUENCE
        else:
            mode = DCAMCAP_START.SNAP

        return self.__result(dcamcap_start(self.__hdcam, mode))

    def cap_snapshot(self):
        """Capture snapshot.

        Capture snapshot. Get the frames specified in buf_alloc().

        Returns:
            bool: True if start snapshot. False if error happened. lasterr() returns the DCAMERR value.
        """
        return self.cap_start(False)

    def cap_stop(self):
        """Stop capturing.

        Stop capturing.

        Returns:
            bool: True if Stop capture. False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        return self.__result(dcamcap_stop(self.__hdcam))

    def cap_status(self):
        """Get capture status.

        Get capture status.

        Returns:
            DCAMCAP_STATUS: Current capturing status.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cStatus = c_int32()
        ret = self.__result(dcamcap_status(self.__hdcam, byref(cStatus)))
        if ret is False:
            return False

        return cStatus.value

    def cap_transferinfo(self):
        """Get transfer info.

        Get transfer info.

        Returns:
            DCAMCAP_TRANSFERINFO: Current image transfer status.
            bool: False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        paramtransferinfo = DCAMCAP_TRANSFERINFO()
        ret = self.__result(dcamcap_transferinfo(self.__hdcam, byref(paramtransferinfo)))
        if ret is False:
            return False

        return paramtransferinfo

    def cap_firetrigger(self):
        """Fire software trigger.

        Fire software trigger.

        Returns:
            bool: True if firing trigger was succeeded. False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.is_opened():
            return self.__result(DCAMERR.INVALIDHANDLE)  # instance is not opened yet.

        cOption = c_int32(0)
        ret = self.__result(dcamcap_firetrigger(self.__hdcam, cOption))
        if ret is False:
            return False

        return True

    def __open_hdcamwait(self):
        """Get DCAMWAIT handle.

        Get DCAMWAIT handle.

        Returns:
            bool: True if get DCAMWAIT handle was succeeded. False if error happened. lasterr() returns the DCAMERR value.
        """
        if not self.__hdcamwait == 0:
            return True

        paramwaitopen = DCAMWAIT_OPEN()
        paramwaitopen.hdcam = self.__hdcam
        ret = self.__result(dcamwait_open(byref(paramwaitopen)))
        if ret is False:
            return False

        if paramwaitopen.hwait == 0:
            return self.__result(DCAMERR.INVALIDWAITHANDLE)

        self.__hdcamwait = paramwaitopen.hwait
        return True

    def wait_capevent_frameready(self, timeout_millisec):
        """Wait DCAMWAIT_CAPEVENT.FRAMEREADY event.

        Wait DCAMWAIT_CAPEVENT.FRAMEREADY event.

        Arg:
            timeout_millisec (int): Timeout by milliseconds.

        Returns:
            bool: True if wait capture. False if error happened. lasterr() returns the DCAMERR value.
        """
        ret = self.wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY, timeout_millisec)
        if ret is False:
            return False

        # ret is DCAMWAIT_CAPEVENT.FRAMEREADY

        return True

    def allocbuffer(self, number_of_frames):
        """Allocate buffer.

        Allocate buffer with Dcam.buf_alloc().
        If success, set value is kept by self.__number_of_frames.

        Args:
            number_of_frames (int): Value to set for Dcam.buf_alloc()

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.buf_alloc(number_of_frames):
            print(
                f"-NG: Dcam.buf_alloc({number_of_frames}) failed with error {self.dcam.lasterr().name}"
            )
            return False

        self.__number_of_frames = number_of_frames
        return True

    def releasebuffer(self):
        """Release allocated buffer.

        Release allcated buffer with Dcam.buf_release().

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.buf_release():
            print(f"-NG: Dcam.buf_release() failed with error {self.dcam.lasterr().name}")
            return False

        return True

    def startcapture(self, is_sequence=True):
        """Start capturing.

        Start capturing with Dcam.cap_start().
        If failure, if shows error message.

        Args:
            is_sequence (bool): if True, sequential capturing, otherwise snap capturing

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.cap_start(is_sequence):
            print(f"-NG: Dcam.cap_start() failed with error {self.dcam.lasterr().name}")
            return False

        return True

    def stopcapture(self):
        """Stop capturing.

        Stop capturing with Dcam.cap_stop().
        If failure, it shows error message

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.cap_stop():
            print(f"-NG: Dcam.cap_stop() failed with error {self.dcam.lasterr().name}")
            return False

        return True

    def is_capstaus_ready(self):
        """Check whether DCAMCAP_STATUS is READY or not.

        Call Dcam.cap_status() and check whetherthe value is READY or not

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        capstatus = self.dcam.cap_status()
        if capstatus is False:
            print(f"-NG: Dcam.cap_status() failed with error {self.dcam.lasterr().name}")
            return False

        return capstatus == DCAMCAP_STATUS.READY

    def firetrigger(self):
        """Fire trigger.

        Fire software trigger when TRIGGERSOURCE is SOFTWARE.

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.cap_firetrigger():
            print(f"-NG: Dcam.cap_firetrigger() failed with error {self.dcam.lasterr().name}")
            return False

        return True

    def wait_capevent_frameready(self, timeout_millisec):
        """Wait capture event frameready.

        Wait for frameready event for amount of time specified by timeout_millisec.
        If frameready event happened, it returns True. Otherwise it returns DCAMERR.

        Args:
            timeout_millisec (int): timeout time for waiting frameready event in ms

        Returns:
            bool: True if success
            DCAMERR: Dcam.lasterr() if failure
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.wait_capevent_frameready(timeout_millisec):
            return self.dcam.lasterr()

        return True

    def get_lastframedata(self):
        """Get last frame data.

        Access last frame data with Dcam.buf_getlastframedata().
        If success, it returns Numpy ndarray stored last image data.
        If failure, it returns False

        Returns:
            NumPy ndarray: NumPy ndarray stored image if success
            bool: False if failure
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        ret = self.dcam.buf_getlastframedata()
        if ret is False:
            print(f"-NG: Dcam.buf_getlastframedata() failed with error {self.dcam.lasterr().name}")
            return False

        return ret

    def get_transferinfo(self):
        """Get transfer status.

        Get the total number of images captured and the frame index of the last captured.

        Returns:
            (int, int): index of the last captured frame, number of captured frames
            bool: False if failure
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        captransferinfo = self.dcam.cap_transferinfo()
        if captransferinfo is False:
            print(f"-NG: Dcam.cap_transferinfo() failed with error {self.dcam.lasterr().name}")
            return False

        if captransferinfo.nFrameCount < 1:
            print("-NG: There are no images retrieved.")
            return False

        return (captransferinfo.nNewestFrameIndex, captransferinfo.nFrameCount)

    def save_rawimages(self, prefix):
        """Save acquired images as raw.

        Save acquired and retained images as raw data.
        The output file name is "{prefix} - {frameindex}.raw"
        "frameindex" starts at 1 and is numbered from the oldest image.

        Args:
            prefix (string): prefix of output filename

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        captransferinfo = self.dcam.cap_transferinfo()
        if captransferinfo is False:
            print(f"-NG: Dcam.cap_transferinfo() failed with error {self.dcam.lasterr().name}")
            return False

        if captransferinfo.nFrameCount < 1:
            print("-NG: There are no images retrieved.")
            return False

        if captransferinfo.nFrameCount > self.__number_of_frames:
            number_of_images = self.__number_of_frames
            start_frameindex = (captransferinfo.nNewestFrameIndex + 1) % self.__number_of_frames
        else:
            number_of_images = captransferinfo.nFrameCount
            start_frameindex = 0

        for i in range(0, number_of_images, 1):
            index = (start_frameindex + i) % self.__number_of_frames
            datai = self.dcam.buf_getframedata(index)
            filename = f"{prefix} - {i + 1}.raw"
            datai.tofile(filename)

        return True

    def get_propertyvalue(self, propid: IntEnum, showerrmsg=True):
        """Get property value.

        Get property value with Dcam.prop_getvalue()
        if showerrmsg is True, it shows error message
        when Dcam.prop_getvalue() return False.
        showerrmsg defaults to True.
        Set showerrmsg to False when it is meaningful that it is an error.

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum
            showerrmsg (bool): if True, print error message.

        Returns:
            double: get value if success
            bool: False if failure
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        propvalue = self.dcam.prop_getvalue(propid.value)
        if propvalue is False:
            if showerrmsg:
                print(
                    f"-NG: Dcam.prop_getvalue({propid.name}) failed with error {self.dcam.lasterr().name}"
                )
            return False

        return propvalue

    def set_propertyvalue(self, propid: IntEnum, val):
        """Set property value.

        Set property value with Dcam.prop_setvalue()
        it shows error message when Dcam.prop_setvalue() return False

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum. property ID.
            val (double): set value

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        if not self.dcam.prop_setvalue(propid, val):
            print(
                f"-NG: Dcam.prop_setvalue({propid.name}, {val}) failed with error {self.dcam.lasterr().name}"
            )
            return False

        return True

    def setget_propertyvalue(self, propid: IntEnum, val):
        """Set and get property value.

        Set and get property value with Dcam.prop_setgetvalue().
        If success, it returns get value. If failure, it returns False

        Args:
            propid (IntEnum): DCAM_IDPROP IntEnum. property ID
            val (double): set value

        Returns:
            double: get value if success.
            bool: False if failure.
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        res = self.dcam.prop_setgetvalue(propid, val)
        if res is False:
            print(
                f"-NG: Dcam.prop_setgetvalue({propid.name}, {val}) failed with error {self.dcam.lasterr().name}"
            )
            return False

        return res

    def prompt_propvalue(self, propid, restrictmode=PromptRestrictMode.No, restrictval=None):
        """Set property value at the prompt.

        Set property specified by propid at the prompt
        If success, it returns set value.
        If failure, it returns False

        Args:
            propid (IntEnum): DCAM_PROP IntEnum. property ID

        Returns:
            None: if property is not available.
            bool: result when property is available
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        val = None
        propattr = self.dcam.prop_getattr(propid)
        if propattr is False:
            # error happened
            return None

        min = propattr.valuemin
        max = propattr.valuemax
        step = propattr.valuestep
        default = propattr.valuedefault

        if (
            propattr.attribute & DCAM_PROP.ATTR.EFFECTIVE
            and propattr.attribute & DCAM_PROP.ATTR.WRITABLE
            and min != max
        ):
            proptype = propattr.attribute & DCAM_PROP.TYPE.MASK
            if proptype == DCAM_PROP.TYPE.MODE:  # change value to text

                def gettextvaluelist(propid):
                    """Return supported values.

                    Returns an array of the supported values for text.

                    Args:
                        propid (int): Property ID

                    Returns: value list
                    """
                    currentvalue = min
                    valuelist = []
                    if restrictmode != PromptRestrictMode.ModeFilter or currentvalue in restrictval:
                        valuelist.append(int(currentvalue))
                    while currentvalue != max:
                        currentvalue = self.dcam.prop_queryvalue(
                            propid, currentvalue, DCAMPROP_OPTION.NEXT
                        )
                        if (
                            restrictmode != PromptRestrictMode.ModeFilter
                            or currentvalue in restrictval
                        ):
                            valuelist.append(int(currentvalue))
                    return valuelist

                valuelist = gettextvaluelist(propid)
                value_is_good = False
                while not value_is_good:
                    print()
                    prompt = (
                        "\nEnter a [value] for "
                        + str(self.dcam.prop_getname(propid))
                        + " between:\n"
                    )
                    for textval in valuelist:
                        valuetext = self.dcam.prop_getvaluetext(propid, textval)
                        prompt += f"[{int(textval)}]" + valuetext + "\n"

                    valuetext = self.dcam.prop_getvaluetext(propid, default)
                    prompt += "\n[default] " + valuetext
                    prompt += "\n\n>"
                    try:
                        instr = input(prompt)
                        val = int(instr)
                    except ValueError:
                        val = int(default)
                        break
                    value_is_good = val in valuelist
            elif proptype == DCAM_PROP.TYPE.LONG:
                if restrictmode == PromptRestrictMode.ClipMinimum and min < restrictval:
                    min = restrictval
                while True:
                    print()
                    prompt = "\nEnter a value for " + str(self.dcam.prop_getname(propid))
                    prompt += " between " + str(int(min))
                    prompt += " and " + str(int(max))
                    prompt += " in steps of " + str(int(step))
                    prompt += " [default is " + str(int(default)) + "]"
                    prompt += "\n\n> "
                    try:
                        instr = input(prompt)
                        val = int(instr)
                    except ValueError:
                        val = int(default)
                        break

                    if val % int(step) == 0 and val >= int(min) and val <= int(max):
                        break
            elif proptype == DCAM_PROP.TYPE.REAL:
                if restrictmode == PromptRestrictMode.ClipMinimum and min < restrictval:
                    min = restrictval
                while True:
                    print()

                    def get_units(unitid):
                        unitlist = {
                            DCAMPROP_UNIT.SECOND: "s",
                            DCAMPROP_UNIT.CELSIUS: "°C",
                            DCAMPROP_UNIT.KELVIN: "K",
                            DCAMPROP_UNIT.METERPERSECOND: "m/s",
                            DCAMPROP_UNIT.PERSECOND: "/s",
                            DCAMPROP_UNIT.DEGREE: "°",
                            DCAMPROP_UNIT.MICROMETER: "µm",
                        }
                        unitstr = unitlist.get(unitid, "")
                        return unitstr

                    unitname = get_units(propattr.iUnit)
                    minstr = f"{min:.6f}".rstrip("0")
                    if minstr[-1] == ".":
                        minstr += "0"
                    maxstr = f"{max:.6f}".rstrip("0")
                    if maxstr[-1] == ".":
                        maxstr += "0"
                    stepstr = f"{step:.6f}".rstrip("0")
                    if stepstr[-1] == ".":
                        stepstr += "0"
                    defstr = f"{default:.6f}".rstrip("0")
                    if defstr[-1] == ".":
                        defstr += "0"
                    prompt = "\nEnter a value for " + str(self.dcam.prop_getname(propid))
                    prompt += " between " + minstr + unitname
                    prompt += " and " + maxstr + unitname
                    prompt += " in steps of " + stepstr + unitname
                    prompt += " [default is " + defstr + unitname + "]"
                    prompt += "\n\n> "
                    try:
                        instr = input(prompt)
                        val = float(instr)
                    except ValueError:
                        val = default
                        break
                    if val >= min and val <= max:
                        # ignore step for REAL check due to float precision possible problems.
                        # nomally the property has AUTOROUNDING
                        break

        if val is not None:
            val = self.set_propertyvalue(propid, val)

        return val

    def _prompt_longpropvalue_stack(self, propid, clipmax=False):
        """Get to set value of long property.

        Prompt to input setting value, but not set to DCAM.
        Input value is returned if success

        Args:
            propid (DCAM_IDPROP): property id
            clipmax (bool or int): clip maximum of range if clipmax is not False

        Returns:
            None: attribute does not have EFFECTIVE or(and) WRITABLE
            int: input value
            bool: False if failure.
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        val = None
        propattr = self.dcam.prop_getattr(propid)
        if propattr is False:
            # error happened
            return False

        attribute = propattr.attribute
        proptype = attribute & DCAM_PROP.TYPE.MASK
        if proptype != DCAM_PROP.TYPE.LONG:
            # not support
            return False

        min = propattr.valuemin
        max = propattr.valuemax
        if clipmax is not False:
            max = clipmax
        step = propattr.valuestep
        default = propattr.valuedefault

        if (
            attribute & DCAM_PROP.ATTR.EFFECTIVE
            and attribute & DCAM_PROP.ATTR.WRITABLE
            and min != max
        ):
            while True:
                print()
                prompt = "\nEnter a value for " + str(self.dcam.prop_getname(propid))
                prompt += " between " + str(int(min))
                prompt += " and " + str(int(max))
                prompt += " in steps of " + str(int(step))
                prompt += " [default is " + str(int(default)) + "]"
                prompt += "\n\n> "
                try:
                    instr = input(prompt)
                    val = int(instr)
                except ValueError:
                    val = int(default)
                    break

                if val % int(step) == 0 and val >= int(min) and val <= int(max):
                    break

        return val

    def prompt_propvalue_subarray(self):
        """Set property value related subarray at the prompt.

        Set following properties at the prompt.
        DCAM_IDPROP.SUBARRAYMODE,
        DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE,
        DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE,
        Control offset and size combinations

        Returns:
            bool: result
        """
        if self.dcam is None:
            print("-NG: Dcamcon is not opened")
            return False

        res = self.prompt_propvalue(DCAM_IDPROP.SUBARRAYMODE)
        if res is None:
            # not available
            return True
        elif res is False:
            # error happened
            return False

        res = True
        subarraymode = self.get_propertyvalue(DCAM_IDPROP.SUBARRAYMODE)
        if subarraymode is False:
            # error happened
            return False
        elif subarraymode == DCAMPROP.MODE.OFF:
            # not need setting subarray parameters
            res = True
        else:
            # subarraymode == DCAMPROP.MODE.ON
            def prompt_offset_and_size(offsetid, sizeid):
                """Set subarray offset and size.

                Set subarray offset and size.

                Args:
                    offsetid (DCAM_IDPROP): SUBARRAYHPOS or SUBARRAYVPOS
                    sizeid (DCAM_IDPROP): SUBARRAYHSIZE or SUBARRAYVSIZE

                Returns:
                    None: if property is not available.
                """
                propattr_offset = self.dcam.prop_getattr(offsetid)
                if propattr_offset is False:
                    # error happen
                    return False

                propattr_size = self.dcam.prop_getattr(sizeid)
                if propattr_size is False:
                    # error happen
                    return False

                is_offset_available = (
                    propattr_offset.attribute & DCAM_PROP.ATTR.EFFECTIVE
                    and propattr_offset.attribute & DCAM_PROP.ATTR.WRITABLE
                    and propattr_offset.valuemin != propattr_offset.valuemax
                )

                is_size_available = (
                    propattr_size.attribute & DCAM_PROP.ATTR.EFFECTIVE
                    and propattr_size.attribute & DCAM_PROP.ATTR.WRITABLE
                    and propattr_size.valuemin != propattr_size.valuemax
                )

                if is_offset_available and is_size_available:
                    # set both offset and size
                    offsetval = self._prompt_longpropvalue_stack(offsetid)
                    if offsetval is False:
                        # error happened
                        res = False
                    elif offsetval is None:
                        # not need to set offset. prompt to set size
                        res = self.prompt_propvalue(sizeid)
                    else:
                        # offsetval is int value. temporarily suspend setting
                        # clip the maximum of size by subtracting offsetval
                        clipmax = propattr_size.valuemax - offsetval
                        sizeval = self._prompt_longpropvalue_stack(sizeid, clipmax)
                        if sizeval is False:
                            res = False
                        elif sizeval is None:
                            # not need to set size. set offset value
                            res = self.set_propertyvalue(offsetid, offsetval)
                        else:
                            # sizeval is int value. need to consider setting order
                            cursize = self.get_propertyvalue(sizeid)
                            if cursize is False:
                                res = False
                            elif sizeval < cursize:
                                if self.set_propertyvalue(
                                    sizeid, sizeval
                                ) and self.set_propertyvalue(offsetid, offsetval):
                                    res = True
                                else:
                                    res = False
                            elif self.set_propertyvalue(
                                offsetid, offsetval
                            ) and self.set_propertyvalue(sizeid, sizeval):
                                res = True
                            else:
                                res = False
                elif is_offset_available:
                    # prompt to set offset only
                    res = self.prompt_propvalue(offsetid)
                elif is_size_available:
                    # prompt to set size only
                    res = self.prompt_propvalue(sizeid)
                else:
                    # nothing to configure
                    res = True

                return res

            # horizontal
            if prompt_offset_and_size(DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE) is False:
                return False

            # vertical
            if prompt_offset_and_size(DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE) is False:
                return False

            res = True

        return res
