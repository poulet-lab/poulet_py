try:
    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Any, ClassVar

    import h5py
    from h5py import File
    from numpy import array, ndarray
    from pandas import DataFrame, read_csv
    from pydantic import BaseModel, Field, PrivateAttr
    from skimage.io import imread

    from poulet_py import BaseData, BaseMetadata, DataSignature, DataStructure
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class WidefieldMaskMetaData(BaseModel):
    center: tuple[float, float] = Field(default=(0.0, 0.0))
    radius: float = Field(default=0.0)


class WidefieldCameraMetadata(BaseModel):
    """Camera and optical settings stored in a v1 acquisition file."""

    format: str = Field(default="")
    fps: int = Field(default=0)
    exposure: float = Field(default=0.0)
    roi_active: bool = Field(default=False)
    roi: tuple[int, int, int, int] | None = Field(default=None)
    binning: int = Field(default=0)
    magnification: float = Field(default=0.0)
    filterset: str = Field(default="")
    led_power: float = Field(default=0.0)


class WidefieldSubjectMetadata(BaseModel):
    """Subject and preparation details stored in a v1 acquisition file."""

    mouse_id: str = Field(default="")
    weight: float = Field(default=0.0)
    anesthesia: str = Field(default="")
    isoflurane: float = Field(default=0.0)


class WidefieldAcquisitionMetadata(BaseModel):
    """Trial context stored in a v1 acquisition file."""

    protocol_name: str = Field(default="")
    time: str = Field(default="")
    timestamp: float = Field(default=0.0)
    experimenter: str = Field(default="")
    comment: str = Field(default="")
    folder: str = Field(default="")


class WidefieldChannelMetadata(BaseModel):
    """Attributes of one channel recorded alongside the imaging."""

    id: str = Field(default="")
    name: str = Field(default="")
    sr: int = Field(default=0)
    device: str = Field(default="")


class WidefieldMetadata(BaseMetadata):
    """Typed metadata read from a v1 widefield acquisition file."""

    mask_data: WidefieldMaskMetaData | None = Field(default=None)
    camera: WidefieldCameraMetadata = Field(default_factory=WidefieldCameraMetadata)
    subject: WidefieldSubjectMetadata = Field(default_factory=WidefieldSubjectMetadata)
    acquisition: WidefieldAcquisitionMetadata = Field(default_factory=WidefieldAcquisitionMetadata)
    analog_output: dict[str, WidefieldChannelMetadata] = Field(default_factory=dict)


class WidefieldData(BaseData[WidefieldMetadata], ABC):
    metadata: WidefieldMetadata = Field(default_factory=WidefieldMetadata)

    _imaging: ndarray[Any, Any] = PrivateAttr()
    _reference_image: ndarray[Any, Any] = PrivateAttr()
    _timestamps: DataFrame = PrivateAttr()
    _analog_output: dict[str, ndarray[Any, Any]] = PrivateAttr(default_factory=dict)

    @property
    def imaging(self):
        if not hasattr(self, "_imaging"):
            self._open_imaging()
        return self._imaging

    @property
    def reference_image(self):
        if not hasattr(self, "_reference_image"):
            self._open_reference_image()
        return self._reference_image

    @property
    def timestamps(self):
        if not hasattr(self, "_timestamps"):
            self._open_timestamps()
        return self._timestamps

    @property
    def analog_output(self):
        if not self._analog_output:
            self._open_analog_output()
        return self._analog_output

    @abstractmethod
    def _open_imaging(self) -> None:
        """Open the imaging data file and store it in self._imaging."""
        ...

    @abstractmethod
    def _open_reference_image(self):
        """Open the green reference image and store it."""
        ...

    @abstractmethod
    def _open_timestamps(self):
        """Open the timestamps file and store it in self._timestamps."""
        ...

    @abstractmethod
    def _open_analog_output(self) -> None:
        """Open the analog output file and store it in self._analog_output"""
        ...

    def __new__(cls, path: Path, **kwargs):
        if cls is WidefieldData:
            if WidefieldDataV1.DATA_SIGNATURE.matches(path):
                return WidefieldDataV1(path=path, **kwargs)
            else:
                msg = f"Unknown data structure for path: {path}"
                raise ValueError(msg)

        return super().__new__(cls)

    def summary(self) -> str:
        lines = ["=" * 60, f"Trial: {self.path.name}", "=" * 60]

        n_frames, height, width = self.imaging.shape
        lines.extend(
            [
                "Imaging data:",
                f"  Shape: {self.imaging.shape}",
                f"  Frames: {n_frames}",
                f"  Resolution: {width} x {height}",
                f"  Dtype: {self.imaging.dtype}",
                (f"  Value range: [{self.imaging.min()}, {self.imaging.max()}]"),
            ]
        )
        size_mb = self.imaging.nbytes / (1024 * 1024)
        lines.append(f"  Memory: {size_mb:.1f} MB")

        lines.extend(
            [
                "Timestamps:",
                f"  Rows: {len(self.timestamps)}",
                f"  Columns: {list(self.timestamps.columns)}",
            ]
        )

        if self.analog_output:
            lines.append("Analog output data:")
            for name, data in self.analog_output.items():
                channel = self.metadata.analog_output.get(name)
                sr = channel.sr if channel else "unknown"
                lines.append(f"  {name}: shape={data.shape}, sr={sr} Hz")

        mouse_id = self.metadata.subject.mouse_id
        protocol = self.metadata.acquisition.protocol_name
        comment = self.metadata.acquisition.comment
        lines.extend(["Metadata:", f"  Mouse: {mouse_id}", f"  Protocol: {protocol}"])
        if comment:
            lines.append(f"  Comment: {comment}")

        lines.append("=" * 60)
        return "\n".join(lines)


class WidefieldDataV1(WidefieldData):
    DATA_SIGNATURE: ClassVar[DataSignature] = DataSignature(
        data_structure=DataStructure.FOLDER_PER_TRIAL,
        data_type=WidefieldData,
        files=["recording.tif*", "recording.csv", "data.h5", "green.tif*"],
    )

    _imaging_path: Path = PrivateAttr()
    _timestamps_path: Path = PrivateAttr()
    _analog_output_path: Path = PrivateAttr()
    _reference_image_path: Path = PrivateAttr()

    def model_post_init(self, __context):
        # TODO use dataframe from session?
        self._imaging_path = self.path.glob("recording.tif*").__next__()
        self._timestamps_path = self.path.glob("recording.csv").__next__()
        self._analog_output_path = self.path.glob("data.h5").__next__()
        self._reference_image_path = self.path.glob("green.tif*").__next__()
        self._analog_output_metadata()

    def _open_imaging(self) -> None:
        self._imaging = imread(str(self._imaging_path))

    def _open_reference_image(self):
        image = imread(str(self._reference_image_path))

        if image.ndim == 3:
            image = image[0]

        self._reference_image = image

    def _open_timestamps(self):
        self._timestamps = read_csv(self._timestamps_path, sep=";")
        self._timestamps = self._timestamps.loc[
            :, ~self._timestamps.columns.str.contains("^Unnamed")
        ]

    def _open_analog_output(self):
        def _visit_datasets(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                self._analog_output[name] = array(obj)

        with File(self._analog_output_path, "r") as f:
            f.visititems(_visit_datasets)

    def _analog_output_metadata(self):
        def _visit_datasets(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                self.metadata.analog_output[name] = WidefieldChannelMetadata(
                    id=obj.attrs.get("id", ""),
                    name=obj.attrs.get("name", ""),
                    sr=obj.attrs["sr"],
                    device=obj.attrs.get("device", ""),
                )

        with File(self._analog_output_path, "r") as f:
            attributes = f.attrs
            self.metadata = WidefieldMetadata(
                level=self.metadata.level,
                mask_data=self.metadata.mask_data,
                camera=WidefieldCameraMetadata(
                    format=attributes.get("camera_format", ""),
                    fps=attributes["camera_fps"],
                    exposure=attributes["camera_exposure"],
                    roi_active=attributes["camera_roi_active"],
                    roi=attributes.get("camera_roi"),
                    binning=attributes["binning"],
                    magnification=attributes["magnification"],
                    filterset=attributes.get("filterset", ""),
                    led_power=attributes["led_power"],
                ),
                subject=WidefieldSubjectMetadata(
                    mouse_id=attributes.get("mouse_id", ""),
                    weight=attributes["weight"],
                    anesthesia=attributes.get("anesthesia", ""),
                    isoflurane=attributes["isoflurane"],
                ),
                acquisition=WidefieldAcquisitionMetadata(
                    protocol_name=attributes.get("protocol_name", ""),
                    time=attributes.get("time", ""),
                    timestamp=attributes["timestamp"],
                    experimenter=attributes.get("experimenter", ""),
                    comment=attributes.get("comment", ""),
                    folder=attributes.get("folder", ""),
                ),
            )
            f.visititems(_visit_datasets)
