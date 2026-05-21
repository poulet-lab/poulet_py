try:
    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Any, ClassVar

    import h5py
    from h5py import File
    from numpy import array, ndarray
    from pandas import DataFrame, read_csv
    from pydantic import PrivateAttr
    from skimage.io import imread

    from poulet_py import BaseData, DataSignature, DataStructure
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class WidefieldData(BaseData, ABC):
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
        """Open the green reference image file and store it in self._reference_image."""
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
                attrs = self._metadata.get("analog_output", {}).get(name, {})
                sr = attrs.get("sr", "unknown")
                lines.append(f"  {name}: shape={data.shape}, sr={sr} Hz")

        mouse_id = (
            self._metadata.get("analog_output", {}).get("global", {}).get("mouse_id", "unknown")
        )
        protocol = (
            self._metadata.get("analog_output", {})
            .get("global", {})
            .get("protocol_name", "unknown")
        )
        comment = self._metadata.get("analog_output", {}).get("global", {}).get("comment", "")
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
        if "analog_output" not in self._metadata:
            self._metadata["analog_output"] = {}

        def _visit_datasets(name: str, obj: Any) -> None:
            self._metadata["analog_output"]["global"] = dict(f.attrs)
            if isinstance(obj, h5py.Dataset):
                self._metadata["analog_output"][name] = dict(obj.attrs)

        with File(self._analog_output_path, "r") as f:
            f.visititems(_visit_datasets)
