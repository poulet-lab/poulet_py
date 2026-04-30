try:
    from pathlib import Path
    from typing import Any, ClassVar

    import h5py
    from h5py import File
    from numpy import array, load, ndarray
    from pandas import DataFrame, read_csv
    from pydantic import PrivateAttr
    from skimage.io import imread

    from poulet_py import LOGGER, BaseData
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class WidefieldData(BaseData):
    VERSION_MARKERS: ClassVar[dict[int, list[str]]] = {
        1: [
            "recording.tif*",
            "timestamps.csv",
            "data.h5",
            "green.tif*",
        ]
    }

    _imaging_data: ndarray[Any, Any] = PrivateAttr()
    _reference_image: ndarray[Any, Any] = PrivateAttr()
    _timestamps: DataFrame = PrivateAttr()
    
    # TODO do we need separate analog output data?
    _analog_output_data: dict[str, ndarray[Any, Any]] = PrivateAttr(default_factory=dict)
    _analog_output_data_attrs: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _analog_output_data_file_attrs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _condition: dict[str, Any] = PrivateAttr(default_factory=dict)
    _roi: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def imaging_data(self):
        return self._imaging_data

    @property
    def reference_image(self):
        return self._reference_image

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def analog_output_data(self):
        return self._analog_output_data

    @property
    def analog_output_data_attrs(self):
        return self._analog_output_data_attrs

    @property
    def analog_output_data_file_attrs(self):
        return self._analog_output_data_file_attrs

    @property
    def condition(self):
        return self._condition

    @property
    def roi(self):
        return self._roi

    def __new__(cls, path: Path, **kwargs):
        if cls is WidefieldData:
            version = cls._detect_version(path)
            if version == 1:
                return WidefieldDataV1(path=path, **kwargs)

        return super().__new__(cls)

    @classmethod
    def _detect_version(cls, path: Path) -> int:
        """Detect version by checking for marker files."""
        for version, markers in cls.VERSION_MARKERS.items():
            if all(any(path.glob(marker)) for marker in markers):
                return version
        msg = f"Unknown data structure in: {path}"
        raise ValueError(msg)

    def _open(self) -> None:
        pass

    def _close(self):
        del self._imaging_data
        del self._reference_image
        del self._timestamps

        self._analog_output_data = {}
        self._analog_output_data_attrs = {}
        self._analog_output_data_file_attrs = {}
        self._condition = {}
        self._roi = {}

    # TODO: add metadata function to create dicitonary of all dictionary, then in summary use the function metadata in summary
    def summary(self) -> str:
        self._ensure_open()

        lines = ["=" * 60, f"Trial: {self.path.name}", "=" * 60]

        n_frames, height, width = self.imaging_data.shape
        lines.extend(
            [
                "Imaging data:",
                f"  Shape: {self.imaging_data.shape}",
                f"  Frames: {n_frames}",
                f"  Resolution: {width} x {height}",
                f"  Dtype: {self.imaging_data.dtype}",
                (f"  Value range: [{self.imaging_data.min()}, {self.imaging_data.max()}]"),
            ]
        )
        size_mb = self.imaging_data.nbytes / (1024 * 1024)
        lines.append(f"  Memory: {size_mb:.1f} MB")

        lines.extend(
            [
                "Timestamps:",
                f"  Rows: {len(self.timestamps)}",
                f"  Columns: {list(self.timestamps.columns)}",
            ]
        )

        if self.analog_output_data:
            lines.append("Analog output data:")
            for name, data in self.analog_output_data.items():
                attrs = self.analog_output_data_attrs.get(name, {})
                sr = attrs.get("sr", "unknown")
                lines.append(f"  {name}: shape={data.shape}, sr={sr} Hz")

        if self.analog_output_data_file_attrs:
            mouse_id = self.analog_output_data_file_attrs.get("mouse_id", "unknown")
            protocol = self.analog_output_data_file_attrs.get("protocol_name", "unknown")
            comment = self.analog_output_data_file_attrs.get("comment", "")
            lines.extend(["Metadata:", f"  Mouse: {mouse_id}", f"  Protocol: {protocol}"])
            if comment:
                lines.append(f"  Comment: {comment}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()



class WidefieldDataV1(WidefieldData):
    VERSION: ClassVar[int] = 1

    _imaging_path: Path = PrivateAttr()
    _timestamps_path: Path = PrivateAttr()
    _analog_output_data_path: Path = PrivateAttr()
    _reference_image_path: Path = PrivateAttr()

    def to_h5(self, output_path: Path) -> None:
        self._ensure_open()

        with File(output_path, "w") as f:
            f.create_dataset("imaging_data", data=self.imaging_data, compression="gzip")
            f.create_dataset("reference_image", data=self.reference_image, compression="gzip")
            f.create_dataset("timestamps", data=self.timestamps, compression="gzip")

            analog_group = f.create_group("analog_output_data")
            for name, data in self.analog_output_data.items():
                dataset = analog_group.create_dataset(name, data=data, compression="gzip")
                for attr_name, attr_value in self.analog_output_data_attrs.get(name, {}).items():
                    dataset.attrs[attr_name] = attr_value

            for attr_name, attr_value in self.analog_output_data_file_attrs.items():
                f.attrs[attr_name] = attr_value

    def _open(self) -> None:
        self._resolve_paths()
        self._open_imaging()
        self._open_reference_image()
        self._open_timestamps()
        self._open_analog_output()
        WidefieldData._open(self)

    def _close(self):
        WidefieldData._close(self)

        del self._imaging_path
        del self._timestamps_path
        del self._analog_output_data_path
        del self._reference_image_path

    def _resolve_paths(self) -> None:
        video_candidates = ["recording.tiff", "recording.tif", "recording.npy"]

        for candidate in video_candidates:
            candidate_path = self.path / candidate
            if candidate_path.exists():
                self.imaging_path = candidate_path
                break

        self.timestamps_path = self.path / "recording.csv"
        if not self.timestamps_path.exists():
            msg = f"Timestamps file not found: {self.timestamps_path}"
            raise FileNotFoundError(msg)

        self.analog_output_data_path = self.path / "data.h5"
        if not self.analog_output_data_path.exists():
            msg = f"Analog output data not found: {self.analog_output_data_path}"
            raise FileNotFoundError(msg)

        self.reference_image_path = self.path / "green.tiff"
        if not self.reference_image_path.exists():
            msg = f"Green reference image not found: {self.reference_image_path}"
            raise FileNotFoundError(msg)

    def _open_imaging(self) -> None:
        LOGGER.info(f"Opening imaging data from: {self.imaging_path.name}")

        if self.imaging_path.suffix.lower() == ".npy":
            data = load(str(self.imaging_path))

        elif self.imaging_path.suffix.lower() in (".tiff", ".tif"):
            data = imread(str(self.imaging_path))
        else:
            msg = f"Unsupported imaging format: {self.imaging_path.suffix}"
            raise ValueError(msg)

        LOGGER.info(f"Opened imaging stack: {data.shape}")

        self._imaging_data = data

    def _open_reference_image(self):
        LOGGER.info(f"Opening green reference from: {self.reference_image_path.name}")

        image = imread(str(self.reference_image_path))

        if image.ndim == 3:
            image = image[0]

        LOGGER.info(f"Opened green reference: {image.shape}")

        self._reference_image = image

    def _open_timestamps(self):
        try:
            self._timestamps = read_csv(self.timestamps_path, sep=";")
            self._timestamps = self._timestamps.loc[
                :, ~self._timestamps.columns.str.contains("^Unnamed")
            ]

            LOGGER.info(f"Opened timestamps: {len(self._timestamps)} rows")
        except Exception:
            LOGGER.exception(f"Error loading timestamps: {self.timestamps_path}")

    def _open_analog_output(self):
        try:
            with File(self.analog_output_data_path, "r") as f:
                self._analog_output_data_file_attrs = dict(f.attrs)

                def _visit_datasets(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        self._analog_output_data[name] = array(obj)
                        self._analog_output_data_attrs[name] = dict(obj.attrs)

                f.visititems(_visit_datasets)

        except Exception:
            LOGGER.exception(f"Error loading H5: {self.analog_output_data_path}")

    #TODO
    def _should_open(self) -> bool:
            if isinstance(self.start, datetime) and isinstance(self.end, datetime):
                folder_time = self._folder_datetime()
                return folder_time is not None and self.start <= folder_time <= self.end

            if isinstance(self.start, int) and isinstance(self.end, int):
                trial_number = self._folder_trial_number()
                return trial_number is not None and self.start <= trial_number <= self.end

            msg = "start and end must both be datetime values or both be integer trial numbers"
            raise TypeError(msg)

    def _folder_datetime(self) -> datetime | None:
        folder_name = self.path.name
        for date_format in ("%y%m%d_%H%M%S", "%Y%m%d_%H%M%S"):
            # TODO: add just time not date
            # TODO: also option to add internal time, so like u say the trials within the first 2 minutes
            try:
                return datetime.strptime(folder_name, date_format)
            except ValueError:
                continue
        return None

    def _folder_trial_number(self) -> int | None:
        match = re.fullmatch(r"(?:trial[_-]?)?0*(\d+)", self.path.name, flags=re.IGNORECASE)
        if match is None:
            return None
        return int(match.group(1))
