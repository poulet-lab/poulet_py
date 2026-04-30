try:
    from pathlib import Path
    from typing import Any, Literal

    from h5py import Dataset, File, Group
    from numpy import ndarray
    from orjson import OPT_SERIALIZE_DATACLASS, OPT_SERIALIZE_NUMPY, OPT_SERIALIZE_UUID, dumps
    from pydantic import Field, PrivateAttr

    from poulet_py import BaseEvent, BaseSink, SinkEvent

except ImportError as e:
    msg = """
Missing 'sinks' module. Install options:
- Dedicated:    pip install poulet_py[sinks]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class HDFSink(BaseSink):
    file: Path | str = Field(..., description="Path to the output file")
    compression: Literal["gzip", "lzf"] = Field(
        default="lzf", description="Compression algorithm to use"
    )
    compression_level: int = Field(default=4, description="Compression level for gzip (1-9)")
    grow_step: int = Field(
        default=500,
        description="Number of samples to grow the dataset by when capacity is exceeded",
    )

    _h5file: File = PrivateAttr()
    _sources: dict = PrivateAttr(default_factory=dict)

    def _open(self):
        path = Path(self.file)

        if path.exists():
            path.unlink()

        self._h5file = File(self.file, "w", libver="latest")
        self._write_meta(self._h5file, self.meta)

    def _close(self):
        if self._h5file:
            for src in self._sources.values():
                if src["dataset"]:
                    size = src["size"]
                    src["dataset"].resize((size, *src["fixed_shape"]))

            self._h5file.flush()
            self._h5file.close()
            del self._h5file

        self._sources.clear()

    def _write_meta(self, part: File | Group | Dataset, meta: dict[str, Any]):
        for k, v in meta.items():
            part.attrs[k] = dumps(
                v,
                option=OPT_SERIALIZE_DATACLASS | OPT_SERIALIZE_NUMPY | OPT_SERIALIZE_UUID,
            ).decode("utf-8")

    def _init_source(self, event: SinkEvent):
        name = event.name
        if name in self._sources:
            return

        self._sources[name] = {
            "dataset": None,
            "size": 0,
            "capacity": 0,
            "fixed_shape": None,
            "meta": event.meta,
        }

    def _init_dataset(self, src_name: str, array: ndarray):
        src = self._sources[src_name]

        if src["dataset"] is not None:
            return

        append_dim = array.shape[0]
        fixed_shape = array.shape[1:]

        initial_capacity = max(append_dim, self.grow_step)

        full_shape = (initial_capacity, *fixed_shape)
        maxshape = (None, *fixed_shape)

        dset = self._h5file.create_dataset(
            src_name,
            shape=full_shape,
            maxshape=maxshape,
            chunks=True,
            dtype=array.dtype,
            compression=self.compression,
            compression_opts=(self.compression_level if self.compression == "gzip" else None),
        )

        if src["meta"]:
            self._write_meta(dset, src["meta"])

        src["dataset"] = dset
        src["capacity"] = initial_capacity
        src["fixed_shape"] = fixed_shape

    def _grow_if_needed(self, src, append_len):
        required = src["size"] + append_len
        if required <= src["capacity"]:
            return

        new_capacity = required + self.grow_step
        new_shape = (new_capacity, *src["fixed_shape"])

        src["dataset"].resize(new_shape)
        src["capacity"] = new_capacity

    def _on_event(self, event: BaseEvent):
        if isinstance(event, SinkEvent):
            name = event.name
            self._init_source(event)

            array = event.payload

            if not isinstance(array, ndarray):
                return

            arr = array.reshape(1) if array.ndim == 0 else array

            self._init_dataset(name, arr)

            src = self._sources[name]

            if arr.shape[1:] != src["fixed_shape"]:
                msg = (
                    f"Shape mismatch for source '{name}'. "
                    f"Expected (*, {src['fixed_shape']}), "
                    f"got {arr.shape}"
                )
                raise ValueError(msg)

            append_len = arr.shape[0]
            self._grow_if_needed(src, append_len)

            start = src["size"]
            end = start + append_len

            src["dataset"][start:end] = arr
            src["size"] = end
