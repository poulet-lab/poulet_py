from numpy import ndarray

try:
    from pathlib import Path
    from typing import Any, Literal

    from h5py import File, Group
    from orjson import OPT_SERIALIZE_DATACLASS, OPT_SERIALIZE_NUMPY, OPT_SERIALIZE_UUID, dumps
    from pydantic import Field, PrivateAttr

    from poulet_py import BaseDataPacket, BaseSink

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
        default=10_000,
        description="Number of samples to grow the dataset by when capacity is exceeded",
    )

    _h5file: File | None = PrivateAttr(default=None)
    _sources: dict = PrivateAttr(default_factory=dict)

    def _init(self):
        path = Path(self.file)

        if path.exists():
            path.unlink()

        self._h5file = File(self.file, "w", libver="latest")
        self._write_meta(self._h5file, self.meta)

    def _close(self):
        if self._h5file:
            for src in self._sources.values():
                for dataset in src["datasets"].values():
                    size = dataset["size"]
                    dataset["data"].resize((size, *dataset["fixed_shape"]))

            self._h5file.flush()
            self._h5file.close()
            self._h5file = None

        self._sources.clear()

    def _write_meta(self, part: File | Group, meta: dict[str, Any]):
        for k, v in meta.items():
            part.attrs[k] = dumps(
                v,
                option=OPT_SERIALIZE_DATACLASS | OPT_SERIALIZE_NUMPY | OPT_SERIALIZE_UUID,
            ).decode("utf-8")

    def _init_source(self, packet: BaseDataPacket):
        name = packet.name
        if name in self._sources:
            return

        grp = self._h5file.require_group(name)

        self._write_meta(grp, packet.meta)

        self._sources[name] = {"group": grp, "datasets": {}}

    def _init_dataset(self, src_name: str, dataset_name: str, array: ndarray):
        src = self._sources[src_name]

        if dataset_name in src["datasets"]:
            return

        grp = src["group"]

        append_dim = array.shape[0]
        fixed_shape = array.shape[1:]

        initial_capacity = max(append_dim, self.grow_step)

        full_shape = (initial_capacity, *fixed_shape)
        maxshape = (None, *fixed_shape)

        dset = grp.create_dataset(
            dataset_name,
            shape=full_shape,
            maxshape=maxshape,
            chunks=True,
            dtype=array.dtype,
            compression=self.compression,
            compression_opts=(self.compression_level if self.compression == "gzip" else None),
        )

        src["datasets"][dataset_name] = {
            "data": dset,
            "size": 0,
            "capacity": initial_capacity,
            "fixed_shape": fixed_shape,
        }

    def _grow_if_needed(self, dataset, append_len):
        required = dataset["size"] + append_len
        if required <= dataset["capacity"]:
            return

        new_capacity = required + self.grow_step
        new_shape = (new_capacity, *dataset["fixed_shape"])

        dataset["data"].resize(new_shape)
        dataset["capacity"] = new_capacity

    def _write(self, packet: BaseDataPacket):
        name = packet.name
        self._init_source(packet)

        src = self._sources[name]

        for dataset_name, array in packet.data.items():
            if array.ndim == 0:
                array = array.reshape(1)

            self._init_dataset(name, dataset_name, array)

            dataset = src["datasets"][dataset_name]

            # Validate fixed dimensions
            if array.shape[1:] != dataset["fixed_shape"]:
                msg = (
                    f"Shape mismatch for dataset '{dataset_name}'. "
                    f"Expected (*, {dataset['fixed_shape']}), "
                    f"got {array.shape}"
                )
                raise ValueError(msg)

            append_len = array.shape[0]
            self._grow_if_needed(dataset, append_len)

            start = dataset["size"]
            end = start + append_len

            dataset["data"][start:end] = array
            dataset["size"] = end
