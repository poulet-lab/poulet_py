try:
    import h5py

    from poulet_py import DataPacket, Writer
except ImportError as e:
    msg = """
Missing 'writers' module. Install options:
- Dedicated:    pip install poulet_py[writers]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class HDFWriter(Writer):
    def __init__(self, path: str):
        self.file = h5py.File(path, "w")
        self.datasets = {}


def write(self, packet: DataPacket):
    key = packet.source

    if key not in self.datasets:
        self.datasets[key] = self.file.create_dataset(
            key,
            shape=packet.data.shape,
            maxshape=(packet.data.shape[0], None),
            chunks=True,
            dtype=packet.data.dtype,
        )
        self.datasets[key].attrs["channels"] = packet.channels

    dset = self.datasets[key]
    old_len = dset.shape[1]
    new_len = old_len + packet.data.shape[1]

    dset.resize((dset.shape[0], new_len))
    dset[:, old_len:new_len] = packet.data
