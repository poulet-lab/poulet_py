try:
    from time import time_ns

    from orjson import OPT_SERIALIZE_DATACLASS, OPT_SERIALIZE_NUMPY, OPT_SERIALIZE_UUID, dumps
    from pydantic import Field

    from poulet_py import LOGGER, BaseSource
except ImportError as e:
    msg = """
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class StimuliMetadataSource(BaseSource):
    max_string_length: int = Field(
        default=512,
        description="Maximum length of the JSON string for metadata. "
        "Longer strings will be truncated.",
    )

    def _set_buffer_dtype(self):
        self._buffer_dtype = [("timestamp", "uint64"), ("metadata", f"S{self.max_string_length}")]

    def _open(self):
        pass

    def _close(self):
        pass

    def _fire(self) -> bool:
        meta = [st.model_dump(exclude_unset=True, exclude_none=True) for st in self._stimuli]
        timestamp = time_ns()

        json_data = dumps(
            meta,
            option=OPT_SERIALIZE_DATACLASS | OPT_SERIALIZE_NUMPY | OPT_SERIALIZE_UUID,
        )

        if len(json_data) >= self.max_string_length:
            LOGGER.warning(
                f"Metadata JSON truncated from {len(json_data)} to {self.max_string_length} bytes"
            )
            json_data = json_data[: self.max_string_length]

        with self._lock:
            idx = self._buffer_idx % self.buffer_size
            self._buffer[idx]["timestamp"] = timestamp
            self._buffer[idx]["metadata"] = json_data
            self._buffer_idx += 1

        return True
