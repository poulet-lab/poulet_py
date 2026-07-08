try:
    from time import monotonic_ns

    from orjson import OPT_SERIALIZE_DATACLASS, OPT_SERIALIZE_NUMPY, OPT_SERIALIZE_UUID, dumps
    from pydantic import Field

    from poulet_py import LOGGER, BaseSource
except ImportError as e:
    raise ImportError("""
Missing 'sources' module. Install options:
- Dedicated:    pip install poulet_py[sources]
- Module:       pip install poulet_py[io]
- Full:         pip install poulet_py[all]
""") from e


class StimuliMetadataSource(BaseSource):
    max_string_length: int = Field(
        default=512,
        description="Maximum length of the JSON string for metadata. "
        "Longer strings will be truncated.",
    )

    def _set_buffer_dtype(self):
        self._source_buffer_dtype = [
            ("timestamp", "uint64"),
            ("metadata", f"S{self.max_string_length}"),
        ]

    def _open(self):
        pass

    def _close(self):
        pass

    def _fire(self) -> bool:
        meta = [
            {type(st).__name__: st.model_dump(exclude_unset=True, exclude_none=True)}
            for st in self._stimuli
        ]
        timestamp = monotonic_ns()

        json_data = dumps(
            meta,
            option=OPT_SERIALIZE_DATACLASS | OPT_SERIALIZE_NUMPY | OPT_SERIALIZE_UUID,
        )

        if len(json_data) >= self.max_string_length:
            LOGGER.warning(
                f"Metadata JSON truncated from {len(json_data)} to {self.max_string_length} bytes"
            )
            json_data = json_data[: self.max_string_length]

        self._write_sample((timestamp, json_data))

        return True
