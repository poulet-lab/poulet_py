try:
    from pathlib import Path
    from typing import Any


except ImportError as e:
    raise ImportError("""
Missing 'tools' module. Install options:
- Module:       pip install poulet_py[tools]
- Full:         pip install poulet_py[all]
""") from e

try:
    from orjson import (
        OPT_INDENT_2,
        OPT_SERIALIZE_DATACLASS,
        OPT_SERIALIZE_NUMPY,
        OPT_SERIALIZE_UUID,
    )
    from orjson import dumps as jdumps

    def dumps(obj: Any) -> bytes:
        return jdumps(
            obj,
            option=OPT_SERIALIZE_NUMPY
            | OPT_INDENT_2
            | OPT_SERIALIZE_DATACLASS
            | OPT_SERIALIZE_UUID,
        )
except ImportError:
    from json import JSONEncoder
    from json import dumps as jdumps

    from numpy import floating, integer, ndarray

    class JEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, ndarray):
                return o.tolist()
            if isinstance(o, integer):
                return int(o)
            if isinstance(o, floating):
                return float(o)
            return super().default(o)

    def dumps(obj: Any) -> bytes:
        return jdumps(obj, cls=JEncoder).encode("utf-8")


def json_serializer(
    data: dict[str, Any],
    file: Path | str | None = None,
) -> bytes | None:
    """Serialize data to JSON.

    Features:
    - Returns bytes if no file provided, writes to file otherwise

    Args:
        data: Dictionary containing data to serialize.
            Numpy arrays are automatically supported.
        file: Output file
            If None, returns serialized bytes instead of writing to file.
            Must end with '.json' extension if provided.

    Returns:
        bytes | None: Serialized JSON as bytes if file is None, otherwise None.

    Raises:
        ValueError: If provided file doesn't end with '.json' extension
        TypeError: If data contains non-serializable types
        JSONEncodeError: If serialization fails for other reasons

    Example:
        >>> data = {"array": np.array([1, 2, 3])}
        >>> # Write to file
        >>> json_serializer(data, "output.json")
        >>> # Get bytes
        >>> json_bytes = json_serializer(data)
    """
    serialized = dumps(data)
    if file is not None:
        path = Path(file) if isinstance(file, str) else file

        if path.suffix.lower() != ".json":
            raise ValueError("file must end with '.json' extension")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(serialized)
        return None

    return serialized
