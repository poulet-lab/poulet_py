try:
    from json import dump
    from os import makedirs
    from os.path import join
    from pathlib import Path
    from typing import Any

    from deprecated import deprecated
    from orjson import OPT_INDENT_2, OPT_SERIALIZE_NUMPY, JSONEncodeError, dumps
except ImportError as e:
    msg = """
Missing 'tools' module. Install options:
- Module:       pip install poulet_py[tools]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


_SERIALIZATION_OPTIONS = OPT_SERIALIZE_NUMPY | OPT_INDENT_2


def json_serializer(
    data: dict[str, Any],
    file: Path | str | None = None,
) -> bytes | None:
    """Serialize data to JSON.

    Features:
    - Efficient serialization using orjson (faster than standard json module)
    - Built-in numpy array support via orjson's native serialization
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
    try:
        serialized = dumps(data, option=_SERIALIZATION_OPTIONS)
    except JSONEncodeError as e:
        msg = "Failed to serialize data to JSON"
        raise TypeError(msg) from e

    if file is not None:
        path = Path(file) if isinstance(file, str) else file

        if path.suffix.lower() != ".json":
            msg = "file must end with '.json' extension"
            raise ValueError(msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(serialized)
        return None

    return serialized


@deprecated(
    "This function is deprecated and will be removed in the next release."
    "Use json_serializer() instead.",
    version="0.0.2",
)
def save_metadata_exp(metadata, path, name):
    makedirs(path, exist_ok=True)
    metadata_file_name = f"{name}.json"
    metadata_path = join(path, metadata_file_name)

    with open(metadata_path, "w") as f:
        dump(metadata, f, indent=4)
