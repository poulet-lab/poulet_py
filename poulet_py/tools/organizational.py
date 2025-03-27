try:
    from datetime import datetime, timezone
    from os import makedirs
    from pathlib import Path
    from re import sub

    from deprecated import deprecated
except ImportError as e:
    msg = "Missing 'tools' module. To install it use: pip install poulet_py[tools]"  # noqa: E501
    raise ImportError(msg) from e


@deprecated(
    "This function is deprecated and will be removed in the next release."
    "Use os.makedirs(path, exists_ok=True) instead.",
    version="0.0.2",
)
def check_or_create(path):
    """
    Function to check whether a folder exists.
    If it does NOT exist, it is created
    """
    makedirs(path, exist_ok=True)


@deprecated(
    "This function is deprecated and will be removed in the next release."
    "Use sanitize_path instead.",
    version="0.0.2",
)
def define_folder_name(name: str, *, add_date: bool = True) -> str:
    sanitized_name = sub(r"[^\w]", "_", name)

    if add_date:
        return (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{sanitized_name}"
        )
    return sanitized_name


def sanitize_path(path: Path | str, *, add_timestamp: bool = False) -> Path:
    """
    Sanitize a full path by making all components filesystem-friendly and
    optionally adding a timestamp to the last component (file or folder).

    Parameters
    ----------
    path : str
        The input path to sanitize (can include folders and filename)
    add_timestamp : bool, optional
        If True, prepend timestamp in YYYYMMDD format to the last component

    Returns
    -------
    str
        Sanitized path with all special characters replaced by underscores and
        optional timestamp on the last component

    Examples
    --------
    >>> sanitize_path("/s$ss!on1/d%ta/345.pkl", add_timestamp=True)
    '/s_ss_on1/d_ta/20250503_345.pkl'

    >>> sanitize_path("my@project/data#files")
    'my_project/data_files'
    """
    path_obj = str(path).lower()
    path_obj = path_obj.replace("\\", "/")
    path_obj = Path(path_obj)
    parts = path_obj.parts

    # Sanitize each component
    sanitized_parts = []
    for part in parts:
        # Skip root parts like '/' or 'C:\'
        if part in ("/") or (len(part) == 2 and part[1] == ":"):
            sanitized_parts.append(part)
        else:
            sanitized_parts.append(sub(r"[^\w\.\-]", "_", part).lower())
    # Add timestamp if requested
    if add_timestamp:
        sanitized_parts[-1] = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{sanitized_parts[-1]}"
        )

    return Path(*sanitized_parts)
