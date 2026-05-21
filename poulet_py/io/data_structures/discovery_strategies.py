try:
    from abc import ABC, abstractmethod
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from pathlib import Path
    from re import Pattern, compile, escape, finditer
    from typing import Any, ClassVar

    from pandas import DataFrame, concat, json_normalize
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[dtst]
- Module group: pip install poulet_py[io]
- Full:         pip install poulet_py[all]
"""
    raise ImportError(msg) from e


class BasePattern(BaseModel, ABC):
    @abstractmethod
    def matches(self, path: Path) -> bool: ...

    @abstractmethod
    def meta(self, path: Path) -> dict[str, Any] | None: ...


class PathPattern(BasePattern):
    DEFAULT_PATTERNS: ClassVar[dict[str, tuple[str, str]]] = {
        "date": (r"\d{4}-\d{2}-\d{2}|\d{8}", "datetime"),
        "datetime": (r"\d{8}_\d{6}", "datetime"),
        "subject_id": (r"[A-Za-z0-9_-]+", "str"),
        "trial": (r"\d+", "int"),
        "session": (r"\d+", "int"),
        "method": (r"[A-Za-z0-9_]+", "str"),
        "experiment": (r"[A-Za-z0-9_-]+", "str"),
        "run": (r"\d+", "int"),
        "*": (r"[^/]+", "str"),
        "**": (r".+", "str"),
    }

    TYPE_CONVERTERS: ClassVar[dict[str, Callable]] = {
        "int": int,
        "float": float,
        "str": str,
        "datetime": lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S"),
        "date": lambda x: datetime.strptime(x.replace("-", ""), "%Y%m%d").date(),
        "bool": lambda x: x.lower() in ("true", "1", "yes"),
    }

    pattern: str = Field(
        ...,
        description="Path pattern with {field} placeholders, e.g., '/data/{date}/{subject_id}/{trial}'",
    )
    custom_fields: dict[str, tuple[str, str] | tuple[str, str, Callable]] = Field(
        default_factory=dict,
        description="Custom regex for specific fields",
    )

    _compiled_pattern: Pattern = PrivateAttr()
    _field_names: Sequence[str] = PrivateAttr(default_factory=list)

    @property
    def compiled_pattern(self) -> Pattern:
        return self._compiled_pattern

    def model_post_init(self, __context):
        self._compiled_pattern = self._compile_pattern()

    def matches(self, path: Path) -> bool:
        return self.compiled_pattern.match(str(path)) is not None

    def meta(self, path: Path) -> dict[str, Any] | None:
        match_obj = self.compiled_pattern.match(str(path))

        if not match_obj:
            return None

        fields = match_obj.groupdict()

        for field_name, value in fields.items():
            field = self._get_field_pattern(field_name)

            if len(field) == 3:
                _, field_type, custom_converter = field
                converter = (
                    custom_converter
                    if custom_converter
                    else self.TYPE_CONVERTERS.get(field_type, str)
                )
            else:
                _, field_type = field
                converter = self.TYPE_CONVERTERS.get(field_type, str)

            try:
                fields[field_name] = converter(value)
            except (ValueError, TypeError) as e:
                LOGGER.warning(f"Failed to convert field '{field_name}' with value '{value}': {e}")

        return fields

    def _get_field_pattern(self, field_name: str) -> tuple[str, str] | tuple[str, str, Callable]:
        if field_name in self.custom_fields:
            return self.custom_fields[field_name]
        return self.DEFAULT_PATTERNS.get(field_name, self.DEFAULT_PATTERNS["*"])

    def _compile_pattern(self) -> Pattern:
        field_names = []
        regex_parts = []

        last_end = 0
        for match_obj in finditer(r"\{([^}]+)\}", self.pattern):
            literal_part = escape(self.pattern[last_end : match_obj.start()])
            regex_parts.append(literal_part)

            field_name = match_obj.group(1)
            field_names.append(field_name)
            field_pattern = self._get_field_pattern(field_name)[0]
            regex_parts.append(f"(?P<{field_name}>{field_pattern})")

            last_end = match_obj.end()

        regex_parts.append(escape(self.pattern[last_end:]))

        self._field_names = field_names
        return compile(".*" + "".join(regex_parts) + ".*")


class DiscoveryStrategy(BaseModel, ABC):
    @staticmethod
    def path_stats(path: Path) -> dict[str, Any]:
        return {
            "name": path.name,
            "parent": path.parent,
            "is_dir": path.is_dir(),
            "size": path.stat().st_size if path.is_file() else None,
        }

    @abstractmethod
    def discover(
        self, root_path: Path = Path(".")
    ) -> tuple[Sequence[Path], Sequence[dict[str, Any] | None] | None]:
        """Discover all data matching this strategy."""
        ...

    def to_df(self, root_path: Path = Path(".")) -> DataFrame:
        paths, meta = self.discover(root_path)

        if not paths:
            return DataFrame()

        data = []
        for path in paths:
            row = self.path_stats(path)
            data.append(row)

        df = DataFrame(data)

        if meta and any(meta):
            meta_df = json_normalize([m if m is not None else {} for m in meta])
            df = concat([df, meta_df], axis=1)

        return df


class PatternBasedDiscovery(DiscoveryStrategy):
    patterns: Sequence[BasePattern] = Field(
        ..., description="Ordered sequence of path patterns to try"
    )

    def discover(
        self, root_path: Path = Path(".")
    ) -> tuple[Sequence[Path], Sequence[dict[str, Any] | None] | None]:
        paths = []
        meta = []

        for path in root_path.rglob("*"):
            for pattern in self.patterns:
                if pattern.matches(path):
                    meta.append(pattern.meta(path))
                    paths.append(path)

        return paths, meta


class ExplicitDiscovery(DiscoveryStrategy):
    paths: Sequence[Path] = Field(..., description="Explicit sequence of data paths")

    def discover(
        self, root_path: Path = Path(".")
    ) -> tuple[Sequence[Path], Sequence[dict[str, Any] | None] | None]:
        paths = []

        for p in self.paths:
            path = root_path / p
            if path.exists():
                paths.append(path)

        return paths, None


class GlobBasedDiscovery(DiscoveryStrategy):
    pattern: str = Field(..., description="Glob pattern for finding sessions")

    def discover(
        self, root_path: Path = Path(".")
    ) -> tuple[Sequence[Path], Sequence[dict[str, Any] | None] | None]:
        files = []

        for path in sorted(root_path.glob(self.pattern)):
            files.append(path)

        return files, None
