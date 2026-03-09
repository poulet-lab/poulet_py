"""
Schema definitions and parsers for experiment metadata.

Each submodule or schema file corresponds to an experiment type (e.g. widefield).
JSON Schema files live in the project root schemas/ folder; this package
provides Pydantic models and parse/validate helpers for use in Python.
"""

from poulet_py.schemas.widefield import (
    DatasetInfo,
    WidefieldTrialMetadata,
    parse_widefield_trial_metadata,
)

__all__ = [
    "DatasetInfo",
    "WidefieldTrialMetadata",
    "parse_widefield_trial_metadata",
]
