"""
Widefield trial metadata schema and parsing.

Pydantic models and helpers for parsing/validating widefield trial metadata
that conforms to the JSON schema in the project's schemas/ folder.
"""

from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Info for a single HDF5 dataset in trial metadata."""

    shape: list[int] = Field(..., description="Dataset shape (e.g. [n_frames]).")
    dtype: str = Field(..., description="NumPy dtype as string.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset-level HDF5 attributes.",
    )


class WidefieldTrialMetadata(BaseModel):
    """
    Standardized widefield trial metadata.

    Matches the structure defined in schemas/widefield_trial_metadata.json.
    Use parse_widefield_trial_metadata() to load and validate a dict.
    """

    source_h5: str = Field(..., description="Original HDF5 file path.")
    trial_name: str = Field(..., description="Trial name (e.g. parent folder).")
    file_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="File-level HDF5 attributes (mouse_id, protocol_name, etc.).",
    )
    datasets: dict[str, DatasetInfo] = Field(
        default_factory=dict,
        description="Per-dataset info (shape, dtype, attributes).",
    )
    parsed: dict[str, Any] | None = Field(
        default=None,
        description="Optional parsed fields (e.g. temperature range).",
    )


def parse_widefield_trial_metadata(data: dict[str, Any]) -> WidefieldTrialMetadata:
    """
    Parse and validate a dictionary as widefield trial metadata.

    Args:
        data: Raw metadata dict (e.g. from JSON or extract_trial_metadata()).

    Returns:
        Validated WidefieldTrialMetadata model.

    Raises:
        ValidationError: If the dict does not conform to the schema.
    """
    return WidefieldTrialMetadata.model_validate(data)
