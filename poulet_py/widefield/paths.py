"""
Path resolution functions for widefield imaging data.

This module provides functions for resolving output paths
based on the standard directory structure used for storing
raw and processed widefield imaging data.

Expected directory structure:
    data/
    ├── raw/
    │   └── session_name/
    │       └── trial_folder/
    │           ├── recording.tiff
    │           ├── recording.csv
    │           ├── data.h5
    │           └── green.tiff
    └── processed/
        └── session_name/
            ├── mask.json  (session-level)
            └── trials/
                └── trial_folder/
                    └── (processed outputs)
"""

from pathlib import Path

from poulet_py import LOGGER


def get_session_processed_folder(trial_path: Path) -> Path | None:
    """
    Get the session-level processed folder for a trial.

    Resolves the path to the processed folder at the session level,
    which is used for storing files shared across all trials in
    a session (e.g., masks).

    Args:
        trial_path: Path to a trial folder within the raw data
            directory structure.

    Returns:
        Path to the session processed folder. The directory is
        created if it doesn't exist. Returns None on error.

    Example:
        Input:  /data/raw/session_001/trial_001/
        Output: /data/processed/session_001/
    """
    try:
        trial_path = Path(trial_path)
        parts = trial_path.parts

        raw_idx = None
        for i, part in enumerate(parts):
            if part == "raw":
                raw_idx = i
                break

        if raw_idx is None:
            LOGGER.error("Could not find 'raw' in trial path structure")
            return None

        session_folder = parts[raw_idx + 1]

        data_folder = trial_path
        for _ in range(len(parts) - raw_idx - 1):
            data_folder = data_folder.parent

        session_processed_folder = data_folder.parent / "processed" / session_folder

        session_processed_folder.mkdir(parents=True, exist_ok=True)
        return session_processed_folder

    except Exception:
        LOGGER.exception("Error creating session processed folder")
        return None


def get_trial_processed_folder(trial_path: Path) -> Path | None:
    """
    Get the trial-level processed folder for a trial.

    Resolves the path to the processed folder for a specific trial,
    which is used for storing trial-specific output files.

    Args:
        trial_path: Path to a trial folder within the raw data
            directory structure.

    Returns:
        Path to the trial processed folder. The directory is
        created if it doesn't exist. Returns None on error.

    Example:
        Input:  /data/raw/session_001/trial_001/
        Output: /data/processed/session_001/trials/trial_001/
    """
    try:
        trial_path = Path(trial_path)
        parts = trial_path.parts

        raw_idx = None
        for i, part in enumerate(parts):
            if part == "raw":
                raw_idx = i
                break

        if raw_idx is None:
            LOGGER.error("Could not find 'raw' in trial path structure")
            return None

        session_folder = parts[raw_idx + 1]
        trial_folder = parts[-1]

        data_folder = trial_path
        for _ in range(len(parts) - raw_idx - 1):
            data_folder = data_folder.parent

        processed_folder = (
            data_folder.parent / "processed" / session_folder / "trials" / trial_folder
        )

        processed_folder.mkdir(parents=True, exist_ok=True)
        return processed_folder

    except Exception:
        LOGGER.exception("Error creating processed folder")
        return None
