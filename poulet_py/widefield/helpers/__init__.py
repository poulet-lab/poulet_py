"""
Helper functions for lab-specific data parsing and analysis.

This module contains functions specific to how we parse and deal with
our own specific data formats and protocols.
"""

from .condition_parsing import get_condition_from_attributes, parse_comment
from .stimulus_detection import detect_stimulus_frames

__all__ = ["detect_stimulus_frames", "get_condition_from_attributes", "parse_comment"]
