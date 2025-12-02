"""
Helper functions for lab-specific data parsing and analysis.

This module contains functions specific to how we parse and deal with
our own specific data formats and protocols.
"""

from .condition_parsing import parse_comment, get_condition_from_attributes
from .stimulus_detection import detect_stimulus_frames

__all__ = [
    'parse_comment',
    'get_condition_from_attributes',
    'detect_stimulus_frames'
]

