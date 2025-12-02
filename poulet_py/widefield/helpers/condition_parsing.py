"""
Functions for parsing condition information from file attributes and comments.
"""

import re
from typing import Any, Dict, Optional

def parse_comment(comment: str) -> Optional[Dict[str, Any]]:
    """
    Parse condition information from comment string.

    Expected format: "temp 32-31.0, 2 sec, 1 times"
    - "temp" indicates temperature protocol
    - "32-31.0" indicates rest_temp-target_temp
    - "2 sec" indicates duration
    - "1 times" indicates repetitions

    Args:
        comment: Comment string to parse.

    Returns:
        Dictionary with parsed values, or None if cannot parse.
    """
    parsed = {}

    temp_pattern = r'temp\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
    temp_match = re.search(temp_pattern, comment, re.IGNORECASE)
    if temp_match:
        parsed['baseline_temperature'] = float(temp_match.group(1))
        parsed['target_temperature'] = float(temp_match.group(2))

    duration_pattern = r'(\d+(?:\.\d+)?)\s*sec'
    duration_match = re.search(duration_pattern, comment, re.IGNORECASE)
    if duration_match:
        parsed['duration'] = float(duration_match.group(1))

    repetitions_pattern = r'(\d+)\s*times?'
    repetitions_match = re.search(
        repetitions_pattern, comment, re.IGNORECASE
    )
    if repetitions_match:
        parsed['repetitions'] = int(repetitions_match.group(1))

    return parsed if parsed else None


def get_condition_from_attributes(
    attributes: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract condition information from file attributes.

    Attempts to extract condition from multiple sources:
    1. Structured attributes (target_temperature, baseline_temperature, etc.)
    2. Comment field parsing (format: "temp 32-31.0, 2 sec, 1 times")
    3. Protocol name

    Args:
        attributes: Dictionary of file attributes from H5 file.

    Returns:
        Dictionary with condition information, or None if cannot extract.
        Keys may include: 'protocol', 'baseline_temperature', 'target_temperature',
        'duration', 'repetitions', 'comment', 'parsed_from'.
    """
    condition = {}
    parsed_from = []

    protocol = attributes.get('protocol_name')
    if protocol:
        if isinstance(protocol, bytes):
            protocol = protocol.decode('utf-8')
        condition['protocol'] = protocol
        parsed_from.append('protocol_name')

    if 'target_temperature' in attributes:
        condition['target_temperature'] = float(
            attributes['target_temperature']
        )
        parsed_from.append('target_temperature')

    if 'baseline_temperature' in attributes:
        condition['baseline_temperature'] = float(
            attributes['baseline_temperature']
        )
        parsed_from.append('baseline_temperature')

    if 'duration' in attributes:
        condition['duration'] = float(attributes['duration'])
        parsed_from.append('duration')

    if 'stim_length' in attributes:
        condition['stim_length'] = float(attributes['stim_length'])
        parsed_from.append('stim_length')

    comment = attributes.get('comment', '')
    if isinstance(comment, bytes):
        comment = comment.decode('utf-8')

    if comment:
        condition['comment'] = comment

        if 'target_temperature' not in condition:
            parsed_comment = parse_comment(comment)
            if parsed_comment:
                condition.update(parsed_comment)
                parsed_from.append('comment_parsed')

    condition['parsed_from'] = parsed_from

    if not condition:
        return None

    return condition

