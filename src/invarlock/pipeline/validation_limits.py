"""Small validation preflights and bounded diagnostics; no contract dependencies."""

from __future__ import annotations

from typing import Any

from jsonschema.exceptions import ValidationError


def check_record_counts(value: Any, name: str, *, max_records: int) -> None:
    """Reject excessive known schedule lists without encoding their contents.

    Missing or malformed containers remain the closed schema's responsibility.
    The caller supplies the current record ceiling; this adds no separate limit.
    """
    if not isinstance(value, dict):
        return
    if name == "run":
        targets = [("records", value.get("records"))]
    elif name == "case_set":
        targets = [("cases", value.get("cases"))]
    elif name == "evidence":
        targets = [
            (f"{side}.records", nested.get("records"))
            for side in ("baseline", "candidate")
            if isinstance(nested := value.get(side), dict)
        ]
    else:
        return
    for location, records in targets:
        if isinstance(records, list) and len(records) > max_records:
            raise ValueError(
                f"{location} is too long: {len(records)} records; maximum {max_records}"
            )


def _abbreviate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = " ... [truncated] ... "
    if limit <= len(marker):
        return text[:limit]
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def format_validation_error(
    error: ValidationError, name: str, *, max_chars: int = 2400
) -> str:
    """Retain location and both ends of a schema reason in bounded output.

    This bounds the returned diagnostic, not jsonschema's construction of its
    original message. Known excessive record arrays are handled before that.
    """
    if type(max_chars) is not int or max_chars < 80:
        raise ValueError("diagnostic size must be an integer of at least 80 characters")
    path_limit = max_chars // 3
    location = ""
    for part in error.absolute_path:
        separator = "." if location else ""
        location = _abbreviate(
            location + separator + _abbreviate(str(part), path_limit), path_limit
        )
    prefix = _abbreviate(f"invalid {name} {location}: ", max_chars // 2)
    return prefix + _abbreviate(error.message, max_chars - len(prefix))
