"""Small, versioned deterministic scorers shared by import and verification."""

from __future__ import annotations

import unicodedata
from collections import Counter
from decimal import Context, Decimal, InvalidOperation, localcontext
from typing import Any

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes

KINDS = (
    "exact_match",
    "normalized_match",
    "numeric_tolerance",
    "json_fields",
    "token_f1",
)
SCORER_VERSION = "1.0.0"
UNICODE_VERSION = unicodedata.unidata_version


class MetricError(ValueError):
    """Invalid scoring configuration or unusable reference material."""


def validate_configuration(kind: str, configuration: dict[str, Any]) -> None:
    allowed = {
        "exact_match": set(),
        "normalized_match": {"casefold", "unicode_version"},
        "token_f1": {"casefold", "unicode_version"},
        "numeric_tolerance": {"absolute", "relative"},
        "json_fields": {"fields"},
    }
    if kind not in allowed or set(configuration) - allowed[kind]:
        raise MetricError(f"unsupported configuration for {kind!r}")
    if kind in ("normalized_match", "token_f1"):
        if "unicode_version" not in configuration:
            raise MetricError(f"{kind} requires an explicit unicode_version")
        if configuration["unicode_version"] != unicodedata.unidata_version:
            raise MetricError(
                f"{kind} requires Unicode {configuration['unicode_version']!r}; "
                f"this runtime provides {unicodedata.unidata_version}. "
                "Use a runtime with the policy's Unicode version."
            )
    if "casefold" in configuration and not isinstance(configuration["casefold"], bool):
        raise MetricError("casefold must be a boolean")
    for key in ("absolute", "relative"):
        if key in configuration:
            value = configuration[key]
            try:
                valid = isinstance(value, (int, float)) and _number(value) >= 0
            except (ValueError, OverflowError):
                valid = False
            if not valid:
                raise MetricError(
                    f"{key} tolerance must be a finite nonnegative number"
                )
    if kind == "json_fields":
        fields = configuration.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or len(fields) > 100
            or any(not isinstance(p, str) or not p.startswith("/") for p in fields)
        ):
            raise MetricError("json_fields requires 1–100 JSON pointer fields")
        if len(set(fields)) != len(fields):
            raise MetricError("JSON pointer fields must be unique")
        for pointer in fields:
            for component in pointer.split("/")[1:]:
                if "~" in component.replace("~0", "").replace("~1", ""):
                    raise MetricError("invalid JSON pointer escape")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise MetricError("text scorer requires string expected and output values")
    return value


def _normalize(value: str, configuration: dict[str, Any]) -> str:
    result = unicodedata.normalize("NFKC", value)
    if configuration.get("casefold", True):
        result = result.casefold()
    return " ".join(result.split())


def _number(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("not a numeric value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("not a decimal number") from exc
    if (
        not result.is_finite()
        or len(result.as_tuple().digits) > 256
        or abs(result.adjusted()) > 1000
    ):
        raise ValueError(
            "numeric values require at most 256 digits and an adjusted exponent within ±1000"
        )
    return result


def _json(value: Any) -> Any:
    return (
        parse_json_bytes(value.encode(), label="structured output")
        if isinstance(value, str)
        else value
    )


def _pointer(value: Any, pointer: str) -> Any:
    for component in pointer.split("/")[1:]:
        key = component.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and key.isdecimal() and str(int(key)) == key:
            value = value[int(key)]
        else:
            raise KeyError(pointer)
    return value


def score(
    kind: str, expected: Any, output: Any, configuration: dict[str, Any] | None = None
) -> float:
    """Return one bounded score; invalid references fail, invalid answers score zero."""
    configuration = configuration or {}
    validate_configuration(kind, configuration)
    if kind == "numeric_tolerance":
        try:
            numeric_target = _number(expected)
        except (ValueError, OverflowError) as exc:
            raise MetricError("numeric reference must be finite") from exc
        try:
            numeric_output = _number(output)
        except (ValueError, OverflowError):
            return 0.0
        # Exact arithmetic across the bounded exponents and significands avoids
        # collapsing distinct large integers or decimal answers during replay.
        with localcontext(Context(prec=4096)):
            return float(
                abs(numeric_output - numeric_target)
                <= max(
                    _number(configuration.get("absolute", 0)),
                    _number(configuration.get("relative", 0)) * abs(numeric_target),
                )
            )
    if kind == "json_fields":
        try:
            reference = _json(expected)
            targets = [_pointer(reference, p) for p in configuration["fields"]]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MetricError(
                "reference must contain every configured JSON field"
            ) from exc
        try:
            structured_output = _json(output)
        except (ValueError, TypeError):
            return 0.0
        matches = 0
        for pointer, field_target in zip(configuration["fields"], targets, strict=True):
            try:
                matches += canonical_json_bytes(
                    _pointer(structured_output, pointer)
                ) == canonical_json_bytes(field_target)
            except (KeyError, IndexError, TypeError):
                pass
        return matches / len(targets)
    target = _text(expected)
    if not isinstance(output, str):
        return 0.0
    if kind == "exact_match":
        return float(target == output)
    target, output = (
        _normalize(target, configuration),
        _normalize(output, configuration),
    )
    if kind == "normalized_match":
        return float(target == output)
    wanted, observed = Counter(target.split()), Counter(output.split())
    denominator = sum(wanted.values()) + sum(observed.values())
    return 2 * sum((wanted & observed).values()) / denominator if denominator else 1.0
