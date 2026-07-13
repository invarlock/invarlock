"""Portable identity fields for hosted evaluation datasets."""

from __future__ import annotations

import re
from typing import Any

HOSTED_DATASET_PROVIDERS = frozenset({"hf_text", "hf_seq2seq", "wikitext2"})
DATASET_IDENTITY_FIELDS = (
    "provider",
    "dataset_name",
    "config_name",
    "revision",
    "split",
)
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def canonical_dataset_revision(value: object) -> str | None:
    """Return an exact immutable hosted-dataset revision, if valid.

    Whitespace, symbolic refs such as ``main``, abbreviated hashes, and
    uppercase digests are rejected. Reports are attestations, so accepting a
    normalized value that differs from the recorded value would be ambiguous.
    """

    if not isinstance(value, str) or value != value.strip():
        return None
    return value if _IMMUTABLE_REVISION_RE.fullmatch(value) else None


def is_hosted_dataset_provider(value: object) -> bool:
    """Return whether a provider kind resolves data from a hosted repository."""

    provider = _optional_text(value)
    return bool(provider and provider.lower() in HOSTED_DATASET_PROVIDERS)


def dataset_identity_from_provider(data_provider: Any) -> dict[str, str]:
    """Extract report-safe provider coordinates from a resolved provider."""

    identity: dict[str, str] = {}
    for output_key, attribute in (
        ("provider", "name"),
        ("dataset_name", "dataset_name"),
        ("config_name", "config_name"),
        ("revision", "revision"),
    ):
        try:
            raw_value = getattr(data_provider, attribute, None)
            value = (
                raw_value
                if output_key == "revision" and isinstance(raw_value, str) and raw_value
                else _optional_text(raw_value)
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            value = None
        if value is not None:
            identity[output_key] = value
    return identity


def dataset_identity_from_report(report: Any) -> dict[str, str | None]:
    """Extract exact independently authorizable dataset coordinates from a report."""

    dataset = report.get("dataset") if isinstance(report, dict) else None
    if not isinstance(dataset, dict):
        dataset = {}
    identity: dict[str, str | None] = {}
    for field in DATASET_IDENTITY_FIELDS:
        value = dataset.get(field)
        identity[field] = (
            value
            if isinstance(value, str) and value and value == value.strip()
            else None
        )
    return identity


__all__ = [
    "HOSTED_DATASET_PROVIDERS",
    "DATASET_IDENTITY_FIELDS",
    "canonical_dataset_revision",
    "dataset_identity_from_provider",
    "dataset_identity_from_report",
    "is_hosted_dataset_provider",
]
