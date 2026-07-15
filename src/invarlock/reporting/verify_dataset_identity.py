"""Strict hosted-dataset identity verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.core.dataset_identity import (
    canonical_dataset_revision,
    is_hosted_dataset_provider,
)


def _nonempty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _declared_revision(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _resolve(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _aliased_text(
    errors: list[str],
    *,
    payload: dict[str, Any],
    paths: tuple[str, ...],
    source: str,
    label: str,
) -> str | None:
    resolved = [
        (path, value)
        for path in paths
        if (value := _nonempty_text(_resolve(payload, path))) is not None
    ]
    if not resolved:
        return None
    first_path, first_value = resolved[0]
    for path, value in resolved[1:]:
        if value != first_value:
            errors.append(
                f"Strict {source} {label} fork: {first_path} differs from {path}."
            )
    return first_value


@dataclass(frozen=True)
class _DatasetIdentity:
    provider: str | None
    dataset_name: str | None
    config_name: str | None
    canonical_config_name: str | None
    revision: str | None


def _subject_identity(errors: list[str], subject: dict[str, Any]) -> _DatasetIdentity:
    return _DatasetIdentity(
        provider=_nonempty_text(_resolve(subject, "dataset.provider")),
        dataset_name=_nonempty_text(_resolve(subject, "dataset.dataset_name")),
        config_name=_aliased_text(
            errors,
            payload=subject,
            paths=("dataset.config_name", "dataset.config"),
            source="subject dataset",
            label="configuration",
        ),
        canonical_config_name=_nonempty_text(_resolve(subject, "dataset.config_name")),
        revision=_declared_revision(_resolve(subject, "dataset.revision")),
    )


def _baseline_identity(errors: list[str], baseline: dict[str, Any]) -> _DatasetIdentity:
    return _DatasetIdentity(
        provider=_aliased_text(
            errors,
            payload=baseline,
            paths=("data.provider", "data.dataset"),
            source="baseline dataset",
            label="provider",
        ),
        dataset_name=_nonempty_text(_resolve(baseline, "data.dataset_name")),
        config_name=_aliased_text(
            errors,
            payload=baseline,
            paths=("data.config_name", "data.config"),
            source="baseline dataset",
            label="configuration",
        ),
        canonical_config_name=_nonempty_text(_resolve(baseline, "data.config_name")),
        revision=_declared_revision(_resolve(baseline, "data.revision")),
    )


def _append_optional_parity(
    errors: list[str],
    *,
    subject_value: str | None,
    baseline_value: str | None,
    label: str,
    subject_path: str,
) -> None:
    if subject_value is None and baseline_value is None:
        return
    if subject_value is None or baseline_value is None:
        errors.append(
            f"Strict baseline {label} parity requires {subject_path} on both reports."
        )
    elif subject_value != baseline_value:
        errors.append(
            f"Strict baseline {label} mismatch: report={subject_value!r} "
            f"supplied_baseline={baseline_value!r}."
        )


def _append_hosted_requirements(
    errors: list[str],
    *,
    identity: _DatasetIdentity,
    side: str,
    name_path: str,
    config_path: str,
) -> str | None:
    if identity.dataset_name is None:
        errors.append(
            f"Strict {side} hosted dataset name requires non-empty {name_path}."
        )
    if identity.canonical_config_name is None:
        errors.append(
            f"Strict {side} hosted dataset configuration requires non-empty "
            f"{config_path}."
        )
    revision = canonical_dataset_revision(identity.revision)
    if revision is None:
        errors.append(
            f"Strict {side} hosted dataset revision must be 40-64 lowercase "
            "hexadecimal characters with no surrounding whitespace."
        )
    return revision


def append_strict_dataset_identity_errors(
    errors: list[str],
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Require exact hosted dataset coordinates and arm-to-arm parity."""

    subject_identity = _subject_identity(errors, subject)
    baseline_identity = _baseline_identity(errors, baseline)

    if (
        subject_identity.provider is not None
        and baseline_identity.provider is not None
        and subject_identity.provider != baseline_identity.provider
    ):
        errors.append(
            "Strict baseline dataset provider mismatch: "
            f"report={subject_identity.provider!r} "
            f"supplied_baseline={baseline_identity.provider!r}."
        )

    hosted = is_hosted_dataset_provider(
        subject_identity.provider
    ) or is_hosted_dataset_provider(baseline_identity.provider)
    if hosted:
        subject_revision = _append_hosted_requirements(
            errors,
            identity=subject_identity,
            side="subject",
            name_path="report.dataset.dataset_name",
            config_path="report.dataset.config_name",
        )
        baseline_revision = _append_hosted_requirements(
            errors,
            identity=baseline_identity,
            side="baseline",
            name_path="supplied_baseline.data.dataset_name",
            config_path="supplied_baseline.data.config_name",
        )
        if (
            subject_revision is not None
            and baseline_revision is not None
            and subject_revision != baseline_revision
        ):
            errors.append(
                "Strict baseline dataset revision mismatch: "
                f"report={subject_revision!r} "
                f"supplied_baseline={baseline_revision!r}."
            )
    else:
        _append_optional_parity(
            errors,
            subject_value=subject_identity.revision,
            baseline_value=baseline_identity.revision,
            label="dataset revision",
            subject_path="dataset.revision",
        )

    _append_optional_parity(
        errors,
        subject_value=subject_identity.dataset_name,
        baseline_value=baseline_identity.dataset_name,
        label="dataset name",
        subject_path="dataset.dataset_name",
    )
    _append_optional_parity(
        errors,
        subject_value=subject_identity.config_name,
        baseline_value=baseline_identity.config_name,
        label="dataset configuration",
        subject_path="dataset.config_name",
    )


__all__ = ["append_strict_dataset_identity_errors"]
