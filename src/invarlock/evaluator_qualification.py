"""Generic qualification of evaluator exports for the runtime-import boundary.

Evaluator-specific execution and native-output parsing stay outside the core.
This module authenticates a closed normalized export, its independent schedule,
the retained upstream output, and the exact runner/dependency identities chosen
by the profile owner. Deterministic per-record exports can become runtime-import
facts; aggregate, human, or model-judge outputs remain observation-only.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from jsonschema import Draft202012Validator

from invarlock.core.runtime_provider import RuntimeScoringRecord
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.public_contracts import (
    EVALUATOR_QUALIFICATION_EXPORT_FORMAT_VERSION,
    EVALUATOR_QUALIFICATION_PROFILE_FORMAT_VERSION,
    EVALUATOR_QUALIFICATION_RESULT_FORMAT_VERSION,
    EVALUATOR_QUALIFICATION_SCHEDULE_FORMAT_VERSION,
    load_evaluator_qualification_export_schema,
    load_evaluator_qualification_profile_schema,
    load_evaluator_qualification_result_schema,
    load_evaluator_qualification_schedule_schema,
)

EVALUATOR_PROFILE_FORMAT = EVALUATOR_QUALIFICATION_PROFILE_FORMAT_VERSION
EVALUATOR_SCHEDULE_FORMAT = EVALUATOR_QUALIFICATION_SCHEDULE_FORMAT_VERSION
EVALUATOR_EXPORT_FORMAT = EVALUATOR_QUALIFICATION_EXPORT_FORMAT_VERSION
EVALUATOR_QUALIFICATION_FORMAT = EVALUATOR_QUALIFICATION_RESULT_FORMAT_VERSION

MAX_QUALIFICATION_JSON_BYTES = 16 * 1024 * 1024
MAX_RAW_UPSTREAM_OUTPUT_BYTES = 64 * 1024 * 1024

QualificationOutcome = Literal["qualified_for_import", "observation_only"]
QualificationAuthority = Literal["verdict_authority", "observation_only"]


class EvaluatorQualificationError(ValueError):
    """Raised when an evaluator export cannot cross the qualification boundary."""


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _schema_error(
    value: Mapping[str, object],
    *,
    schema: Mapping[str, object],
) -> str | None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if not errors:
        return None
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path)
    return f"{path}: {first.message}" if path else first.message


def _load_closed_json(
    path: Path,
    *,
    label: str,
    schema: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    try:
        raw = read_regular_file_bytes(
            path,
            label=label,
            max_bytes=MAX_QUALIFICATION_JSON_BYTES,
        )
        value = parse_json_bytes(raw, label=label)
    except StrictJsonError as exc:
        raise EvaluatorQualificationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EvaluatorQualificationError(f"{label} must be a JSON object")
    if raw != canonical_json_bytes(value):
        raise EvaluatorQualificationError(f"{label} must use canonical JSON")
    error = _schema_error(value, schema=schema)
    if error is not None:
        raise EvaluatorQualificationError(f"{label} is invalid: {error}")
    return cast(dict[str, object], value), raw


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvaluatorQualificationError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _objects(value: object, *, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise EvaluatorQualificationError(f"{field} must be an array of objects")
    return cast(list[dict[str, object]], value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluatorQualificationError(f"{field} must be a string")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorQualificationError(f"{field} must be a number")
    return float(value)


def _bare_digest(value: str) -> str:
    return value.removeprefix("sha256:")


def _require_binding(
    bindings: Mapping[str, object],
    name: str,
    expected: str,
    *,
    mismatch: str | None = None,
) -> None:
    observed = bindings.get(name)
    if observed != expected:
        raise EvaluatorQualificationError(
            mismatch or f"{name.replace('_', ' ')} does not match"
        )


def _runtime_records_digest(records: Sequence[RuntimeScoringRecord]) -> str:
    payload = [
        {
            "error_code": record.error_code,
            "input_sha256": record.input_sha256,
            "logprob_sum": record.logprob_sum,
            "output_sha256": record.output_sha256,
            "output_text": record.output_text,
            "record_id": record.record_id,
            "status": record.status,
            "token_count": record.token_count,
            "utf8_byte_count": record.utf8_byte_count,
        }
        for record in records
    ]
    return _sha256(canonical_json_bytes(payload, newline=False))


@dataclass(frozen=True)
class EvaluatorQualificationResult:
    """Digest-bound result of one generic evaluator export qualification."""

    profile_id: str
    outcome: QualificationOutcome
    authority: QualificationAuthority
    reason_codes: tuple[str, ...]
    record_count: int
    scores: tuple[float, ...]
    mean_score: float | None
    records_sha256: str | None
    profile_sha256: str
    schedule_sha256: str
    export_sha256: str
    raw_output_sha256: str
    runner_sha256: str
    dependency_lock_sha256: str
    _records: tuple[RuntimeScoringRecord, ...]
    format: str = EVALUATOR_QUALIFICATION_FORMAT

    def runtime_records(self) -> tuple[RuntimeScoringRecord, ...]:
        """Return qualified import facts; observation-only results return none."""

        if self.authority != "verdict_authority":
            return ()
        return self._records

    def as_dict(self) -> dict[str, object]:
        """Return the closed public result without internal runtime objects."""

        return {
            "authority": self.authority,
            "bindings": {
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "export_sha256": self.export_sha256,
                "profile_sha256": self.profile_sha256,
                "raw_output_sha256": self.raw_output_sha256,
                "runner_sha256": self.runner_sha256,
                "schedule_sha256": self.schedule_sha256,
            },
            "format": self.format,
            "mean_score": self.mean_score,
            "outcome": self.outcome,
            "profile_id": self.profile_id,
            "reason_codes": list(self.reason_codes),
            "record_count": self.record_count,
            "records_sha256": self.records_sha256,
            "scores": list(self.scores),
        }

    def as_json(self) -> str:
        """Return canonical JSON suitable for retention beside upstream output."""

        return canonical_json_bytes(self.as_dict()).decode("utf-8")

    def write(self, destination: str | Path) -> Path:
        """Atomically publish a new result without replacing an existing file."""

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self.as_json().encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            temporary = None
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise EvaluatorQualificationError(
                    f"qualification result already exists: {path}"
                ) from exc
            raise EvaluatorQualificationError(
                f"could not write qualification result: {exc}"
            ) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        return path


def _validate_common_bindings(
    *,
    profile: Mapping[str, object],
    profile_raw: bytes,
    schedule_raw: bytes,
    export: Mapping[str, object],
    raw_output: bytes,
) -> tuple[str, str]:
    profile_id = _string(profile["profile_id"], field="profile.profile_id")
    if export.get("profile_id") != profile_id:
        raise EvaluatorQualificationError("export profile_id does not match profile")
    profile_upstream = _object(profile["upstream"], field="profile.upstream")
    package = _object(profile_upstream["package"], field="profile.upstream.package")
    if export.get("upstream") != package:
        raise EvaluatorQualificationError(
            "upstream package identity does not match profile"
        )
    execution = _object(profile["execution"], field="profile.execution")
    runner_sha256 = _string(
        execution["runner_sha256"],
        field="profile.execution.runner_sha256",
    )
    dependency_lock_sha256 = _string(
        execution["dependency_lock_sha256"],
        field="profile.execution.dependency_lock_sha256",
    )
    bindings = _object(export["bindings"], field="export.bindings")
    _require_binding(bindings, "profile_sha256", _sha256(profile_raw))
    _require_binding(bindings, "schedule_sha256", _sha256(schedule_raw))
    _require_binding(
        bindings,
        "raw_output_sha256",
        _sha256(raw_output),
        mismatch="raw upstream output digest does not match export",
    )
    _require_binding(bindings, "runner_sha256", runner_sha256)
    _require_binding(
        bindings,
        "dependency_lock_sha256",
        dependency_lock_sha256,
    )
    return runner_sha256, dependency_lock_sha256


def _observation_only_result(
    *,
    profile: Mapping[str, object],
    profile_raw: bytes,
    schedule_raw: bytes,
    export: Mapping[str, object],
    export_raw: bytes,
    raw_output: bytes,
    runner_sha256: str,
    dependency_lock_sha256: str,
) -> EvaluatorQualificationResult:
    authority = _object(profile["authority"], field="profile.authority")
    reason = _string(authority["reason"], field="profile.authority.reason")
    summary = export.get("summary")
    if not isinstance(summary, dict):
        raise EvaluatorQualificationError(
            "observation-only export must bind one upstream summary"
        )
    summary_object = cast(dict[str, object], summary)
    if summary_object.get("sha256") != _sha256(raw_output):
        raise EvaluatorQualificationError(
            "observation-only summary digest does not match raw upstream output"
        )
    return EvaluatorQualificationResult(
        profile_id=_string(profile["profile_id"], field="profile.profile_id"),
        outcome="observation_only",
        authority="observation_only",
        reason_codes=(reason,),
        record_count=0,
        scores=(),
        mean_score=None,
        records_sha256=None,
        profile_sha256=_sha256(profile_raw),
        schedule_sha256=_sha256(schedule_raw),
        export_sha256=_sha256(export_raw),
        raw_output_sha256=_sha256(raw_output),
        runner_sha256=runner_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        _records=(),
    )


def _deterministic_result(
    *,
    profile: Mapping[str, object],
    profile_raw: bytes,
    schedule: Mapping[str, object],
    schedule_raw: bytes,
    export: Mapping[str, object],
    export_raw: bytes,
    raw_output: bytes,
    runner_sha256: str,
    dependency_lock_sha256: str,
) -> EvaluatorQualificationResult:
    if export.get("summary") is not None:
        raise EvaluatorQualificationError(
            "deterministic export must not substitute an aggregate summary"
        )
    expected = _objects(schedule["records"], field="schedule.records")
    observed = _objects(export["records"], field="export.records")
    expected_identities = tuple(
        (record["record_id"], record["input_sha256"]) for record in expected
    )
    observed_identities = tuple(
        (record["record_id"], record["input_sha256"]) for record in observed
    )
    if observed_identities != expected_identities:
        raise EvaluatorQualificationError(
            "export record order and input identities must exactly match schedule"
        )
    runtime_records: list[RuntimeScoringRecord] = []
    scores: list[float] = []
    for expected_record, observed_record in zip(expected, observed, strict=True):
        record_id = _string(observed_record["record_id"], field="record.record_id")
        if observed_record["status"] != "ok":
            raise EvaluatorQualificationError(
                f"export record {record_id!r} is not successful"
            )
        output_text = _string(
            observed_record["output_text"],
            field=f"record {record_id!r} output_text",
        )
        output_sha256 = _string(
            observed_record["output_sha256"],
            field=f"record {record_id!r} output_sha256",
        )
        if _sha256(output_text.encode("utf-8")) != output_sha256:
            raise EvaluatorQualificationError(
                f"export record {record_id!r} output digest is invalid"
            )
        expected_score = (
            1.0 if output_sha256 == expected_record["reference_output_sha256"] else 0.0
        )
        reported_score = _number(
            observed_record["reported_score"],
            field=f"record {record_id!r} reported_score",
        )
        if reported_score != expected_score:
            raise EvaluatorQualificationError(
                f"export record {record_id!r} reported score does not match "
                "independent exact-match replay"
            )
        input_sha256 = _string(
            observed_record["input_sha256"],
            field=f"record {record_id!r} input_sha256",
        )
        runtime_records.append(
            RuntimeScoringRecord(
                record_id=record_id,
                input_sha256=_bare_digest(input_sha256),
                status="ok",
                output_text=output_text,
                output_sha256=_bare_digest(output_sha256),
            )
        )
        scores.append(expected_score)
    runtime_record_tuple = tuple(runtime_records)
    result = EvaluatorQualificationResult(
        profile_id=_string(profile["profile_id"], field="profile.profile_id"),
        outcome="qualified_for_import",
        authority="verdict_authority",
        reason_codes=(),
        record_count=len(runtime_record_tuple),
        scores=tuple(scores),
        mean_score=sum(scores) / len(scores),
        records_sha256=_runtime_records_digest(runtime_record_tuple),
        profile_sha256=_sha256(profile_raw),
        schedule_sha256=_sha256(schedule_raw),
        export_sha256=_sha256(export_raw),
        raw_output_sha256=_sha256(raw_output),
        runner_sha256=runner_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        _records=runtime_record_tuple,
    )
    error = _schema_error(
        result.as_dict(),
        schema=load_evaluator_qualification_result_schema(),
    )
    if error is not None:  # pragma: no cover - internal construction invariant
        raise EvaluatorQualificationError(f"qualification result is invalid: {error}")
    return result


def qualify_evaluator_export(
    *,
    profile_path: str | Path,
    schedule_path: str | Path,
    export_path: str | Path,
    raw_output_path: str | Path,
) -> EvaluatorQualificationResult:
    """Qualify one normalized evaluator export without evaluator-specific code."""

    profile, profile_raw = _load_closed_json(
        Path(profile_path),
        label="evaluator qualification profile",
        schema=load_evaluator_qualification_profile_schema(),
    )
    schedule, schedule_raw = _load_closed_json(
        Path(schedule_path),
        label="evaluator qualification schedule",
        schema=load_evaluator_qualification_schedule_schema(),
    )
    export, export_raw = _load_closed_json(
        Path(export_path),
        label="evaluator qualification export",
        schema=load_evaluator_qualification_export_schema(),
    )
    try:
        raw_output = read_regular_file_bytes(
            Path(raw_output_path),
            label="raw upstream output",
            max_bytes=MAX_RAW_UPSTREAM_OUTPUT_BYTES,
        )
    except StrictJsonError as exc:
        raise EvaluatorQualificationError(str(exc)) from exc
    runner_sha256, dependency_lock_sha256 = _validate_common_bindings(
        profile=profile,
        profile_raw=profile_raw,
        schedule_raw=schedule_raw,
        export=export,
        raw_output=raw_output,
    )
    authority = _object(profile["authority"], field="profile.authority")
    if authority["mode"] == "observation_only":
        result = _observation_only_result(
            profile=profile,
            profile_raw=profile_raw,
            schedule_raw=schedule_raw,
            export=export,
            export_raw=export_raw,
            raw_output=raw_output,
            runner_sha256=runner_sha256,
            dependency_lock_sha256=dependency_lock_sha256,
        )
        error = _schema_error(
            result.as_dict(),
            schema=load_evaluator_qualification_result_schema(),
        )
        if error is not None:  # pragma: no cover - internal construction invariant
            raise EvaluatorQualificationError(
                f"qualification result is invalid: {error}"
            )
        return result
    return _deterministic_result(
        profile=profile,
        profile_raw=profile_raw,
        schedule=schedule,
        schedule_raw=schedule_raw,
        export=export,
        export_raw=export_raw,
        raw_output=raw_output,
        runner_sha256=runner_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )


__all__ = [
    "EVALUATOR_EXPORT_FORMAT",
    "EVALUATOR_PROFILE_FORMAT",
    "EVALUATOR_QUALIFICATION_FORMAT",
    "EVALUATOR_SCHEDULE_FORMAT",
    "EvaluatorQualificationError",
    "EvaluatorQualificationResult",
    "qualify_evaluator_export",
]
