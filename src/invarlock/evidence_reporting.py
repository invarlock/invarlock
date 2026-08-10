"""One human renderer for the canonical report inside an evidence pack."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from invarlock import evidence_pack_integrity as integrity
from invarlock.core.scorer_extension import (
    ScorerExtensionError,
    decode_scorer_binding,
)
from invarlock.evidence_pack_contract import (
    COMPARISON_REPORT_FORMAT,
    LEGACY_COMPARISON_REPORT_FORMAT,
    EvidencePackError,
    canonical_json_bytes,
    evidence_observation_errors,
    validated_derived_measurements,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.evidence_pack_snapshot import PackSnapshot
from invarlock.paired_exact_match import (
    PAIRED_CONFIDENCE_INTERVAL_METHODS,
    PairedExactMatchError,
    paired_exact_match_statistics,
)
from invarlock.public_contracts import load_evidence_pack_schema

_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class EvidenceReportError(ValueError):
    """Raised when canonical evidence cannot be rendered safely."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class EvidenceReport:
    text: str
    html_path: Path | None
    evidence_signer: str
    pack_manifest_digest: str
    observations: tuple[dict[str, Any], ...] = ()


def _load_object_with_bytes(
    path: Path, *, label: str, max_bytes: int
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_regular_file_bytes(path, label=label, max_bytes=max_bytes)
        payload = parse_json_bytes(raw, label=label)
    except StrictJsonError as exc:
        raise EvidenceReportError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise EvidenceReportError(f"{label} must be a JSON object")
    return cast(dict[str, Any], payload), raw


def _manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    errors = sorted(
        Draft202012Validator(load_evidence_pack_schema()).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [
        "evidence manifest schema failed at "
        + (".".join(str(part) for part in error.absolute_path) or "<root>")
        + f": {error.message}"
        for error in errors
    ]


def _manifest_payload_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for block_name in ("inputs", "evidence"):
        block = manifest.get(block_name)
        if not isinstance(block, dict):
            continue
        for reference in block.values():
            if isinstance(reference, dict) and isinstance(reference.get("path"), str):
                paths.add(reference["path"])
    paired = manifest.get("paired_records")
    if isinstance(paired, dict) and isinstance(paired.get("path"), str):
        paths.add(paired["path"])
    observations = manifest.get("observations")
    if isinstance(observations, dict):
        for reference in observations.values():
            if isinstance(reference, dict) and isinstance(reference.get("path"), str):
                paths.add(reference["path"])
    return paths


def _load_observations(
    evidence: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    references = manifest.get("observations")
    request_reference = manifest.get("evidence")
    request_entry = (
        request_reference.get("request")
        if isinstance(request_reference, dict)
        else None
    )
    request_path = (
        request_entry.get("path") if isinstance(request_entry, dict) else None
    )
    if not isinstance(request_path, str):
        return [], ["manifest request reference is invalid"]
    safe_request_path = integrity._normalize_pack_path(evidence, request_path)
    if safe_request_path is None:
        return [], ["manifest request path is unsafe"]
    try:
        request_raw = read_regular_file_bytes(
            safe_request_path,
            label="normalized request",
            max_bytes=_MAX_REPORT_BYTES,
        )
        request = parse_json_bytes(request_raw, label="normalized request")
    except StrictJsonError as exc:
        return [], [str(exc)]
    if not isinstance(request, dict) or canonical_json_bytes(request) != request_raw:
        return [], ["normalized request is not canonical JSON"]
    requested_items = request.get("observations", [])
    if not isinstance(requested_items, list):
        return [], ["normalized request observations are invalid"]
    requested: dict[str, dict[str, Any]] = {}
    for item in requested_items:
        observation_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(observation_id, str) or observation_id in requested:
            return [], ["normalized request observation entry is invalid"]
        requested[observation_id] = item
    if references is None:
        return (
            ([], [])
            if not requested
            else ([], ["normalized request observations are missing from manifest"])
        )
    if not isinstance(references, dict):
        return [], ["manifest observations are invalid"]
    if set(references) != set(requested):
        return [], ["manifest observations do not match normalized request"]
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        return [], ["manifest inputs are invalid"]

    def material_digest(role: str) -> str | None:
        reference = inputs.get(role)
        value = (
            reference.get("material_digest") if isinstance(reference, dict) else None
        )
        return value if isinstance(value, str) else None

    schedule_digest = material_digest("dataset")
    policy_digest = material_digest("policy")
    artifacts = {
        side: digest
        for side in ("baseline", "subject")
        if isinstance((digest := material_digest(side)), str)
    }
    comparison_id = manifest.get("comparison_id")
    if (
        not isinstance(comparison_id, str)
        or not isinstance(schedule_digest, str)
        or not isinstance(policy_digest, str)
        or len(artifacts) != 2
    ):
        return [], ["manifest observation bindings are unavailable"]
    loaded: list[dict[str, Any]] = []
    errors: list[str] = []
    for observation_id, reference in sorted(references.items()):
        if not isinstance(observation_id, str) or not isinstance(reference, dict):
            errors.append("manifest observation entry is invalid")
            continue
        relative = reference.get("path")
        if not isinstance(relative, str):
            errors.append(f"observation {observation_id!r} path is invalid")
            continue
        safe_path = integrity._normalize_pack_path(evidence, relative)
        if safe_path is None:
            errors.append(f"observation {observation_id!r} path is unsafe")
            continue
        try:
            raw = read_regular_file_bytes(
                safe_path,
                label=f"observation {observation_id}",
                max_bytes=_MAX_REPORT_BYTES,
            )
            payload = parse_json_bytes(raw, label=f"observation {observation_id}")
        except StrictJsonError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(payload, dict):
            errors.append(f"observation {observation_id!r} must be a JSON object")
            continue
        local_errors: list[str] = []
        if canonical_json_bytes(payload) != raw:
            local_errors.append(
                f"observation {observation_id!r} must use canonical JSON"
            )
        digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if digest != reference.get("digest"):
            local_errors.append(
                f"manifest digest does not bind observation {observation_id!r}"
            )
        observation_payload = payload.get("payload")
        payload_digest = (
            f"sha256:{hashlib.sha256(canonical_json_bytes(observation_payload)).hexdigest()}"
            if isinstance(observation_payload, dict)
            else None
        )
        expected_descriptor = {
            "id": observation_id,
            "kind": payload.get("kind"),
            "scope": payload.get("scope"),
            "payload_digest": payload_digest,
        }
        if requested[observation_id] != expected_descriptor:
            local_errors.append(
                f"observation {observation_id!r} does not match normalized request"
            )
        local_errors.extend(
            evidence_observation_errors(
                payload,
                observation_id=observation_id,
                reference=reference,
                comparison_id=comparison_id,
                schedule_digest=schedule_digest,
                policy_digest=policy_digest,
                artifact_digests=artifacts,
            )
        )
        errors.extend(local_errors)
        if not local_errors:
            loaded.append(payload)
    return loaded, errors


def _reference_binding_errors(evidence: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for block_name in ("inputs", "evidence"):
        block = manifest.get(block_name)
        if not isinstance(block, dict):
            continue
        for role, reference in block.items():
            if not isinstance(reference, dict):
                continue
            relative = reference.get("path")
            if not isinstance(relative, str):
                continue
            try:
                raw = read_regular_file_bytes(
                    evidence / relative,
                    label=f"{block_name}.{role}",
                    max_bytes=_MAX_REPORT_BYTES,
                )
            except StrictJsonError as exc:
                errors.append(str(exc))
                continue
            digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
            if reference.get("digest") != digest:
                errors.append(f"manifest digest does not bind {relative}")
            if block_name == "inputs":
                try:
                    identity = parse_json_bytes(raw, label=f"input identity {role}")
                except StrictJsonError as exc:
                    errors.append(str(exc))
                    continue
                if not isinstance(identity, dict):
                    errors.append(f"input identity {role} must be a JSON object")
                elif reference.get("material_digest") != identity.get("digest"):
                    errors.append(f"manifest material digest does not bind {relative}")
    paired = manifest.get("paired_records")
    if isinstance(paired, dict) and isinstance(paired.get("path"), str):
        relative = paired["path"]
        try:
            raw = read_regular_file_bytes(
                evidence / relative,
                label="paired records",
                max_bytes=_MAX_REPORT_BYTES,
            )
            payload = parse_json_bytes(raw, label="paired records")
        except StrictJsonError as exc:
            errors.append(str(exc))
        else:
            digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
            if paired.get("digest") != digest:
                errors.append("manifest digest does not bind paired records")
            records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(records, list) or len(records) != paired.get("count"):
                errors.append("manifest count does not bind paired records")
    return errors


def _signature_verified_report(
    evidence: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest, manifest_raw = _load_object_with_bytes(
        evidence / "manifest.json",
        label="evidence manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    errors = _manifest_schema_errors(manifest)
    if canonical_json_bytes(manifest) != manifest_raw:
        errors.append("evidence manifest is not canonical JSON")
    signature_errors, signature_warnings, signer = integrity.verify_signature(
        evidence,
        strict=True,
        expected_fingerprints=None,
    )
    errors.extend(signature_errors)
    errors.extend(signature_warnings)
    # Do not interpret any manifest-controlled path until the signed manifest
    # has passed the closed schema. The schema pins every payload path.
    if errors:
        raise EvidenceReportError("; ".join(dict.fromkeys(errors)))
    try:
        checksums = read_regular_file_bytes(
            evidence / "checksums.sha256",
            label="checksums.sha256",
            max_bytes=1024 * 1024,
        )
    except StrictJsonError as exc:
        errors.append(str(exc))
        checksums = b""
    errors.extend(
        integrity.verify_manifest_binds_checksums_payload(manifest, checksums)
    )
    checksum_errors, covered = integrity.verify_checksums(evidence)
    errors.extend(checksum_errors)
    extra_errors, _warnings = integrity.verify_no_extra_files(
        evidence, covered_paths=covered, strict=True
    )
    errors.extend(extra_errors)
    expected_payloads = _manifest_payload_paths(manifest)
    expected_inventory = {
        *expected_payloads,
        "checksums.sha256",
        "manifest.json",
        integrity.MANIFEST_SIGNATURE_FILENAME,
    }
    actual_inventory = {
        path.relative_to(evidence).as_posix()
        for path in evidence.rglob("*")
        if path.is_file()
    }
    unexpected_files = sorted(actual_inventory - expected_inventory)
    missing_files = sorted(expected_inventory - actual_inventory)
    if unexpected_files:
        errors.append(
            "evidence pack contains files outside the evidence manifest: "
            + ", ".join(unexpected_files)
        )
    if missing_files:
        errors.append(
            "evidence pack is missing closed inventory files: "
            + ", ".join(missing_files)
        )
    missing_coverage = sorted(expected_payloads - covered)
    unexpected_coverage = sorted(covered - expected_payloads)
    if missing_coverage:
        errors.append(
            "checksums.sha256 does not cover manifest payloads: "
            + ", ".join(missing_coverage)
        )
    if unexpected_coverage:
        errors.append(
            "checksums.sha256 covers files outside the evidence manifest: "
            + ", ".join(unexpected_coverage)
        )
    errors.extend(_reference_binding_errors(evidence, manifest))
    observations, observation_errors = _load_observations(evidence, manifest)
    errors.extend(observation_errors)
    if errors:
        raise EvidenceReportError("; ".join(dict.fromkeys(errors)))
    if not isinstance(signer, str):
        raise EvidenceReportError("evidence signature fingerprint is unavailable")
    evidence_block = manifest.get("evidence")
    reference = (
        evidence_block.get("evaluation_report")
        if isinstance(evidence_block, dict)
        else None
    )
    relative = reference.get("path") if isinstance(reference, dict) else None
    if relative != "reports/evaluation.report.json":
        raise EvidenceReportError("manifest does not bind the canonical report role")
    report_path = evidence / "reports/evaluation.report.json"
    try:
        report_path.resolve().relative_to(evidence.resolve())
    except ValueError as exc:
        raise EvidenceReportError(
            "canonical report path escapes the evidence pack"
        ) from exc
    report, report_raw = _load_object_with_bytes(
        evidence / "reports/evaluation.report.json",
        label="canonical evaluation report",
        max_bytes=_MAX_REPORT_BYTES,
    )
    if canonical_json_bytes(report) != report_raw:
        raise EvidenceReportError("canonical evaluation report is not canonical JSON")
    return _closed_comparison_report(report), signer, observations


def _write_html_no_clobber(path: Path, html: str) -> Path:
    destination = Path(path).absolute()
    if destination.name in {"", ".", ".."}:
        raise EvidenceReportError("HTML destination must name a regular file")
    root_fd = os.open("/", _DIRECTORY_FLAGS)
    current_fd = root_fd
    descriptor: int | None = None
    try:
        for component in destination.parent.parts[1:]:
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=current_fd)
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination.name, flags, 0o600, dir_fd=current_fd)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(html)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if exc.errno == errno.EEXIST:
            raise EvidenceReportError(
                f"HTML destination already exists: {destination}"
            ) from exc
        raise EvidenceReportError(
            f"could not write HTML report: {exc}", exit_code=1
        ) from exc
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    return destination


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceReportError(f"canonical report {field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceReportError(f"canonical report {field} must be finite")
    return result


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceReportError(
            f"canonical report {field} must be a non-negative integer"
        )
    return value


def _validate_paired_binary(
    value: object,
    *,
    record_count: int,
    comparison_value: float,
    uncertainty: dict[str, Any],
) -> None:
    expected_fields = {
        "baseline_pass_subject_fail",
        "baseline_fail_subject_pass",
        "both_pass",
        "both_fail",
        "discordant_pairs",
        "mcnemar_exact_two_sided_p_value",
        "effect_size_pp",
        "effect_size_confidence_interval",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise EvidenceReportError("canonical report paired_binary is invalid")
    regressions = _nonnegative_integer(
        value["baseline_pass_subject_fail"],
        field="paired_binary.baseline_pass_subject_fail",
    )
    improvements = _nonnegative_integer(
        value["baseline_fail_subject_pass"],
        field="paired_binary.baseline_fail_subject_pass",
    )
    both_pass = _nonnegative_integer(
        value["both_pass"], field="paired_binary.both_pass"
    )
    both_fail = _nonnegative_integer(
        value["both_fail"], field="paired_binary.both_fail"
    )
    discordant = _nonnegative_integer(
        value["discordant_pairs"], field="paired_binary.discordant_pairs"
    )
    if (
        regressions + improvements != discordant
        or regressions + improvements + both_pass + both_fail != record_count
    ):
        raise EvidenceReportError("canonical report paired_binary counts are invalid")
    baseline = [True] * both_pass + [True] * regressions
    subject = [True] * both_pass + [False] * regressions
    baseline += [False] * improvements + [False] * both_fail
    subject += [True] * improvements + [False] * both_fail
    method = uncertainty.get("method")
    if method not in PAIRED_CONFIDENCE_INTERVAL_METHODS:
        raise EvidenceReportError("canonical report uncertainty method is invalid")
    try:
        replayed = paired_exact_match_statistics(
            baseline,
            subject,
            confidence_interval_method=cast(str, method),
        )
    except PairedExactMatchError as exc:
        raise EvidenceReportError(
            f"canonical report paired_binary cannot be replayed: {exc}"
        ) from exc
    p_value = _number(
        value["mcnemar_exact_two_sided_p_value"],
        field="paired_binary.mcnemar_exact_two_sided_p_value",
    )
    effect = _number(value["effect_size_pp"], field="paired_binary.effect_size_pp")
    interval = value["effect_size_confidence_interval"]
    expected_interval = replayed.effect_size_confidence_interval
    if (
        not 0.0 <= p_value <= 1.0
        or not math.isclose(
            p_value,
            replayed.mcnemar_exact_two_sided_p_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(effect, replayed.effect_size_pp, abs_tol=1e-12)
        or not math.isclose(effect, comparison_value, abs_tol=1e-12)
        or not isinstance(interval, dict)
        or set(interval) != {"method", "confidence_level", "lower_pp", "upper_pp"}
        or interval.get("method") != expected_interval.method
        or interval.get("confidence_level") != expected_interval.confidence_level
        or not math.isclose(
            _number(interval.get("lower_pp"), field="paired_binary.interval.lower_pp"),
            expected_interval.lower_pp,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _number(interval.get("upper_pp"), field="paired_binary.interval.upper_pp"),
            expected_interval.upper_pp,
            abs_tol=1e-12,
        )
        or not math.isclose(
            expected_interval.lower_pp,
            _number(uncertainty.get("lower"), field="uncertainty.lower"),
            abs_tol=1e-12,
        )
        or not math.isclose(
            expected_interval.upper_pp,
            _number(uncertainty.get("upper"), field="uncertainty.upper"),
            abs_tol=1e-12,
        )
    ):
        raise EvidenceReportError("canonical report paired_binary values are invalid")


def _validate_sample_qualification(
    value: object,
    *,
    metric: str,
    record_count: int,
    interval_lower: float,
    interval_upper: float,
) -> bool:
    """Replay the closed sample-count and interval-precision qualification."""

    if not isinstance(value, dict) or set(value) != {
        "record_count",
        "interval_width",
        "passed",
    }:
        raise EvidenceReportError("canonical report sample_qualification is invalid")
    count_qualification = value.get("record_count")
    width_qualification = value.get("interval_width")
    if not isinstance(count_qualification, dict) or set(count_qualification) != {
        "minimum",
        "observed",
        "passed",
    }:
        raise EvidenceReportError(
            "canonical report sample_qualification record_count is invalid"
        )
    if not isinstance(width_qualification, dict) or set(width_qualification) != {
        "maximum",
        "observed",
        "unit",
        "passed",
    }:
        raise EvidenceReportError(
            "canonical report sample_qualification interval_width is invalid"
        )
    minimum = _nonnegative_integer(
        count_qualification.get("minimum"),
        field="sample_qualification.record_count.minimum",
    )
    observed_count = _nonnegative_integer(
        count_qualification.get("observed"),
        field="sample_qualification.record_count.observed",
    )
    maximum_width = _number(
        width_qualification.get("maximum"),
        field="sample_qualification.interval_width.maximum",
    )
    observed_width = _number(
        width_qualification.get("observed"),
        field="sample_qualification.interval_width.observed",
    )
    expected_unit = (
        "ratio" if metric == "normalized_nll_per_utf8_byte" else "percentage_points"
    )
    if (
        minimum < 1
        or minimum > 10_000
        or observed_count != record_count
        or maximum_width <= 0.0
        or (expected_unit == "percentage_points" and maximum_width > 200.0)
        or observed_width < 0.0
        or width_qualification.get("unit") != expected_unit
        or not math.isclose(
            observed_width,
            interval_upper - interval_lower,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise EvidenceReportError(
            "canonical report sample_qualification values are invalid"
        )
    count_passed = observed_count >= minimum
    width_passed = observed_width <= maximum_width
    qualified = count_passed and width_passed
    recorded_count_passed = count_qualification.get("passed")
    recorded_width_passed = width_qualification.get("passed")
    recorded_qualified = value.get("passed")
    if (
        not isinstance(recorded_count_passed, bool)
        or recorded_count_passed != count_passed
        or not isinstance(recorded_width_passed, bool)
        or recorded_width_passed != width_passed
        or not isinstance(recorded_qualified, bool)
        or recorded_qualified != qualified
    ):
        raise EvidenceReportError(
            "canonical report sample_qualification verdict is invalid"
        )
    return qualified


def _validate_side_accuracy(
    value: object,
    *,
    metric: str,
    side_means: dict[str, float],
) -> bool:
    """Replay the signed per-side accuracy floor for exact-match reports."""

    if (
        metric != "exact_match"
        or not isinstance(value, dict)
        or set(value)
        != {
            "minimum",
            "baseline",
            "subject",
            "passed",
        }
    ):
        raise EvidenceReportError("canonical report side_accuracy is invalid")
    minimum = _number(value.get("minimum"), field="side_accuracy.minimum")
    if not 0.0 <= minimum <= 1.0:
        raise EvidenceReportError("canonical report side_accuracy minimum is invalid")
    recorded: dict[str, bool] = {}
    for side in ("baseline", "subject"):
        qualification = value.get(side)
        if not isinstance(qualification, dict) or set(qualification) != {
            "observed",
            "passed",
        }:
            raise EvidenceReportError(
                f"canonical report side_accuracy {side} is invalid"
            )
        observed = _number(
            qualification.get("observed"),
            field=f"side_accuracy.{side}.observed",
        )
        expected = side_means[side]
        passed = expected >= minimum
        recorded_passed = qualification.get("passed")
        if (
            not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            or not isinstance(recorded_passed, bool)
            or recorded_passed != passed
        ):
            raise EvidenceReportError(
                f"canonical report side_accuracy {side} values are invalid"
            )
        recorded[side] = passed
    qualified = recorded["baseline"] and recorded["subject"]
    recorded_qualified = value.get("passed")
    if not isinstance(recorded_qualified, bool) or recorded_qualified != qualified:
        raise EvidenceReportError("canonical report side_accuracy verdict is invalid")
    return qualified


def _comparison_report_shape(
    report: dict[str, Any],
) -> tuple[str, str, int, dict[str, float], dict[str, Any], str, float, float]:
    expected = {
        "format",
        "comparison_id",
        "metric",
        "record_count",
        "baseline",
        "subject",
        "comparison",
        "uncertainty",
        "policy_digest",
        "verdict",
    }
    metric = report.get("metric")
    if metric == "exact_match":
        expected.add("paired_binary")
    elif metric == "normalized_nll_per_utf8_byte":
        expected.add("derived_measurements")
    else:
        expected.update({"scorer_extension", "scorer_replay"})
    if "sample_qualification" in report:
        expected.add("sample_qualification")
    if "side_accuracy" in report:
        expected.add("side_accuracy")
    if set(report) != expected:
        raise EvidenceReportError("canonical comparison report fields are invalid")
    report_format = report.get("format")
    if report_format not in {
        LEGACY_COMPARISON_REPORT_FORMAT,
        COMPARISON_REPORT_FORMAT,
    }:
        raise EvidenceReportError("canonical comparison report format is invalid")
    for field in ("comparison_id", "metric", "policy_digest"):
        if not isinstance(report.get(field), str) or not report[field]:
            raise EvidenceReportError(f"canonical report {field} is invalid")
    metric = cast(str, report["metric"])
    count = report.get("record_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise EvidenceReportError("canonical report record_count is invalid")
    side_means: dict[str, float] = {}
    for side in ("baseline", "subject"):
        value = report.get(side)
        if not isinstance(value, dict) or set(value) != {"mean_score"}:
            raise EvidenceReportError(f"canonical report {side} is invalid")
        side_means[side] = _number(value["mean_score"], field=f"{side}.mean_score")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise EvidenceReportError("canonical report comparison is invalid")
    kind = comparison.get("kind")
    expected_comparison_fields = (
        {"kind", "value", "minimum"}
        if kind in {"exact_match_delta_pp", "scorer_extension_delta_pp"}
        else {"kind", "value", "maximum"}
    )
    if (
        kind
        not in {
            "exact_match_delta_pp",
            "normalized_nll_ratio",
            "scorer_extension_delta_pp",
        }
        or set(comparison) != expected_comparison_fields
    ):
        raise EvidenceReportError("canonical report comparison is invalid")
    comparison_value = _number(comparison["value"], field="comparison.value")
    limit_field = (
        "minimum"
        if kind in {"exact_match_delta_pp", "scorer_extension_delta_pp"}
        else "maximum"
    )
    limit = _number(comparison[limit_field], field=f"comparison.{limit_field}")
    expected_kind = {
        "exact_match": "exact_match_delta_pp",
        "normalized_nll_per_utf8_byte": "normalized_nll_ratio",
    }.get(metric)
    if expected_kind is None:
        try:
            binding = decode_scorer_binding(report.get("scorer_extension"))
        except ScorerExtensionError as exc:
            raise EvidenceReportError(str(exc)) from exc
        replay = report.get("scorer_replay")
        if (
            binding.scorer_id != metric
            or not isinstance(replay, dict)
            or set(replay) != {"baseline", "subject"}
        ):
            raise EvidenceReportError("canonical report scorer binding is invalid")
        expected_kind = "scorer_extension_delta_pp"
    if kind != expected_kind:
        raise EvidenceReportError(
            "canonical report metric and comparison kind do not agree"
        )
    if report.get("verdict") not in {"pass", "fail"}:
        raise EvidenceReportError("canonical report verdict is invalid")
    return (
        report_format,
        metric,
        count,
        side_means,
        comparison,
        kind,
        comparison_value,
        limit,
    )


def _comparison_uncertainty(
    report: dict[str, Any], *, metric: str, report_format: str
) -> tuple[dict[str, Any], float, float]:
    uncertainty = report.get("uncertainty")
    if not isinstance(uncertainty, dict):
        raise EvidenceReportError("canonical report uncertainty is invalid")
    if metric == "exact_match":
        if set(uncertainty) != {
            "method",
            "scope",
            "interval_mass",
            "lower",
            "upper",
        }:
            raise EvidenceReportError("canonical report uncertainty is invalid")
        expected_method = (
            "newcombe_hybrid_score_paired_v1"
            if report_format == LEGACY_COMPARISON_REPORT_FORMAT
            else "newcombe_hybrid_score_paired_v2"
        )
        if uncertainty.get("method") != expected_method:
            raise EvidenceReportError("canonical report uncertainty method is invalid")
        if uncertainty.get("scope") != "paired_binary_outcomes":
            raise EvidenceReportError("canonical report uncertainty scope is invalid")
    else:
        if set(uncertainty) != {
            "method",
            "scope",
            "interval_mass",
            "replicates",
            "lower",
            "upper",
        }:
            raise EvidenceReportError("canonical report uncertainty is invalid")
        if uncertainty.get("method") != "paired_percentile_bootstrap_sha256_v1":
            raise EvidenceReportError("canonical report uncertainty method is invalid")
        if uncertainty.get("scope") != "authenticated_schedule":
            raise EvidenceReportError("canonical report uncertainty scope is invalid")
        if uncertainty.get("replicates") != 2048:
            raise EvidenceReportError(
                "canonical report uncertainty replicates are invalid"
            )
    if uncertainty.get("interval_mass") != 0.95:
        raise EvidenceReportError("canonical report uncertainty mass is invalid")
    lower = _number(uncertainty.get("lower"), field="uncertainty.lower")
    upper = _number(uncertainty.get("upper"), field="uncertainty.upper")
    if lower > upper:
        raise EvidenceReportError("canonical report uncertainty bounds are invalid")
    return uncertainty, lower, upper


def _comparison_acceptance(
    report: dict[str, Any],
    *,
    metric: str,
    count: int,
    side_means: dict[str, float],
    kind: str,
    comparison_value: float,
    limit: float,
    uncertainty: dict[str, Any],
    lower: float,
    upper: float,
) -> bool:
    baseline_mean = side_means["baseline"]
    subject_mean = side_means["subject"]
    if kind in {"exact_match_delta_pp", "scorer_extension_delta_pp"}:
        if not 0.0 <= baseline_mean <= 1.0 or not 0.0 <= subject_mean <= 1.0:
            raise EvidenceReportError(
                "canonical report score means must be between zero and one"
            )
        if not -100.0 <= comparison_value <= 100.0:
            raise EvidenceReportError(
                "canonical report score comparison is out of range"
            )
        if not -100.0 <= limit <= 100.0:
            raise EvidenceReportError(
                "canonical report score policy limit is out of range"
            )
        if not -100.0 <= lower <= upper <= 100.0:
            raise EvidenceReportError(
                "canonical report score uncertainty is out of range"
            )
        expected_value = (subject_mean - baseline_mean) * 100.0
        if not math.isclose(
            comparison_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise EvidenceReportError(
                "canonical report comparison value does not match the side means"
            )
        if kind == "exact_match_delta_pp":
            _validate_paired_binary(
                report.get("paired_binary"),
                record_count=count,
                comparison_value=comparison_value,
                uncertainty=uncertainty,
            )
        passed = lower >= limit
    else:
        if baseline_mean <= 0.0 or subject_mean < 0.0:
            raise EvidenceReportError(
                "canonical report ratio means must be non-negative with a positive baseline"
            )
        if comparison_value < 0.0 or limit <= 0.0 or lower < 0.0:
            raise EvidenceReportError(
                "canonical report ratio comparison and uncertainty are invalid"
            )
        expected_value = subject_mean / baseline_mean
        try:
            validated_derived_measurements(report.get("derived_measurements"))
        except EvidencePackError as exc:
            raise EvidenceReportError(
                f"canonical report derived measurements are invalid: {exc}"
            ) from exc
        passed = upper <= limit
    if not math.isclose(comparison_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceReportError(
            "canonical report comparison value does not match the side means"
        )
    if "sample_qualification" in report:
        passed = passed and _validate_sample_qualification(
            report["sample_qualification"],
            metric=metric,
            record_count=count,
            interval_lower=lower,
            interval_upper=upper,
        )
    if "side_accuracy" in report:
        passed = passed and _validate_side_accuracy(
            report["side_accuracy"], metric=metric, side_means=side_means
        )
    return passed


def _closed_comparison_report(report: dict[str, Any]) -> dict[str, Any]:
    (
        report_format,
        metric,
        count,
        side_means,
        comparison,
        kind,
        comparison_value,
        limit,
    ) = _comparison_report_shape(report)
    uncertainty, lower, upper = _comparison_uncertainty(
        report, metric=metric, report_format=report_format
    )
    passed = _comparison_acceptance(
        report,
        metric=metric,
        count=count,
        side_means=side_means,
        kind=kind,
        comparison_value=comparison_value,
        limit=limit,
        uncertainty=uncertainty,
        lower=lower,
        upper=upper,
    )
    expected_verdict = "pass" if passed else "fail"
    if report["verdict"] != expected_verdict:
        raise EvidenceReportError(
            "canonical report verdict does not match the uncertainty bound and policy limit"
        )
    return report


def _format_number(value: object) -> str:
    return format(_number(value, field="numeric value"), ".8g")


def _render_markdown(
    report: dict[str, Any],
    *,
    explain: bool,
    evidence_signer: str,
    observations: list[dict[str, Any]],
) -> str:
    comparison = cast(dict[str, Any], report["comparison"])
    uncertainty = cast(dict[str, Any], report["uncertainty"])
    baseline = cast(dict[str, Any], report["baseline"])
    subject = cast(dict[str, Any], report["subject"])
    if comparison["kind"] == "exact_match_delta_pp":
        comparison_label = "Exact-match delta (pp)"
        limit_label = "Minimum allowed (pp)"
        limit_value = comparison["minimum"]
    elif comparison["kind"] == "normalized_nll_ratio":
        comparison_label = "Normalized NLL ratio"
        limit_label = "Maximum allowed ratio"
        limit_value = comparison["maximum"]
    elif comparison["kind"] == "scorer_extension_delta_pp":
        comparison_label = "Extension scorer delta (pp)"
        limit_label = "Minimum allowed (pp)"
        limit_value = comparison["minimum"]
    else:  # pragma: no cover - closed report validation rejects other kinds
        raise EvidenceReportError("canonical report comparison kind is invalid")
    interval_label = (
        "Paired 95% confidence interval"
        if comparison["kind"] in {"exact_match_delta_pp", "scorer_extension_delta_pp"}
        else "Paired resampling interval (authenticated schedule)"
    )
    lines = [
        "# InvarLock comparison report",
        "",
        f"- **Comparison:** `{report['comparison_id']}`",
        f"- **Metric:** `{report['metric']}`",
        f"- **Records:** {report['record_count']}",
        f"- **Verdict:** **{str(report['verdict']).upper()}**",
        "- **Bundle integrity:** embedded evidence signature verified",
        "- **Acceptance path:** `invarlock verify` records the expected signer and "
        + "independent anchors in a signed receipt",
        f"- **Evidence signer:** `{evidence_signer}`",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Baseline mean | {_format_number(baseline['mean_score'])} |",
        f"| Subject mean | {_format_number(subject['mean_score'])} |",
        f"| {comparison_label} | {_format_number(comparison['value'])} |",
        f"| {interval_label} | "
        + f"[{_format_number(uncertainty['lower'])}, "
        + f"{_format_number(uncertainty['upper'])}] |",
        f"| {limit_label} | {_format_number(limit_value)} |",
        "",
        f"Policy: `{report['policy_digest']}`",
        "",
        "This report is a human rendering of the signature-authenticated "
        + "evidence bundle. Run `invarlock verify` with independently supplied "
        + "anchors to create the signed acceptance receipt.",
    ]
    paired_binary = report.get("paired_binary")
    if isinstance(paired_binary, dict):
        lines.extend(
            [
                "",
                "## Paired outcome analysis",
                "",
                "| Paired outcome | Count |",
                "| --- | ---: |",
                "| Baseline pass → subject fail | "
                + f"{paired_binary['baseline_pass_subject_fail']} |",
                "| Baseline fail → subject pass | "
                + f"{paired_binary['baseline_fail_subject_pass']} |",
                f"| Both pass | {paired_binary['both_pass']} |",
                f"| Both fail | {paired_binary['both_fail']} |",
                "",
                "McNemar exact two-sided p-value: "
                + f"`{_format_number(paired_binary['mcnemar_exact_two_sided_p_value'])}`.",
                "The p-value describes paired asymmetry; the policy verdict uses the "
                + "effect-size confidence bound above.",
            ]
        )
    sample_qualification = report.get("sample_qualification")
    if isinstance(sample_qualification, dict):
        count_qualification = cast(dict[str, Any], sample_qualification["record_count"])
        width_qualification = cast(
            dict[str, Any], sample_qualification["interval_width"]
        )
        width_label = (
            "Confidence-interval width (pp)"
            if width_qualification["unit"] == "percentage_points"
            else "Resampling-interval width (ratio)"
        )
        lines.extend(
            [
                "",
                "## Sample qualification",
                "",
                "The authenticated policy requires both enough paired records and "
                + "a sufficiently precise interval before the metric can pass.",
                "",
                "| Qualification | Observed | Required | Result |",
                "| --- | ---: | ---: | --- |",
                "| Paired records | "
                + f"{count_qualification['observed']} | "
                + f"≥ {count_qualification['minimum']} | "
                + f"{'pass' if count_qualification['passed'] else 'fail'} |",
                f"| {width_label} | "
                + f"{_format_number(width_qualification['observed'])} | "
                + f"≤ {_format_number(width_qualification['maximum'])} | "
                + f"{'pass' if width_qualification['passed'] else 'fail'} |",
            ]
        )
    derived = report.get("derived_measurements")
    if isinstance(derived, dict):
        measurement = cast(dict[str, Any], derived["perplexity_ratio"])
        lines.extend(["", "## Derived likelihood interpretation", ""])
        if measurement["status"] == "available":
            lines.extend(
                [
                    "The authenticated tokenizer and target token counts are comparable. "
                    + "These values are derived from the same expected-continuation "
                    + "likelihood facts but do not affect acceptance.",
                    "",
                    "| Derived measure | Value |",
                    "| --- | ---: |",
                    "| Baseline perplexity | "
                    + f"{_format_number(measurement['baseline_perplexity'])} |",
                    "| Subject perplexity | "
                    + f"{_format_number(measurement['subject_perplexity'])} |",
                    f"| Perplexity ratio | {_format_number(measurement['ratio'])} |",
                ]
            )
        else:
            lines.append(
                "Perplexity interpretation is unavailable: "
                + f"`{measurement['reason']}`. Acceptance remains based on "
                + "normalized NLL per expected UTF-8 byte."
            )
    if observations:
        lines.extend(
            [
                "",
                "## Authenticated observations",
                "",
                "These authenticated observations provide supplementary context. "
                + "The paired metric and policy remain the complete acceptance calculation.",
                "",
                "| Observation | Kind | Scope |",
                "| --- | --- | --- |",
            ]
        )
        for observation in observations:
            lines.append(
                f"| `{observation['observation_id']}` | "
                + f"`{observation['kind']}` | `{observation['scope']}` |"
            )
        for observation in observations:
            payload = json.dumps(
                observation["payload"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            lines.extend(
                [
                    "",
                    f"### `{observation['observation_id']}`",
                    "",
                    *(f"    {line}" for line in payload.splitlines()),
                ]
            )
    if explain:
        lines.extend(
            [
                "",
                "The displayed verdict is the canonical report bound by the "
                + "manifest, checksums, and embedded evidence signature.",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_html(
    report: dict[str, Any],
    *,
    explain: bool,
    evidence_signer: str,
    observations: list[dict[str, Any]],
) -> str:
    comparison = cast(dict[str, Any], report["comparison"])
    uncertainty = cast(dict[str, Any], report["uncertainty"])
    baseline = cast(dict[str, Any], report["baseline"])
    subject = cast(dict[str, Any], report["subject"])
    limit_field = (
        "minimum"
        if comparison["kind"] in {"exact_match_delta_pp", "scorer_extension_delta_pp"}
        else "maximum"
    )
    comparison_label = {
        "exact_match_delta_pp": "Exact-match delta (pp)",
        "normalized_nll_ratio": "Normalized NLL ratio",
        "scorer_extension_delta_pp": "Extension scorer delta (pp)",
    }[comparison["kind"]]
    interval_label = (
        "Paired 95% confidence interval"
        if comparison["kind"] in {"exact_match_delta_pp", "scorer_extension_delta_pp"}
        else "Paired resampling interval (authenticated schedule)"
    )
    note = (
        "<p>The displayed verdict is the canonical report bound by the "
        "manifest, checksums, and embedded evidence signature.</p>"
        if explain
        else ""
    )
    observations_html = ""
    paired_html = ""
    paired_binary = report.get("paired_binary")
    if isinstance(paired_binary, dict):
        paired_html = (
            "<h2>Paired outcome analysis</h2>"
            "<table><thead><tr><th>Paired outcome</th><th>Count</th></tr></thead>"
            "<tbody>"
            "<tr><td>Baseline pass → subject fail</td><td>"
            f"{paired_binary['baseline_pass_subject_fail']}</td></tr>"
            "<tr><td>Baseline fail → subject pass</td><td>"
            f"{paired_binary['baseline_fail_subject_pass']}</td></tr>"
            f"<tr><td>Both pass</td><td>{paired_binary['both_pass']}</td></tr>"
            f"<tr><td>Both fail</td><td>{paired_binary['both_fail']}</td></tr>"
            "</tbody></table>"
            "<p>McNemar exact two-sided p-value: <code>"
            f"{_format_number(paired_binary['mcnemar_exact_two_sided_p_value'])}"
            "</code>. The p-value describes paired asymmetry; the policy verdict "
            "uses the effect-size confidence bound above.</p>"
        )
    derived_html = ""
    derived = report.get("derived_measurements")
    if isinstance(derived, dict):
        measurement = cast(dict[str, Any], derived["perplexity_ratio"])
        if measurement["status"] == "available":
            derived_html = (
                "<h2>Derived likelihood interpretation</h2>"
                "<p>The authenticated tokenizer and target token counts are "
                "comparable. These values are derived from expected-continuation "
                "likelihood facts and do not affect acceptance.</p>"
                "<table><thead><tr><th>Derived measure</th><th>Value</th></tr>"
                "</thead><tbody>"
                "<tr><td>Baseline perplexity</td><td>"
                f"{_format_number(measurement['baseline_perplexity'])}</td></tr>"
                "<tr><td>Subject perplexity</td><td>"
                f"{_format_number(measurement['subject_perplexity'])}</td></tr>"
                "<tr><td>Perplexity ratio</td><td>"
                f"{_format_number(measurement['ratio'])}</td></tr>"
                "</tbody></table>"
            )
        else:
            derived_html = (
                "<h2>Derived likelihood interpretation</h2>"
                "<p>Perplexity interpretation is unavailable: <code>"
                f"{escape(str(measurement['reason']))}</code>. Acceptance remains "
                "based on normalized NLL per expected UTF-8 byte.</p>"
            )
    if observations:
        rows = "".join(
            "<tr>"
            f"<td><code>{escape(str(item['observation_id']))}</code></td>"
            f"<td><code>{escape(str(item['kind']))}</code></td>"
            f"<td><code>{escape(str(item['scope']))}</code></td>"
            "</tr>"
            for item in observations
        )
        details = "".join(
            f"<h3><code>{escape(str(item['observation_id']))}</code></h3>"
            "<pre>"
            + escape(
                json.dumps(
                    item["payload"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            + "</pre>"
            for item in observations
        )
        observations_html = (
            "<h2>Authenticated observations</h2>"
            "<p>These authenticated observations provide supplementary context. "
            "The paired metric and policy remain the complete acceptance calculation.</p>"
            "<table><thead><tr><th>Observation</th><th>Kind</th><th>Scope</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>{details}"
        )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>InvarLock comparison report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:52rem;margin:3rem auto;"
        "padding:0 1rem;color:#17202a}table{border-collapse:collapse;width:100%}"
        "th,td{border-bottom:1px solid #d5d8dc;padding:.65rem;text-align:left}"
        "td:last-child{text-align:right}code{overflow-wrap:anywhere}</style></head><body>"
        "<h1>InvarLock comparison report</h1>"
        f"<p><strong>Comparison:</strong> <code>{escape(str(report['comparison_id']))}</code>"
        f"<br><strong>Metric:</strong> <code>{escape(str(report['metric']))}</code>"
        f"<br><strong>Records:</strong> {report['record_count']}"
        f"<br><strong>Verdict:</strong> {escape(str(report['verdict']).upper())}"
        "<br><strong>Bundle integrity:</strong> embedded evidence signature verified"
        "<br><strong>Acceptance path:</strong> <code>invarlock verify</code> records "
        "the expected signer and independent anchors in a signed receipt"
        f"<br><strong>Evidence signer:</strong> <code>{escape(evidence_signer)}</code>"
        "</p>"
        "<table><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>Baseline mean</td><td>{_format_number(baseline['mean_score'])}</td></tr>"
        f"<tr><td>Subject mean</td><td>{_format_number(subject['mean_score'])}</td></tr>"
        f"<tr><td>{comparison_label}</td><td>{_format_number(comparison['value'])}</td></tr>"
        f"<tr><td>{interval_label}</td><td>"
        f"[{_format_number(uncertainty['lower'])}, "
        f"{_format_number(uncertainty['upper'])}]</td></tr>"
        f"<tr><td>{limit_field.title()}</td><td>{_format_number(comparison[limit_field])}</td></tr>"
        "</tbody></table>"
        f"<p><strong>Policy:</strong> <code>{escape(str(report['policy_digest']))}</code></p>"
        "<p>This report is a human rendering of the signature-authenticated "
        "evidence bundle. Run <code>invarlock verify</code> with independently "
        "supplied anchors to create the signed acceptance receipt.</p>"
        f"{paired_html}"
        f"{derived_html}"
        f"{observations_html}"
        f"{note}</body></html>\n"
    )


def render_evidence(
    evidence_path: Path,
    *,
    html_path: Path | None = None,
    explain: bool = False,
) -> EvidenceReport:
    """Render the signature-authenticated canonical report without mutation."""

    evidence = Path(evidence_path)
    if not evidence.is_dir() or evidence.is_symlink():
        raise EvidenceReportError("evidence must be a real directory")
    if html_path is not None:
        try:
            Path(html_path).absolute().resolve().relative_to(evidence.resolve())
        except ValueError:
            pass
        else:
            raise EvidenceReportError(
                "HTML destination must remain outside the immutable evidence pack"
            )
    snapshot, capture_errors = PackSnapshot.capture(
        evidence, validate_structural_json=False
    )
    if snapshot is None:
        raise EvidenceReportError("; ".join(capture_errors))
    try:
        with snapshot.files.materialized() as snapshot_root:
            report, evidence_signer, observations = _signature_verified_report(
                snapshot_root
            )
            text = _render_markdown(
                report,
                explain=explain,
                evidence_signer=evidence_signer,
                observations=observations,
            )
            rendered_html = (
                _render_html(
                    report,
                    explain=explain,
                    evidence_signer=evidence_signer,
                    observations=observations,
                )
                if html_path is not None
                else None
            )
            materialized_errors = snapshot.files.materialized_stability_errors(
                snapshot_root
            )
    except RuntimeError as exc:
        raise EvidenceReportError(str(exc)) from exc
    stability_errors = [*materialized_errors, *snapshot.stability_errors()]
    if stability_errors:
        raise EvidenceReportError("; ".join(stability_errors))
    manifest_entry = snapshot.files.entry("manifest.json")
    if manifest_entry is None:  # pragma: no cover - capture contract owns inventory
        raise EvidenceReportError("evidence manifest snapshot is unavailable")
    output = (
        _write_html_no_clobber(html_path, rendered_html)
        if html_path is not None and rendered_html is not None
        else None
    )
    return EvidenceReport(
        text=text,
        html_path=output,
        evidence_signer=evidence_signer,
        pack_manifest_digest="sha256:" + manifest_entry.sha256,
        observations=tuple(observations),
    )


__all__ = ["EvidenceReport", "EvidenceReportError", "render_evidence"]
