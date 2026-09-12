"""Render the canonical report inside an evidence pack."""

from __future__ import annotations

import hashlib
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from jsonschema import Draft202012Validator

from invarlock import evidence_pack_integrity as integrity
from invarlock.core.scorer_extension import (
    ScorerExtensionError,
    decode_scorer_binding,
)
from invarlock.evidence_explanation import core_policy_checks
from invarlock.evidence_pack_contract import (
    COMPARISON_REPORT_FORMAT,
    COMPARISON_REPORT_FORMATS,
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
from invarlock.report_presentation import (
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    number,
    xml_text,
)
from invarlock.report_presentation import (
    render_html as render_report_html,
)
from invarlock.report_presentation import (
    render_markdown as render_report_markdown,
)

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

    def __init__(
        self, message: str, *, exit_code: int = 2, payload: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload
        self.written_outputs = (
            dict(payload.get("written_outputs", {})) if payload else {}
        )
        self.failed_output = payload.get("failed_output") if payload else None


@dataclass(frozen=True)
class EvidenceReport:
    text: str
    html_path: Path | None
    evidence_signer: str
    pack_manifest_digest: str
    observations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvidenceReportV2:
    text: str
    kind: str
    pack_manifest_digest: str
    requested_outputs: dict[str, str]
    written_outputs: dict[str, str]
    failed_output: str | None = None
    errors: tuple[str, ...] = ()

    def as_json(self) -> str:
        return canonical_json_bytes(
            {
                "format_version": "invarlock/evidence-report-v2",
                "kind": self.kind,
                "ok": not self.errors,
                "pack_manifest_digest": self.pack_manifest_digest,
                "requested_outputs": self.requested_outputs,
                "written_outputs": self.written_outputs,
                "failed_output": self.failed_output,
                "errors": list(self.errors),
            }
        ).decode("utf-8")


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
    temporary_name: str | None = None
    try:
        for component in destination.parent.parts[1:]:
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=current_fd)
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            previous_descriptor = current_fd
            current_fd = child_fd
            if previous_descriptor != root_fd:
                os.close(previous_descriptor)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _attempt in range(10):
            candidate_name = ".invarlock-report-" + secrets.token_hex(16)
            try:
                descriptor = os.open(candidate_name, flags, 0o600, dir_fd=current_fd)
            except FileExistsError:
                continue
            temporary_name = candidate_name
            break
        else:
            raise EvidenceReportError("could not allocate a temporary report file")
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except BaseException:
            owned_descriptor = descriptor
            descriptor = None
            os.close(owned_descriptor)
            raise
        descriptor = None
        with handle:
            handle.write(html)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(current_fd)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=current_fd,
                dst_dir_fd=current_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise EvidenceReportError(
                f"HTML destination already exists: {destination}"
            ) from exc
    except OSError as exc:
        raise EvidenceReportError(
            f"could not write HTML report: {exc}", exit_code=1
        ) from exc
    finally:
        try:
            try:
                if descriptor is not None:
                    os.close(descriptor)
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=current_fd)
                    except OSError:
                        pass
            finally:
                if current_fd != root_fd:
                    os.close(current_fd)
        finally:
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
    report_format = report.get("format")
    if report_format not in COMPARISON_REPORT_FORMATS:
        raise EvidenceReportError("canonical comparison report format is invalid")
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
    if report_format == COMPARISON_REPORT_FORMAT and "side_accuracy" in report:
        expected.add("side_accuracy")
    elif "side_accuracy" in report:
        raise EvidenceReportError(
            "canonical report side_accuracy requires comparison-report-v3"
        )
    if set(report) != expected:
        raise EvidenceReportError("canonical comparison report fields are invalid")
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
        sample_passed = _validate_sample_qualification(
            report["sample_qualification"],
            metric=metric,
            record_count=count,
            interval_lower=lower,
            interval_upper=upper,
        )
        passed = passed and sample_passed
    if "side_accuracy" in report:
        accuracy_passed = _validate_side_accuracy(
            report["side_accuracy"], metric=metric, side_means=side_means
        )
        passed = passed and accuracy_passed
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


def _native_report_context(
    evidence: Path,
    report: dict[str, Any],
    identities: dict[str, dict[str, Any]],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Project descriptive facts only after the materialized pack is authenticated."""

    def mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def text(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "Unavailable in evidence"
        return value[:256] + ("… (preview)" if len(value) > 256 else "")

    request_path = evidence / "request.json"
    request = (
        _load_object_with_bytes(
            request_path, label="normalized request", max_bytes=_MAX_REPORT_BYTES
        )[0]
        if request_path.exists()
        else {}
    )
    comparison = mapping(request.get("comparison"))
    mode = mapping(request.get("execution")).get("mode")
    context = [
        (
            "Workflow",
            {"run": "Runtime execution", "import": "Imported runtime evidence"}.get(
                mode, "Unavailable in evidence"
            )
            if isinstance(mode, str)
            else "Unavailable in evidence",
        ),
        ("Task", text(comparison.get("task"))),
        ("Paired records", f"{report['record_count']:,}"),
    ]
    for side, label in (("baseline", "Baseline"), ("subject", "Candidate")):
        selected = mapping(comparison.get(side))
        context.extend(
            (
                (
                    label + " model ID",
                    text(mapping(selected.get("artifact")).get("model_id")),
                ),
                (
                    label + " provider",
                    text(mapping(selected.get("runtime")).get("provider")),
                ),
            )
        )
    dataset, _ = _load_object_with_bytes(
        evidence / "inputs" / "dataset.json",
        label="dataset identity",
        max_bytes=_MAX_REPORT_BYTES,
    )
    context.append(("Schedule digest", text(dataset.get("digest"))))
    preparation = mapping(comparison.get("dataset"))
    context.extend(
        (
            ("Dataset", text(preparation.get("name") or dataset.get("locator"))),
            ("Dataset split", text(preparation.get("split"))),
        )
    )
    if mode == "run" and preparation:
        context.extend(
            (
                ("Dataset source SHA-256", text(preparation.get("source_sha256"))),
                ("Dataset source format", text(preparation.get("source_format"))),
            )
        )
        for key, label in (
            ("selected_record_count", "Selected records"),
            ("limit", "Selection limit"),
        ):
            value = preparation.get(key)
            context.append(
                (
                    label,
                    f"{value:,}"
                    if type(value) is int and value >= 0
                    else "Not specified",
                )
            )
    baseline = identities["baseline"].get("digest")
    subject = identities["subject"].get("digest")
    changes = (
        "Baseline and candidate have the same authenticated artifact digest."
        if baseline == subject
        else "Baseline and candidate have different authenticated artifact digests; the evidence does not identify a transformation procedure.",
    )
    return tuple(context), changes


def _report_view(
    report: dict[str, Any],
    *,
    evidence_signer: str,
    observations: list[dict[str, Any]],
    subjects: tuple[tuple[str, str], ...] = (),
    context: tuple[tuple[str, str], ...] = (),
    changes: tuple[str, ...] = (),
) -> ReportView:
    """Explain validated canonical facts without creating acceptance authority."""
    comparison = report["comparison"]
    uncertainty = report["uncertainty"]
    kind = comparison["kind"]
    exact = kind == "exact_match_delta_pp"
    ratio = kind == "normalized_nll_ratio"
    checks = tuple(CheckView(**check) for check in core_policy_checks(report))
    unmet = [check.name for check in checks if not check.passed]
    summary = (
        "Every configured check passed for this paired evaluation. Independent recipient acceptance is a separate step."
        if report["verdict"] == "pass"
        else "This evaluation did not meet its recorded policy. Checks not met: "
        + ", ".join(unmet)
        + "."
    )
    label = (
        "Paired 95% confidence interval"
        if uncertainty["scope"] == "paired_binary_outcomes"
        else "95% finite-schedule resampling interval"
    )
    unit = "ratio" if ratio else "pp"
    names = {
        "exact_match": "Exact-match accuracy",
        "normalized_nll_per_utf8_byte": "Normalized negative log-likelihood",
    }
    notes = [
        "Lower scores are better; the change is the candidate-to-baseline ratio."
        if ratio
        else "Higher scores are better; the change is candidate minus baseline in percentage points.",
        "The policy tests the interval bound, not just the observed change.",
    ]
    if "sample_qualification" not in report:
        notes.append(
            "This policy does not specify a minimum sample count or interval-width requirement."
        )
    if exact and "side_accuracy" not in report:
        notes.append("This policy does not specify an absolute accuracy floor.")
    details: list[tuple[str, Any]] = []
    if "paired_binary" in report:
        details.append(("Paired outcome analysis", report["paired_binary"]))
    if "derived_measurements" in report:
        details.append(
            ("Derived likelihood interpretation", report["derived_measurements"])
        )
        measurement = report["derived_measurements"]["perplexity_ratio"]
        if measurement["status"] == "available":
            notes.extend(
                (
                    "Baseline perplexity: "
                    + number(measurement["baseline_perplexity"])
                    + ".",
                    "Candidate perplexity: "
                    + number(measurement["subject_perplexity"])
                    + ".",
                    "Perplexity ratio: "
                    + number(measurement["ratio"])
                    + ". These derived values do not affect acceptance.",
                )
            )
        else:
            notes.append(
                "Perplexity interpretation unavailable: "
                + measurement["reason"].replace("_", " ")
                + ". Acceptance uses normalized NLL per expected UTF-8 byte."
            )
    if observations:
        details.append(("Authenticated observations", observations))
        notes.append(
            "Authenticated observations are supplementary; the paired metric and policy remain the complete acceptance calculation."
        )
    value_scale = 100 if exact else 1
    suffix = "%" if exact else " nats / byte" if ratio else " score"
    metric = MetricView(
        name=names.get(report["metric"], report["metric"]),
        scope="All paired records",
        decision=report["verdict"],
        baseline=number(report["baseline"]["mean_score"] * value_scale) + suffix,
        candidate=number(report["subject"]["mean_score"] * value_scale) + suffix,
        change=number(comparison["value"], signed=not ratio) + " " + unit,
        count=f"{report['record_count']:,}",
        explanation="All configured checks passed."
        if not unmet
        else "Checks not met: " + ", ".join(unmet) + ".",
        checks=checks,
        interval=IntervalView(
            lower=uncertainty["lower"],
            upper=uncertainty["upper"],
            estimate=comparison["value"],
            threshold=comparison["maximum" if ratio else "minimum"],
            label=label,
            unit=unit,
        ),
        notes=tuple(notes),
    )
    return ReportView(
        title="InvarLock comparison report",
        family="Runtime evaluation evidence",
        decision=report["verdict"],
        summary=summary,
        metrics=(metric,),
        assurance=(
            (
                "Bundle integrity",
                "Inventory, checksums and embedded evidence signature verified.",
            ),
            ("Recorded policy result", "Read from the authenticated canonical report."),
            (
                "Independent recipient acceptance",
                "Not performed by report. An embedded signer is not a recipient-owned trust anchor.",
            ),
        ),
        identity=(
            ("Comparison", report["comparison_id"]),
            ("Metric", report["metric"]),
            ("Policy", report["policy_digest"]),
            ("Evidence signer", evidence_signer),
        ),
        subjects=subjects,
        context=context,
        changes=changes,
        next_steps=(
            "Review the decision checks and their requirements.",
            "Run invarlock verify with your independently supplied trust profile and receipt destination to create the signed acceptance or rejection receipt.",
            "Keep this report alongside the original immutable evidence and the separate verification receipt.",
        ),
        limitations=(
            "This report summarizes the evidence bundle. Rendering checks its integrity and embedded signature; independent acceptance requires verification with your own trust inputs.",
            "Results apply to the recorded cases, metric and policy. A pass does not establish general model quality, safety or representative production performance.",
            "The interval describes paired binary outcomes under its stated method. Population interpretation requires an appropriate sampling design."
            if exact
            else "The resampling interval describes stability on the authenticated schedule, not population uncertainty or representativeness.",
        ),
        details=tuple(details),
        technical=report,
    )


def _render_markdown(
    report: dict[str, Any],
    *,
    explain: bool,
    evidence_signer: str,
    observations: list[dict[str, Any]],
) -> str:
    return render_report_markdown(
        _report_view(
            report, evidence_signer=evidence_signer, observations=observations
        ),
        include_details=explain,
    )


def _render_html(
    report: dict[str, Any],
    *,
    explain: bool,
    evidence_signer: str,
    observations: list[dict[str, Any]],
) -> str:
    del explain  # Every HTML report includes collapsible technical details.
    return render_report_html(
        _report_view(report, evidence_signer=evidence_signer, observations=observations)
    )


def _render_native_evidence(
    evidence_path: Path,
    *,
    html_path: Path | None = None,
    explain: bool = False,
    _view_sink: list[ReportView] | None = None,
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
            subjects = []
            identities = {}
            for role, label in (
                ("baseline", "Baseline artifact"),
                ("subject", "Candidate artifact"),
            ):
                identity, _ = _load_object_with_bytes(
                    snapshot_root / "inputs" / f"{role}.json",
                    label=f"{role} identity",
                    max_bytes=_MAX_REPORT_BYTES,
                )
                identities[role] = identity
                locator = identity.get("locator")
                display = (
                    locator
                    if isinstance(locator, str) and locator
                    else str(identity["digest"])
                )
                subjects.append((label, display))
            context, changes = _native_report_context(snapshot_root, report, identities)
            view = _report_view(
                report,
                evidence_signer=evidence_signer,
                observations=observations,
                subjects=tuple(subjects),
                context=context,
                changes=changes,
            )
            text = render_report_markdown(view, include_details=explain)
            rendered_html = render_report_html(view) if html_path is not None else None
            materialized_errors = snapshot.files.materialized_stability_errors(
                snapshot_root
            )
    except RuntimeError as exc:
        raise EvidenceReportError(str(exc)) from exc
    stability_errors = [*materialized_errors, *snapshot.stability_errors()]
    if stability_errors:
        raise EvidenceReportError("; ".join(stability_errors))
    if _view_sink is not None:
        _view_sink.append(view)
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


def render_evidence(
    evidence_path: Path,
    *,
    html_path: Path | None = None,
    explain: bool = False,
    markdown_path: Path | None = None,
    junit_path: Path | None = None,
) -> EvidenceReport | EvidenceReportV2:
    """Render either evidence family without replay or implicit receipt discovery."""
    from invarlock.captured_contracts import atomic_write, sha
    from invarlock.captured_reporting import (
        CapturedReportError,
        _load,
        _view,
        is_captured_manifest,
    )

    evidence = Path(evidence_path)
    if not evidence.is_dir() or evidence.is_symlink():
        raise EvidenceReportError("evidence must be a real directory")
    try:
        captured = is_captured_manifest(evidence)
    except CapturedReportError as exc:
        raise EvidenceReportError(
            f"evidence manifest schema failed: {exc}", exit_code=exc.exit_code
        ) from exc
    if not captured and markdown_path is None and junit_path is None:
        return _render_native_evidence(evidence, html_path=html_path, explain=explain)
    requested = {
        name: str(path)
        for name, path in (
            ("html", html_path),
            ("markdown", markdown_path),
            ("junit", junit_path),
        )
        if path is not None
    }
    payload: dict[str, Any] = {
        "format_version": "invarlock/evidence-report-v2",
        "kind": "captured" if captured else "runtime",
        "ok": False,
        "pack_manifest_digest": None,
        "requested_outputs": requested,
        "written_outputs": {},
        "failed_output": None,
        "errors": [],
    }
    try:
        resolved: set[Path] = set()
        for value in requested.values():
            destination = Path(value).absolute()
            if destination.name in {"", ".", ".."}:
                raise EvidenceReportError("report destination must name a file")
            canonical = destination.resolve()
            if canonical.is_relative_to(
                evidence.resolve()
            ) or destination.is_relative_to(evidence.absolute()):
                raise EvidenceReportError(
                    "report destination must remain outside the immutable evidence pack"
                )
            if any(
                canonical.is_relative_to(other) or other.is_relative_to(canonical)
                for other in resolved
            ):
                raise EvidenceReportError("report destinations collide")
            resolved.add(canonical)
            if destination.exists() or destination.is_symlink():
                raise EvidenceReportError(
                    f"report destination already exists: {destination}"
                )
            for parent in destination.parents:
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise EvidenceReportError(
                        "report destination parent must be a real directory"
                    )
        if captured:
            manifest, values, signer, _ = _load(evidence)
            view = _view(manifest, values, signer)
            manifest_digest = sha(canonical_json_bytes(manifest))
        else:
            views: list[ReportView] = []
            native = _render_native_evidence(
                evidence, explain=explain, _view_sink=views
            )
            view = views[0]
            manifest_digest = native.pack_manifest_digest
        payload["pack_manifest_digest"] = manifest_digest
        text = render_report_markdown(view, include_details=explain)
        rendered = {}
        if "html" in requested:
            rendered["html"] = render_report_html(view).encode("utf-8")
        if "markdown" in requested:
            rendered["markdown"] = text.encode("utf-8")
        if "junit" in requested:
            suite = Element(
                "testsuite",
                name="InvarLock recorded policy checks",
                tests=str(len(view.metrics)),
                failures=str(
                    sum(m.decision in {"fail", "regression"} for m in view.metrics)
                ),
                errors=str(
                    sum(m.decision == "insufficient_evidence" for m in view.metrics)
                ),
            )
            for metric in view.metrics:
                case = SubElement(
                    suite,
                    "testcase",
                    name=xml_text(metric.name),
                    classname=xml_text(metric.scope),
                )
                if metric.decision != "pass":
                    SubElement(
                        case,
                        "error"
                        if metric.decision == "insufficient_evidence"
                        else "failure",
                        message=xml_text(metric.explanation),
                    )
            rendered["junit"] = tostring(suite, encoding="utf-8", xml_declaration=True)
        for name, raw in rendered.items():
            payload["failed_output"] = name
            atomic_write(Path(requested[name]), raw)
            payload["written_outputs"][name] = requested[name]
        payload["failed_output"] = None
    except (OSError, ValueError, RuntimeError) as exc:
        payload["errors"] = [str(exc)[:1024]]
        raise EvidenceReportError(str(exc), payload=payload) from exc
    return EvidenceReportV2(
        text=text,
        kind=payload["kind"],
        pack_manifest_digest=manifest_digest,
        requested_outputs=requested,
        written_outputs=dict(payload["written_outputs"]),
    )


__all__ = [
    "EvidenceReport",
    "EvidenceReportV2",
    "EvidenceReportError",
    "render_evidence",
]
