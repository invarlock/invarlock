from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

from invarlock.reporting.report_summary import compute_console_validation_block

ReportExportFormat = Literal[
    "mlflow-tags",
    "model-card-md",
    "release-review-md",
]

_UNKNOWN = "unknown"
_REPORT_LOCAL_GATE_KEYS = frozenset(
    {
        "guard_metric_impact_acceptable",
        "guard_warning_policy_acceptable",
        "invariants_pass",
        "moe_identity_ok",
        "preview_final_drift_acceptable",
        "primary_metric_acceptable",
        "primary_metric_tail_acceptable",
        "rmt_stable",
        "spectral_stable",
    }
)


class VerifyResultMismatchError(ValueError):
    """Raised when a verifier JSON result is for a different report."""


class VerifyResultValidationError(VerifyResultMismatchError):
    """Raised when an external verifier result is malformed or unbound.

    A standalone ``verify --json`` document is not a signature.  It can still
    be useful handoff metadata, but only after its structure and the receipt
    binding to the exact exported report bytes have been checked.
    """


@dataclass(frozen=True)
class ReportExportContext:
    report_path: Path
    report_sha256: str
    status: Literal["report_local_pass", "report_local_fail", "receipt_bound_untrusted"]
    report_local_status: Literal["pass", "fail"]
    verifier_status: Literal["not_provided", "receipt_bound_untrusted"]
    verifier_outcome: Literal["pass", "fail", "not_provided"]
    receipt_status: Literal["not_provided", "bound_unsigned"]
    verifier_reason: str
    runtime_provenance_status: str
    policy_profile: str
    baseline: str
    subject: str
    run_id: str
    schema_version: str
    primary_metric: str
    primary_metric_kind: str
    primary_metric_final: str
    primary_metric_ratio: str
    edit_name: str
    policy_digest: str
    failed_gate_count: int
    report_url: str | None = None
    evidence_url: str | None = None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_text(value: Any, *, default: str = _UNKNOWN) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _first_text(*values: Any, default: str = _UNKNOWN) -> str:
    for value in values:
        text = _clean_text(value, default="")
        if text:
            return text
    return default


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return f"{number:.6g}"
    return _clean_text(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_report_status(report: Mapping[str, Any]) -> Literal["pass", "fail"]:
    block = compute_console_validation_block(dict(report))
    return "pass" if bool(block.get("overall_pass")) else "fail"


def _derive_policy_profile(
    report: Mapping[str, Any], override: str | None = None
) -> str:
    if override:
        return _clean_text(override)
    meta = _as_mapping(report.get("meta"))
    context = _as_mapping(report.get("context"))
    provenance = _as_mapping(report.get("provenance"))
    window_plan = _as_mapping(provenance.get("window_plan"))
    return _first_text(
        report.get("policy_profile"),
        report.get("profile"),
        meta.get("profile"),
        context.get("profile"),
        window_plan.get("profile"),
    )


def _derive_subject(report: Mapping[str, Any]) -> str:
    meta = _as_mapping(report.get("meta"))
    provenance = _as_mapping(report.get("provenance"))
    edited = _as_mapping(provenance.get("edited"))
    return _first_text(
        edited.get("model_id"),
        edited.get("checkpoint"),
        meta.get("model_id"),
        edited.get("run_id"),
        report.get("run_id"),
    )


def _derive_baseline(report: Mapping[str, Any]) -> str:
    baseline_ref = _as_mapping(report.get("baseline_ref"))
    provenance = _as_mapping(report.get("provenance"))
    baseline = _as_mapping(provenance.get("baseline"))
    return _first_text(
        baseline_ref.get("model_id"),
        baseline.get("model_id"),
        baseline_ref.get("run_id"),
        baseline.get("run_id"),
        baseline.get("report_path"),
    )


def _derive_edit_name(report: Mapping[str, Any]) -> str:
    edit = _as_mapping(report.get("edit"))
    return _first_text(report.get("edit_name"), edit.get("name"))


def _derive_policy_digest(report: Mapping[str, Any]) -> str:
    policy_provenance = _as_mapping(report.get("policy_provenance"))
    auto = _as_mapping(report.get("auto"))
    policy_digest = _as_mapping(report.get("policy_digest"))
    return _first_text(
        policy_provenance.get("policy_digest"),
        auto.get("policy_digest"),
        policy_digest.get("thresholds_hash"),
        policy_digest.get("policy_version"),
    )


def _derive_primary_metric(report: Mapping[str, Any]) -> str:
    primary_metric = _as_mapping(report.get("primary_metric"))
    if not primary_metric:
        return _UNKNOWN
    parts = [_clean_text(primary_metric.get("kind"), default="primary")]
    for key in (
        "preview",
        "final",
        "ratio_vs_baseline",
        "delta_vs_baseline_pp",
    ):
        value = primary_metric.get(key)
        if value is not None:
            parts.append(f"{key}={_format_scalar(value)}")
    return " ".join(parts)


def _derive_primary_metric_field(report: Mapping[str, Any], key: str) -> str:
    primary_metric = _as_mapping(report.get("primary_metric"))
    if key == "kind":
        return _clean_text(primary_metric.get("kind"))
    value = primary_metric.get(key)
    return _format_scalar(value) if value is not None else _UNKNOWN


def _failed_gate_count(report: Mapping[str, Any]) -> int:
    validation = _as_mapping(report.get("validation"))
    return sum(
        1
        for key, value in validation.items()
        if key in _REPORT_LOCAL_GATE_KEYS and isinstance(value, bool) and not value
    )


_VERIFY_RESULT_REASONS = frozenset({"ok", "policy_fail", "malformed"})
_VERIFY_RESULT_FORMAT = "verify-v1"
_VERIFY_RECEIPT_FORMAT = "invarlock.verify-receipt.v1"


def _verify_result_error(message: str) -> NoReturn:
    raise VerifyResultValidationError(message)


def _require_json_boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    _verify_result_error(f"{label} must be a JSON boolean.")


def _require_json_string(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _verify_result_error(f"{label} must be a JSON string.")
    if not allow_empty and not value.strip():
        _verify_result_error(f"{label} must not be empty.")
    return value


def _require_verify_reason(value: Any, *, label: str) -> str:
    reason = _require_json_string(value, label=label)
    if reason not in _VERIFY_RESULT_REASONS:
        _verify_result_error(
            f"{label} must be one of: {', '.join(sorted(_VERIFY_RESULT_REASONS))}."
        )
    return reason


def _ensure_finite_json_value(value: Any, *, label: str) -> None:
    """Reject values that cannot occur in a canonical finite JSON receipt."""

    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _verify_result_error(f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _verify_result_error(f"{label} contains a non-string object key.")
            _ensure_finite_json_value(item, label=f"{label}.{key}")
        return
    _verify_result_error(f"{label} contains a non-JSON value.")


def _validate_verify_result_item(item: Any, *, index: int) -> Mapping[str, Any]:
    label = f"Verify result item {index}"
    if not isinstance(item, Mapping):
        _verify_result_error(f"{label} must be a JSON object with an id.")
    item_id = _require_json_string(item.get("id"), label=f"{label}.id")
    _ = item_id
    schema_version = _require_json_string(
        item.get("schema_version"), label=f"{label}.schema_version"
    )
    if schema_version != "v1":
        _verify_result_error(f"{label}.schema_version must be 'v1'.")
    _require_json_string(item.get("kind"), label=f"{label}.kind", allow_empty=True)
    ok = _require_json_boolean(item.get("ok"), label=f"{label}.ok")
    reason = _require_verify_reason(item.get("reason"), label=f"{label}.reason")
    if (ok and reason != "ok") or (not ok and reason == "ok"):
        _verify_result_error(f"{label}.ok and {label}.reason are inconsistent.")
    if "ci" not in item:
        _verify_result_error(f"{label}.ci is required.")
    ci = item["ci"]
    if ci is not None:
        if not isinstance(ci, list) or len(ci) != 2:
            _verify_result_error(f"{label}.ci must be null or a two-item JSON array.")
        for ci_index, value in enumerate(ci):
            if isinstance(value, bool) or not isinstance(value, int | float):
                _verify_result_error(
                    f"{label}.ci[{ci_index}] must be a finite JSON number."
                )
            if not math.isfinite(float(value)):
                _verify_result_error(
                    f"{label}.ci[{ci_index}] must be a finite JSON number."
                )
    return item


def _validated_verify_result_item(
    verify_result: object | None,
    report_path: Path,
) -> Mapping[str, Any] | None:
    if verify_result is None:
        return None
    if not isinstance(verify_result, Mapping):
        _verify_result_error("Verify result must decode to a JSON object.")
    _ensure_finite_json_value(verify_result, label="Verify result")
    format_version = _require_json_string(
        verify_result.get("format_version"), label="Verify result.format_version"
    )
    if format_version != _VERIFY_RESULT_FORMAT:
        _verify_result_error(
            f"Verify result.format_version must be '{_VERIFY_RESULT_FORMAT}'."
        )
    summary = verify_result.get("summary")
    if not isinstance(summary, Mapping):
        _verify_result_error("Verify result.summary must be a JSON object.")
    _require_json_boolean(summary.get("ok"), label="Verify result.summary.ok")
    _require_verify_reason(summary.get("reason"), label="Verify result.summary.reason")
    results = verify_result.get("results")
    if not isinstance(results, list) or not results:
        raise VerifyResultMismatchError(
            "Verify result must contain a non-empty results list for evaluation "
            f"report {report_path.resolve()}."
        )

    resolved = str(report_path.resolve())
    matching_items: list[Mapping[str, Any]] = []
    mismatched_ids: list[str] = []
    for index, raw_item in enumerate(results):
        item = _validate_verify_result_item(raw_item, index=index)
        item_id = str(item["id"])
        try:
            item_path = str(Path(item_id).expanduser().resolve())
        except (OSError, RuntimeError) as exc:
            _verify_result_error(
                f"Verify result item {index}.id cannot be resolved: {exc}"
            )
        if item_path == resolved:
            matching_items.append(item)
        else:
            mismatched_ids.append(item_id)
    if len(matching_items) > 1:
        _verify_result_error(
            "Verify result must contain exactly one item for evaluation report "
            f"{resolved}; found {len(matching_items)}."
        )
    if matching_items:
        return matching_items[0]
    preview = ", ".join(mismatched_ids[:3])
    suffix = "" if len(mismatched_ids) <= 3 else ", ..."
    raise VerifyResultMismatchError(
        "Verify result does not contain an item for evaluation report "
        f"{resolved}. Found item id(s): {preview}{suffix}"
    )


def _validate_receipt_binding(
    item: Mapping[str, Any],
    *,
    report_sha256: str,
) -> tuple[Literal["pass", "fail"], str, str]:
    verification = item.get("verification")
    if not isinstance(verification, Mapping):
        _verify_result_error("Matching verify result item must include verification.")
    receipt = verification.get("receipt")
    if not isinstance(receipt, Mapping):
        _verify_result_error(
            "Matching verify result item must include a verification.receipt."
        )
    receipt_format = _require_json_string(
        receipt.get("format_version"), label="Verify receipt.format_version"
    )
    if receipt_format != _VERIFY_RECEIPT_FORMAT:
        _verify_result_error(
            f"Verify receipt.format_version must be '{_VERIFY_RECEIPT_FORMAT}'."
        )
    signed = _require_json_boolean(receipt.get("signed"), label="Verify receipt.signed")
    if signed:
        _verify_result_error(
            "Verify receipt declares signed=true, but this export cannot authenticate "
            "a standalone signature claim."
        )
    receipt_digest = _require_json_string(
        receipt.get("subject_report_sha256"),
        label="Verify receipt.subject_report_sha256",
    )
    if (
        len(receipt_digest) != 64
        or receipt_digest.lower() != receipt_digest
        or any(character not in "0123456789abcdef" for character in receipt_digest)
    ):
        _verify_result_error(
            "Verify receipt.subject_report_sha256 must be a lowercase SHA-256 hex digest."
        )
    if receipt_digest != report_sha256:
        _verify_result_error(
            "Verify receipt.subject_report_sha256 does not bind to the exact "
            "evaluation report bytes being exported."
        )
    ok = _require_json_boolean(item.get("ok"), label="Matching verify result item.ok")
    reason = _require_verify_reason(
        item.get("reason"), label="Matching verify result item.reason"
    )
    runtime = verification.get("runtime_provenance")
    if runtime is not None and not isinstance(runtime, Mapping):
        _verify_result_error(
            "Matching verify result item.verification.runtime_provenance must be an object."
        )
    runtime_status = _clean_text(
        runtime.get("status") if isinstance(runtime, Mapping) else None
    )
    return ("pass" if ok else "fail"), reason, runtime_status


def _verifier_fields(
    verify_result: Mapping[str, Any] | None,
    report_path: Path,
    *,
    report_sha256: str,
) -> tuple[
    Literal["not_provided", "receipt_bound_untrusted"],
    Literal["pass", "fail", "not_provided"],
    Literal["not_provided", "bound_unsigned"],
    str,
    str,
]:
    item = _validated_verify_result_item(verify_result, report_path)
    if item is None:
        return "not_provided", "not_provided", "not_provided", _UNKNOWN, _UNKNOWN
    outcome, reason, runtime_status = _validate_receipt_binding(
        item,
        report_sha256=report_sha256,
    )
    # Current verify receipts explicitly set signed=false.  Binding proves only
    # that this supplied JSON names these bytes; it does not authenticate who
    # produced the JSON or independently prove the claimed verifier outcome.
    return "receipt_bound_untrusted", outcome, "bound_unsigned", reason, runtime_status


def build_report_export_context(
    report_path: str | Path,
    report: Mapping[str, Any],
    *,
    policy_profile: str | None = None,
    report_url: str | None = None,
    evidence_url: str | None = None,
    verify_result: Mapping[str, Any] | None = None,
    report_bytes: bytes | None = None,
) -> ReportExportContext:
    resolved = Path(report_path).resolve()
    report_sha256 = (
        hashlib.sha256(report_bytes).hexdigest()
        if report_bytes is not None
        else _sha256_file(resolved)
    )
    (
        verifier_status,
        verifier_outcome,
        receipt_status,
        verifier_reason,
        runtime_status,
    ) = _verifier_fields(
        verify_result,
        resolved,
        report_sha256=report_sha256,
    )
    report_status = derive_report_status(report)
    export_status: Literal[
        "report_local_pass", "report_local_fail", "receipt_bound_untrusted"
    ]
    if verifier_status == "not_provided":
        export_status = (
            "report_local_pass" if report_status == "pass" else "report_local_fail"
        )
    else:
        export_status = "receipt_bound_untrusted"
    return ReportExportContext(
        report_path=resolved,
        report_sha256=report_sha256,
        status=export_status,
        report_local_status=report_status,
        verifier_status=verifier_status,
        verifier_outcome=verifier_outcome,
        receipt_status=receipt_status,
        verifier_reason=verifier_reason,
        runtime_provenance_status=runtime_status,
        policy_profile=_derive_policy_profile(report, policy_profile),
        baseline=_derive_baseline(report),
        subject=_derive_subject(report),
        run_id=_clean_text(report.get("run_id")),
        schema_version=_clean_text(report.get("schema_version")),
        primary_metric=_derive_primary_metric(report),
        primary_metric_kind=_derive_primary_metric_field(report, "kind"),
        primary_metric_final=_derive_primary_metric_field(report, "final"),
        primary_metric_ratio=_derive_primary_metric_field(report, "ratio_vs_baseline"),
        edit_name=_derive_edit_name(report),
        policy_digest=_derive_policy_digest(report),
        failed_gate_count=_failed_gate_count(report),
        report_url=_clean_text(report_url, default="") or None,
        evidence_url=_clean_text(evidence_url, default="") or None,
    )


def render_mlflow_tags_export(context: ReportExportContext) -> dict[str, Any]:
    tags = {
        "invarlock.status": context.status,
        "invarlock.report_local_status": context.report_local_status,
        "invarlock.report_sha256": context.report_sha256,
        "invarlock.policy_profile": context.policy_profile,
        "invarlock.baseline": context.baseline,
        "invarlock.subject": context.subject,
        "invarlock.edit_name": context.edit_name,
        "invarlock.verifier_status": context.verifier_status,
        "invarlock.verifier_outcome": context.verifier_outcome,
        "invarlock.receipt_status": context.receipt_status,
        "invarlock.verifier_reason": context.verifier_reason,
        "invarlock.failed_gate_count": str(context.failed_gate_count),
        "invarlock.primary_metric_kind": context.primary_metric_kind,
        "invarlock.primary_metric_final": context.primary_metric_final,
        "invarlock.primary_metric_ratio_vs_baseline": context.primary_metric_ratio,
        "invarlock.run_id": context.run_id,
        "invarlock.schema_version": context.schema_version,
    }
    if context.policy_digest != _UNKNOWN:
        tags["invarlock.policy_digest"] = context.policy_digest
    if context.runtime_provenance_status != _UNKNOWN:
        tags["invarlock.runtime_provenance_status"] = context.runtime_provenance_status
    if context.report_url:
        tags["invarlock.report_url"] = context.report_url
    if context.evidence_url:
        tags["invarlock.evidence_url"] = context.evidence_url
    return {
        "schema_version": "invarlock.mlflow-tags.v1",
        "artifact": {
            "path": str(context.report_path),
            "artifact_path": "invarlock",
        },
        "tags": tags,
    }


def _link_or_code(label: str, url: str | None) -> str:
    if url:
        return f"[{label}]({url})"
    return _inline_code(label)


def _inline_code(value: str) -> str:
    text = _clean_text(value).replace("`", "'")
    return f"`{text}`"


def _markdown_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", r"\|")


def _markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    lines.extend(
        f"| {_markdown_cell(label)} | {_markdown_cell(value)} |"
        for label, value in rows
    )
    return "\n".join(lines)


def render_model_card_evidence_block(context: ReportExportContext) -> str:
    status = context.status.upper()
    report_ref = context.report_url or context.report_path.name
    rows = [
        ("Status", status),
        ("Report-local Gate Status", context.report_local_status.upper()),
        ("Verifier Status", _inline_code(context.verifier_status)),
        ("Verifier Outcome", _inline_code(context.verifier_outcome)),
        ("Verifier Receipt", _inline_code(context.receipt_status)),
        ("Verifier Reason", _inline_code(context.verifier_reason)),
        ("Runtime Provenance", _inline_code(context.runtime_provenance_status)),
        ("Report SHA-256", _inline_code(context.report_sha256)),
        ("Policy Profile", _inline_code(context.policy_profile)),
        ("Baseline", _inline_code(context.baseline)),
        ("Subject", _inline_code(context.subject)),
        ("Primary Metric", _inline_code(context.primary_metric)),
        ("Report", _link_or_code(report_ref, context.report_url)),
    ]
    if context.evidence_url:
        rows.append(
            ("Evidence Pack", _link_or_code("evidence pack", context.evidence_url))
        )
    return "\n".join(
        [
            "## InvarLock Evidence",
            "",
            _markdown_table(rows),
            "",
            (
                "This block summarizes report-local gates and any supplied "
                "receipt-bound verifier outcome. An unsigned receipt is not "
                "independent verification or deployment approval."
            ),
            "",
        ]
    )


def _validation_checklist(report: Mapping[str, Any]) -> list[str]:
    validation = _as_mapping(report.get("validation"))
    if not validation:
        return ["- [ ] No validation gates were present in the report."]
    lines: list[str] = []
    for key in sorted(validation):
        value = validation.get(key)
        if key not in _REPORT_LOCAL_GATE_KEYS or not isinstance(value, bool):
            continue
        marker = "x" if value else " "
        lines.append(f"- [{marker}] `{key}`")
    return lines or ["- [ ] No boolean report-local gates were present in the report."]


def render_release_review_packet(
    context: ReportExportContext, report: Mapping[str, Any]
) -> str:
    report_ref = context.report_url or str(context.report_path)
    evidence_ref = context.evidence_url or "Not provided"
    lines = [
        "# InvarLock Release Review",
        "",
        "## Decision Summary",
        "",
        f"- Status: **{context.status.upper()}**",
        f"- Report-local gate status: `{context.report_local_status}`",
        f"- Run ID: `{context.run_id}`",
        f"- Policy profile: `{context.policy_profile}`",
        f"- Verifier status: `{context.verifier_status}`",
        f"- Verifier outcome: `{context.verifier_outcome}`",
        f"- Verifier receipt: `{context.receipt_status}`",
        f"- Verifier reason: `{context.verifier_reason}`",
        f"- Runtime provenance: `{context.runtime_provenance_status}`",
        f"- Report SHA-256: `{context.report_sha256}`",
        "",
        "## What Changed",
        "",
        f"- Baseline: `{context.baseline}`",
        f"- Subject: `{context.subject}`",
        f"- Edit: `{context.edit_name}`",
        f"- Primary metric: `{context.primary_metric}`",
        "",
        "## Evidence",
        "",
        f"- Evaluation report: {report_ref}",
        f"- Evidence pack: {evidence_ref}",
        f"- Policy digest: `{context.policy_digest}`",
        f"- Failed report-local validation gates: `{context.failed_gate_count}`",
        "",
        "## Gate Checklist",
        "",
        *_validation_checklist(report),
        "",
        "## Evidence Checklist",
        "",
        "- [ ] Verify the report SHA-256 against the shipped artifact.",
        "- [ ] Confirm baseline and subject identities match the release request.",
        "- [ ] Review failed or missing gates before approving release.",
        "- [ ] Attach the report, rendered HTML, and evidence pack to the release record.",
        "- [ ] Treat this packet as a checklist wrapper, not an approval substitute.",
        "",
    ]
    return "\n".join(lines)


def render_report_export(
    export_format: str,
    context: ReportExportContext,
    report: Mapping[str, Any],
) -> str | dict[str, Any]:
    if export_format == "mlflow-tags":
        return render_mlflow_tags_export(context)
    if export_format == "model-card-md":
        return render_model_card_evidence_block(context)
    if export_format == "release-review-md":
        return render_release_review_packet(context, report)
    raise ValueError(
        "Unsupported export format. Expected one of: "
        "mlflow-tags, model-card-md, release-review-md."
    )


def serialize_report_export(exported: str | dict[str, Any]) -> str:
    if isinstance(exported, str):
        return exported if exported.endswith("\n") else exported + "\n"
    return json.dumps(exported, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "ReportExportContext",
    "ReportExportFormat",
    "VerifyResultMismatchError",
    "VerifyResultValidationError",
    "build_report_export_context",
    "derive_report_status",
    "render_mlflow_tags_export",
    "render_model_card_evidence_block",
    "render_release_review_packet",
    "render_report_export",
    "serialize_report_export",
]
