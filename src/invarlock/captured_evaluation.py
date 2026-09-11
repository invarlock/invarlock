"""Captured evaluation transaction, independent of runtime execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from invarlock import captured_evidence_publication
from invarlock.captured_contracts import read_file, sha
from invarlock.captured_evidence_publication import (
    CapturedEvidenceError,
    publish_captured_evidence,
)
from invarlock.captured_normalization import (
    captured_comparison_id,
    captured_request_digest,
    normalize_captured_request,
)
from invarlock.core.evaluation_request import (
    CapturedEvaluationRequest,
    _reference_parts,
    _resolve_output_reference,
)
from invarlock.evaluation_comparison.capacity import (
    DEFAULT_MAX_BOOTSTRAP_DRAWS,
    check_missing_id_capacity,
)
from invarlock.evaluation_comparison.comparison import (
    _check_policy,
    check_bootstrap_budget,
    compare_runs,
    metric_summaries,
)
from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.adapters import _parse_run_bytes
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_json import parse_json_bytes


class CapturedEvaluationError(ValueError):
    """Expected validation, comparison, signing, or publication failure."""


@dataclass(frozen=True)
class CapturedEvaluationTransactionResult:
    evidence_path: Path
    comparison_id: str
    baseline_run_digest: str
    subject_run_digest: str
    policy_digest: str
    authentication: str
    policy_verdict: str
    pack_manifest_digest: str
    request_digest: str
    metric_summaries: tuple[dict[str, Any], ...] = field(
        default=(), repr=False, compare=False
    )

    def as_json(self) -> str:
        return canonical_json_bytes(
            {
                "format_version": "invarlock/evaluation-result-v2",
                "kind": "captured",
                "ok": True,
                "evidence": str(self.evidence_path),
                "comparison_id": self.comparison_id,
                "baseline_run_digest": self.baseline_run_digest,
                "subject_run_digest": self.subject_run_digest,
                "policy_digest": self.policy_digest,
                "authentication": self.authentication,
                "policy_verdict": self.policy_verdict,
                "decision": self.policy_verdict,
                "pack_manifest_digest": self.pack_manifest_digest,
                "request_digest": self.request_digest,
            }
        ).decode("utf-8")


@dataclass(frozen=True)
class CapturedEvaluationPreflightResult:
    """Structural captured checks, deliberately before scoring or publication."""

    execution_mode: str
    requested_authentication: str
    baseline_run_digest: str
    subject_run_digest: str
    policy_digest: str
    record_count: int
    required_bootstrap_draws: int
    missing_id_bytes: int
    checks: tuple[str, ...]
    request_digest: str
    output: str
    max_bootstrap_draws: int | None
    scope_count: int

    def as_json(self) -> str:
        return canonical_json_bytes(
            {
                "format_version": "invarlock/evaluation-preflight-v3",
                "kind": "captured",
                "ok": True,
                "execution_mode": self.execution_mode,
                "requested_authentication": self.requested_authentication,
                "baseline_run_digest": self.baseline_run_digest,
                "subject_run_digest": self.subject_run_digest,
                "policy_digest": self.policy_digest,
                "record_count": self.record_count,
                "required_bootstrap_draws": self.required_bootstrap_draws,
                "missing_id_bytes": self.missing_id_bytes,
                "checks": list(self.checks),
                "request_digest": self.request_digest,
                "output": self.output,
                "max_bootstrap_draws": self.max_bootstrap_draws,
                "scope_count": self.scope_count,
            }
        ).decode("utf-8")


def _json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = parse_json_bytes(
            raw,
            label=label,
        )
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise CapturedEvaluationError(f"{label} is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise CapturedEvaluationError(f"{label} must be a JSON object")
    return value


def _read_input(request: CapturedEvaluationRequest, path: Path) -> bytes:
    try:
        relative = path.relative_to(request.root).as_posix()
        _reference_parts(relative, label="captured input")
        return read_file(path, contracts.MAX_INPUT_BYTES)
    except (OSError, ValueError) as exc:
        raise CapturedEvaluationError(
            f"captured input could not be read safely: {exc}"
        ) from exc


def _run(request: CapturedEvaluationRequest, side: str) -> dict[str, Any]:
    source = getattr(request, side)
    try:
        return _parse_run_bytes(
            _read_input(request, source.path),
            adapter=source.adapter,
            source=dict(source.source) if source.source is not None else None,
            run_id=source.run_id,
            artifact_digest=source.artifact_digest,
            score_provenance=(
                dict(source.score_provenance)
                if source.score_provenance is not None
                else None
            ),
        )
    except EvaluationRecordsError as exc:
        raise CapturedEvaluationError(f"{side} run could not be loaded: {exc}") from exc


def _prepare_captured_request(
    request: CapturedEvaluationRequest,
    *,
    signing_key_path: Path | None,
    unsigned: bool = False,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    CapturedEvaluationPreflightResult,
]:
    """Check captured bindings and capacity without scoring, signing, or output."""
    if type(unsigned) is not bool:
        raise CapturedEvaluationError("unsigned must be a boolean")
    if max_bootstrap_draws is not None and (
        type(max_bootstrap_draws) is not int or max_bootstrap_draws < 0
    ):
        raise CapturedEvaluationError(
            "max_bootstrap_draws must be a non-negative integer"
        )
    try:
        if unsigned:
            if signing_key_path is not None:
                raise CapturedEvaluationError(
                    "unsigned mode cannot be combined with a signing key"
                )
        elif signing_key_path is None:
            raise CapturedEvaluationError(
                "captured evaluation requires an Ed25519 signing key"
            )
        _resolve_output_reference(
            request.root,
            request.evidence.relative_to(request.root).as_posix(),
            label="output.evidence",
        )
    except (OSError, ValueError) as exc:
        raise CapturedEvaluationError(str(exc)) from exc
    baseline = _run(request, "baseline")
    subject = _run(request, "subject")
    policy = _json(_read_input(request, request.policy), label="captured policy")
    authored: dict[str, Any] = {
        "format_version": request.format_version,
        "execution": {"mode": request.execution_mode},
        "comparison": {"policy": request.policy.relative_to(request.root).as_posix()},
        "output": {"evidence": request.evidence.relative_to(request.root).as_posix()},
    }
    for side in ("baseline", "subject"):
        source = getattr(request, side)
        spec = {
            "path": source.path.relative_to(request.root).as_posix(),
            "adapter": source.adapter,
        }
        for name in (
            "source",
            "run_id",
            "artifact_digest",
            "score_provenance",
            "expected_run_digest",
        ):
            value = getattr(source, name)
            if value is not None:
                spec[name] = (
                    dict(value) if name in {"source", "score_provenance"} else value
                )
        authored["comparison"][side] = spec
    try:
        normalized = normalize_captured_request(
            authored, baseline=baseline, subject=subject, policy=policy
        )
        _check_policy(policy)
        for metric in policy["metrics"]:
            if metric["kind"] == "recorded":
                for run in (baseline, subject):
                    if (
                        run["score_provenance"].get(metric["score_key"])
                        != metric["accepted_provenance"]
                    ):
                        raise EvaluationRecordsError(
                            f"metric {metric['name']}: scorer provenance differs from approved policy"
                        )
        baseline_rows = {row["id"]: row for row in baseline["records"]}
        subject_rows = {row["id"]: row for row in subject["records"]}
        if baseline_rows.keys() != subject_rows.keys():
            raise EvaluationRecordsError(
                "baseline/subject record IDs differ; export the complete paired schedule"
            )
        pairs = []
        for record_id in sorted(baseline_rows):
            left, right = baseline_rows[record_id], subject_rows[record_id]
            for key in ("input", "expected", "metadata"):
                if canonical_json_bytes(left[key]) != canonical_json_bytes(right[key]):
                    raise EvaluationRecordsError(
                        f"record {record_id}: {key} changed between runs"
                    )
            pairs.append((left, right))
        scopes = []
        for subset in [{"where": {}}, *policy["slices"]]:
            selected = [
                pair
                for pair in pairs
                if all(
                    pair[0]["metadata"].get(k) == v for k, v in subset["where"].items()
                )
            ]
            scopes.append((selected,))
        required_draws = check_bootstrap_budget(
            policy,
            sum(len(selected[0]) for selected in scopes),
            max_bootstrap_draws,
        )
        capacity_scopes = [
            (str(index), selected[0]) for index, selected in enumerate(scopes)
        ]
        missing_id_bytes = check_missing_id_capacity(
            capacity_scopes, policy["metrics"], byte_limit=contracts.MAX_INPUT_BYTES
        )
    except (
        EvaluationRecordsError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise CapturedEvaluationError(f"captured preflight failed: {exc}") from exc
    if not unsigned:
        assert signing_key_path is not None
        try:
            captured_evidence_publication._private_key(signing_key_path)
        except (OSError, ValueError) as exc:
            raise CapturedEvaluationError(str(exc)) from exc
    run_digests = {
        side: normalized["comparison"][side]["run_digest"]
        for side in ("baseline", "subject")
    }
    result = CapturedEvaluationPreflightResult(
        execution_mode="captured",
        requested_authentication="unsigned_local" if unsigned else "signed",
        baseline_run_digest=run_digests["baseline"],
        subject_run_digest=run_digests["subject"],
        policy_digest=normalized["comparison"]["policy_digest"],
        record_count=len(baseline["records"]),
        required_bootstrap_draws=required_draws,
        missing_id_bytes=missing_id_bytes,
        checks=(
            "request",
            "run_bindings",
            "policy",
            "paired_capacity",
            "output_destination",
            "signing_mode",
        ),
        request_digest=captured_request_digest(normalized),
        output=str(request.evidence),
        max_bootstrap_draws=max_bootstrap_draws,
        scope_count=len(scopes),
    )
    return baseline, subject, policy, normalized, result


def preflight_captured_request(
    request: CapturedEvaluationRequest,
    *,
    signing_key_path: Path | None = None,
    unsigned: bool = False,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> CapturedEvaluationPreflightResult:
    return _prepare_captured_request(
        request,
        signing_key_path=signing_key_path,
        unsigned=unsigned,
        max_bootstrap_draws=max_bootstrap_draws,
    )[4]


def evaluate_captured_request(
    request: CapturedEvaluationRequest,
    *,
    signing_key_path: Path | None = None,
    unsigned: bool = False,
    max_bootstrap_draws: int | None = DEFAULT_MAX_BOOTSTRAP_DRAWS,
) -> CapturedEvaluationTransactionResult:
    """Load, compare, bind, and publish one captured request."""
    baseline, subject, policy, normalized_request, prepared = _prepare_captured_request(
        request,
        signing_key_path=signing_key_path,
        unsigned=unsigned,
        max_bootstrap_draws=max_bootstrap_draws,
    )
    try:
        comparison = compare_runs(
            baseline, subject, policy, max_bootstrap_draws=max_bootstrap_draws
        )
    except (
        EvaluationRecordsError,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise CapturedEvaluationError(f"captured comparison failed: {exc}") from exc
    request_digest = prepared.request_digest
    try:
        publish_captured_evidence(
            request.evidence,
            baseline=baseline,
            subject=subject,
            policy=policy,
            comparison=comparison,
            request_digest=request_digest,
            normalized_request=normalized_request,
            signing_key_path=signing_key_path,
            unsigned=unsigned,
        )
    except CapturedEvidenceError as exc:
        raise CapturedEvaluationError(str(exc)) from exc
    return CapturedEvaluationTransactionResult(
        evidence_path=request.evidence,
        comparison_id=captured_comparison_id(
            request_digest=request_digest,
            baseline_run_digest=prepared.baseline_run_digest,
            subject_run_digest=prepared.subject_run_digest,
            policy_digest=prepared.policy_digest,
        ),
        baseline_run_digest=prepared.baseline_run_digest,
        subject_run_digest=prepared.subject_run_digest,
        policy_digest=digest(policy),
        authentication="unsigned_local" if unsigned else "signed",
        policy_verdict=comparison["decision"],
        pack_manifest_digest=sha(
            read_file(request.evidence / "manifest.json", 64 * 1024)
        ),
        request_digest=request_digest,
        metric_summaries=metric_summaries(comparison),
    )


__all__ = [
    "CapturedEvaluationError",
    "CapturedEvaluationPreflightResult",
    "CapturedEvaluationTransactionResult",
    "evaluate_captured_request",
    "preflight_captured_request",
]
