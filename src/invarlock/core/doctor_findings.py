from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .report_inputs import ReportInputError, load_report_input_json

DATASET_SPLIT_FALLBACK_WARNING = "Dataset split was inferred via fallback; set dataset.split explicitly to avoid drift."


@dataclass(frozen=True)
class DoctorFinding:
    code: str
    severity: str
    message: str
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            **self.extra,
        }


class DoctorAccumulator:
    def __init__(self) -> None:
        self._findings: list[dict[str, object]] = []
        self._had_error = False

    def add(self, code: str, severity: str, message: str, **extra: object) -> None:
        finding = DoctorFinding(
            code=code, severity=severity, message=message, extra=extra
        )
        self._findings.append(finding.as_dict())
        if severity == "error":
            self._had_error = True

    def extend(
        self, findings: list[DoctorFinding], *, mark_error: bool | None = None
    ) -> None:
        for finding in findings:
            self._findings.append(finding.as_dict())
            if finding.severity == "error":
                self._had_error = True
        if mark_error is True:
            self._had_error = True

    @property
    def findings(self) -> list[dict[str, object]]:
        return self._findings

    @property
    def had_error(self) -> bool:
        return self._had_error

    def sort(self) -> None:
        order = {"error": 0, "warning": 1, "note": 2}
        self._findings.sort(
            key=lambda item: (
                order.get(str(item.get("severity")), 9),
                str(item.get("code", "Z999")),
            )
        )

    def summary(self) -> dict[str, int]:
        return {
            "errors": sum(
                1 for finding in self._findings if finding.get("severity") == "error"
            ),
            "warnings": sum(
                1 for finding in self._findings if finding.get("severity") == "warning"
            ),
            "notes": sum(
                1 for finding in self._findings if finding.get("severity") == "note"
            ),
        }


def load_explicit_report_input(
    path_value: str | None,
    *,
    label: str,
    field: str,
) -> tuple[Path | None, dict[str, Any] | None, list[DoctorFinding], bool]:
    """Load a report JSON input for doctor flows."""

    if not path_value:
        return None, None, [], False

    try:
        resolved, payload = load_report_input_json(path_value)
    except ReportInputError as exc:
        finding = DoctorFinding(
            code="D014",
            severity="error",
            message=_format_report_input_error(label=label, exc=exc),
            extra={"field": field},
        )
        return exc.path, None, [finding], True
    return resolved, payload, [], False


def build_cross_check_findings(
    baseline_report: str | None,
    subject_report: str | None,
    *,
    cfg_metric_kind: str | None,
    strict: bool,
    profile: str | None,
) -> tuple[list[DoctorFinding], bool]:
    """Perform baseline vs subject cross-checks and return findings."""

    findings: list[DoctorFinding] = []
    _, bdata, baseline_errors, baseline_invalid = load_explicit_report_input(
        baseline_report,
        label="Baseline",
        field="baseline_report",
    )
    _, sdata, subject_errors, subject_invalid = load_explicit_report_input(
        subject_report,
        label="Subject",
        field="subject_report",
    )
    findings.extend(baseline_errors)
    findings.extend(subject_errors)
    had_error = baseline_invalid or subject_invalid
    if bdata is None or sdata is None:
        return findings, had_error

    bprov = _as_dict(bdata.get("provenance"))
    sprov = _as_dict(sdata.get("provenance"))
    bdig = _as_dict(bprov.get("provider_digest"))
    sdig = _as_dict(sprov.get("provider_digest"))

    btok = bdig.get("tokenizer_sha256")
    stok = sdig.get("tokenizer_sha256")
    if (
        isinstance(btok, str)
        and isinstance(stok, str)
        and btok
        and stok
        and btok != stok
    ):
        findings.append(
            DoctorFinding(
                code="D009",
                severity="warning",
                message=(
                    "tokenizer digests differ between baseline and subject; "
                    "run will abort in ci/release (E002)."
                ),
                extra={"field": "provenance.provider_digest.tokenizer_sha256"},
            )
        )

    bmask = bdig.get("masking_sha256")
    smask = sdig.get("masking_sha256")
    is_mlm = "ppl_mlm" in {
        _as_lower(_primary_metric_kind(bdata)),
        _as_lower(_primary_metric_kind(sdata)),
        _as_lower(cfg_metric_kind),
    }
    if (
        is_mlm
        and isinstance(btok, str)
        and isinstance(stok, str)
        and btok
        and stok
        and btok == stok
        and (not bmask or not smask)
    ):
        findings.append(
            DoctorFinding(
                code="D010",
                severity="warning",
                message=(
                    "ppl_mlm with matching tokenizer but missing masking digests; "
                    "ci/release may abort on mask parity."
                ),
                extra={
                    "baseline_has_mask": bool(bmask),
                    "subject_has_mask": bool(smask),
                },
            )
        )

    bsplit = bprov.get("dataset_split")
    ssplit = sprov.get("dataset_split")
    if (
        isinstance(bsplit, str)
        and isinstance(ssplit, str)
        and bsplit
        and ssplit
        and bsplit != ssplit
    ):
        severity = "error" if strict else "warning"
        findings.append(
            DoctorFinding(
                code="D011",
                severity=severity,
                message=f"dataset split mismatch (baseline={bsplit}, subject={ssplit})",
                extra={
                    "field": "provenance.dataset_split",
                    "baseline": bsplit,
                    "subject": ssplit,
                },
            )
        )
        if severity == "error":
            had_error = True

    spm = _as_dict(_as_dict(sdata.get("metrics")).get("primary_metric"))
    pm_kind = _as_lower(spm.get("kind"))
    if pm_kind in {"accuracy", "vqa_accuracy"}:
        estimated = bool(spm.get("estimated"))
        counts_source = _as_lower(spm.get("counts_source"))
        if estimated or counts_source == "pseudo_config":
            report_profile = _as_lower(_as_dict(sdata.get("meta")).get("profile"))
            profile_flag = _as_lower(profile)
            effective_profile = profile_flag or report_profile or "dev"
            severity = "warning" if effective_profile == "dev" else "error"
            findings.append(
                DoctorFinding(
                    code="D012",
                    severity=severity,
                    message=(
                        "accuracy primary metric uses pseudo/estimated counts; "
                        "use labeled preset for measured accuracy."
                    ),
                    extra={"field": "metrics.primary_metric"},
                )
            )
            if severity == "error":
                had_error = True

    return findings, had_error


def build_split_fallback_findings(
    report_data: dict[str, Any] | None,
) -> list[DoctorFinding]:
    prov = report_data.get("provenance", {}) if isinstance(report_data, dict) else {}
    if isinstance(prov, dict) and prov.get("split_fallback"):
        return [
            DoctorFinding(
                code="D003",
                severity="warning",
                message=(
                    "dataset split fallback was used. "
                    "Set dataset.provider.hf_dataset.split explicitly."
                ),
            )
        ]
    return []


def build_tiny_relax_finding(
    *,
    subject_report: dict[str, Any] | None,
    baseline_report: dict[str, Any] | None,
    env_enabled: bool,
) -> DoctorFinding | None:
    subject_tiny = _report_tiny_relax(subject_report)
    baseline_tiny = _report_tiny_relax(baseline_report)
    if env_enabled or subject_tiny or baseline_tiny:
        return DoctorFinding(
            code="D013",
            severity="note",
            message=(
                "tiny relax (dev) active; gates widened and drift/overhead may be informational."
            ),
            extra={"field": "auto.tiny_relax"},
        )
    return None


def build_provider_kind_findings(
    provider_cfg: object,
) -> tuple[list[DoctorFinding], bool]:
    supported_providers = {"wikitext2", "hf_text", "synthetic", "local_jsonl"}
    bad_kind: str | None = None

    if isinstance(provider_cfg, dict):
        kind = str(provider_cfg.get("kind", "")).strip()
        if not kind or kind not in supported_providers:
            bad_kind = kind or ""
    elif isinstance(provider_cfg, str):
        if provider_cfg not in supported_providers:
            bad_kind = provider_cfg
    else:
        kind = str(_mapping_get(provider_cfg, "kind") or "").strip()
        if not kind or kind not in supported_providers:
            bad_kind = kind or ""

    if not bad_kind:
        return [], False

    return (
        [
            DoctorFinding(
                code="D001",
                severity="error",
                message=(
                    f'dataset.provider.kind "{bad_kind}" is not supported. '
                    "Use one of: wikitext2 | hf_text | synthetic | local_jsonl."
                ),
                extra={
                    "field": "dataset.provider.kind",
                    "hint": "Use one of: wikitext2 | hf_text | synthetic | local_jsonl",
                },
            )
        ],
        True,
    )


def build_provider_schema_findings(
    provider_cfg: object,
) -> tuple[list[DoctorFinding], bool]:
    findings: list[DoctorFinding] = []
    had_error = False
    kind = str(_mapping_get(provider_cfg, "kind") or "").strip()

    if kind == "local_jsonl":
        raw_path = (
            _mapping_get(provider_cfg, "file")
            or _mapping_get(provider_cfg, "path")
            or _mapping_get(provider_cfg, "data_files")
        )
        try:
            exists = bool(raw_path) and Path(str(raw_path)).exists()
        except Exception:
            exists = False
        if not exists:
            findings.append(
                DoctorFinding(
                    code="D011",
                    severity="error",
                    message="local_jsonl: path does not exist",
                    extra={"field": "dataset.provider.file"},
                )
            )
            had_error = True

        raw_text_field = _mapping_get(provider_cfg, "text_field")
        text_field = str(raw_text_field or "").strip()
        if raw_text_field is not None and not text_field:
            findings.append(
                DoctorFinding(
                    code="D012",
                    severity="warning",
                    message=(
                        "local_jsonl: set dataset.field.text or map 'text' to your column"
                    ),
                    extra={"field": "dataset.provider.text_field"},
                )
            )

    if kind == "hf_text":
        raw_text_field = _mapping_get(provider_cfg, "text_field")
        text_field = str(raw_text_field or "").strip()
        if raw_text_field is not None and not text_field:
            findings.append(
                DoctorFinding(
                    code="D012",
                    severity="warning",
                    message="hf_text: set dataset.field.text or map 'text' to your column",
                    extra={"field": "dataset.provider.text_field"},
                )
            )

    return findings, had_error


def build_bootstrap_replicates_findings(replicates: int | None) -> list[DoctorFinding]:
    if isinstance(replicates, int) and replicates < 200:
        return [
            DoctorFinding(
                code="D004",
                severity="warning",
                message=(
                    "bootstrap replicates (<200) may produce unstable CIs; "
                    "increase reps or expect wider intervals."
                ),
                extra={"field": "eval.bootstrap.replicates"},
            )
        ]
    return []


def build_capacity_findings(
    *,
    cap: dict[str, Any],
    tier: str,
) -> tuple[list[DoctorFinding], bool, dict[str, Any] | None]:
    try:
        from invarlock.core.auto_tuning import get_tier_policies
    except Exception:
        return [], False, None

    use_tier = (tier or "balanced").lower()
    tier_policies = get_tier_policies()
    tier_defaults = tier_policies.get(use_tier, tier_policies.get("balanced", {}))
    metrics_policy = (
        tier_defaults.get("metrics", {}) if isinstance(tier_defaults, dict) else {}
    )
    pm_policy = (
        metrics_policy.get("pm_ratio", {}) if isinstance(metrics_policy, dict) else {}
    )
    acc_policy = (
        metrics_policy.get("accuracy", {}) if isinstance(metrics_policy, dict) else {}
    )

    min_tokens = int(pm_policy.get("min_tokens", 0) or 0)
    token_frac = float(pm_policy.get("min_token_fraction", 0.0) or 0.0)
    min_examples = int(acc_policy.get("min_examples", 0) or 0)
    examples_frac = float(acc_policy.get("min_examples_fraction", 0.0) or 0.0)

    tokens_avail = cap.get("tokens_available")
    examples_avail = cap.get("examples_available")
    eff_tokens = int(min_tokens)
    eff_examples = int(min_examples)
    if isinstance(tokens_avail, (int, float)) and token_frac > 0:
        eff_tokens = max(eff_tokens, int(math.ceil(float(tokens_avail) * token_frac)))
    if isinstance(examples_avail, (int, float)) and examples_frac > 0:
        eff_examples = max(
            eff_examples,
            int(math.ceil(float(examples_avail) * examples_frac)),
        )

    findings: list[DoctorFinding] = []
    if eff_tokens > 0 or eff_examples > 0:
        findings.append(
            DoctorFinding(
                code="D007",
                severity="note",
                message=(
                    f"Floors: tokens >= {eff_tokens} (effective), "
                    f"examples >= {eff_examples} (effective)"
                ),
                extra={"tokens_min": eff_tokens, "examples_min": eff_examples},
            )
        )

    insufficient = False
    if (
        isinstance(tokens_avail, (int, float))
        and eff_tokens > 0
        and tokens_avail < eff_tokens
    ):
        insufficient = True
    if (
        isinstance(examples_avail, (int, float))
        and eff_examples > 0
        and examples_avail < eff_examples
    ):
        insufficient = True

    if insufficient:
        findings.append(
            DoctorFinding(
                code="D008",
                severity="error",
                message=(
                    "Insufficient capacity: "
                    f"tokens_available={tokens_avail}, "
                    f"examples_available={examples_avail} below effective floors"
                ),
            )
        )

    policy_meta = {
        "tier": use_tier,
        "floors": {
            "pm_ratio": {
                "min_tokens": min_tokens,
                "min_token_fraction": token_frac,
            },
            "accuracy": {
                "min_examples": min_examples,
                "min_examples_fraction": examples_frac,
            },
        },
    }
    return findings, insufficient, policy_meta


def build_doctor_result(
    *,
    format_version: str,
    findings: list[dict[str, object]],
    exit_code: int,
    contracts: dict[str, Any],
    support_matrix: dict[str, Any],
    model_family_catalog: dict[str, Any],
    adapter_capabilities: dict[str, Any],
    plugin_compatibility: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    accumulator = DoctorAccumulator()
    accumulator._findings = list(findings)
    accumulator.sort()
    return {
        "format_version": format_version,
        "summary": accumulator.summary(),
        "contracts": contracts,
        "support_matrix": support_matrix,
        "model_family_catalog": model_family_catalog,
        "adapter_capabilities": adapter_capabilities,
        "plugin_compatibility": plugin_compatibility,
        "policy": policy,
        "findings": accumulator.findings,
        "resolution": {"exit_code": exit_code},
    }


def _format_report_input_error(*, label: str, exc: ReportInputError) -> str:
    if exc.reason == "not_found":
        return f"{label} report not found: {exc.path}"
    if exc.reason == "ambiguous_directory":
        return (
            f"{label} report directory is ambiguous: {exc.path}; contains both "
            "report.json and evaluation.report.json. Pass an explicit file path."
        )
    if exc.reason == "missing_canonical":
        return (
            f"{label} report directory does not contain a canonical report file: "
            f"{exc.path}. Pass an explicit file path."
        )
    if exc.reason == "non_regular":
        return (
            f"{label} report must be a regular JSON file or canonical report directory: "
            f"{exc.path}"
        )
    if exc.reason == "unreadable":
        return f"{label} report is not readable: {exc.path} ({exc.detail})"
    if exc.reason == "invalid_json":
        return f"{label} report is not valid JSON: {exc.path} ({exc.detail})"
    if exc.reason == "non_object":
        return f"{label} report must decode to a JSON object: {exc.path}"
    return f"{label} report input is invalid: {exc.path}"


def _mapping_get(value: object, key: str) -> Any:
    try:
        if isinstance(value, dict):
            return value.get(key)
        if hasattr(value, key):
            return getattr(value, key)
        getter = getattr(value, "get", None)
        if callable(getter):
            return getter(key)
    except Exception:
        return None
    return None


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_lower(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _primary_metric_kind(report_data: dict[str, Any]) -> str:
    metrics = _as_dict(report_data.get("metrics"))
    primary_metric = _as_dict(metrics.get("primary_metric"))
    return _as_lower(primary_metric.get("kind"))


def _report_tiny_relax(report_data: dict[str, Any] | None) -> bool:
    auto = report_data.get("auto", {}) if isinstance(report_data, dict) else {}
    return bool(auto.get("tiny_relax")) if isinstance(auto, dict) else False


__all__ = [
    "DATASET_SPLIT_FALLBACK_WARNING",
    "DoctorAccumulator",
    "DoctorFinding",
    "build_bootstrap_replicates_findings",
    "build_capacity_findings",
    "build_cross_check_findings",
    "build_doctor_result",
    "build_provider_kind_findings",
    "build_provider_schema_findings",
    "build_split_fallback_findings",
    "build_tiny_relax_finding",
    "load_explicit_report_input",
]
