from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .report_summary import (
    build_quality_gates_summary,
    compute_console_validation_block,
)
from .utils import _fmt_by_kind, _short_digest

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


@dataclass(frozen=True)
class ReportFact:
    label: str
    value: str
    status: str = "info"
    detail: str = ""
    source: str = ""


@dataclass(frozen=True)
class ReportSection:
    key: str
    title: str
    summary: str
    priority: str
    source_blocks: tuple[str, ...]
    facts: tuple[ReportFact, ...]

    @property
    def facts_by_label(self) -> dict[str, ReportFact]:
        return {fact.label: fact for fact in self.facts}


@dataclass(frozen=True)
class EvaluationReportOutline:
    title: str
    report_kind: str
    overall_status: str
    sections: tuple[ReportSection, ...]

    @property
    def section_keys(self) -> tuple[str, ...]:
        return tuple(section.key for section in self.sections)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _status_bool(value: Any, *, default: bool | None = None) -> tuple[str, str]:
    if value is None:
        if default is None:
            return "N/A", "info"
        value = default
    if not isinstance(value, bool):
        return "N/A", "info"
    ok = value
    return ("PASS", "pass") if ok else ("FAIL", "fail")


def _plain_gate_status(value: str) -> str:
    status = value.strip()
    for prefix in ("✅", "❌", "⚠️", "⚠"):
        status = status.replace(prefix, "")
    return " ".join(status.split())


def _format_percent_range(values: list[float]) -> str:
    if not values:
        return "N/A"
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return f"{lo * 100.0:.1f}%"
    return f"{lo * 100.0:.1f}% to {hi * 100.0:.1f}%"


def _format_ci(primary_metric: dict[str, Any], kind: str) -> str:
    ci = primary_metric.get("display_ci") or primary_metric.get("ci")
    if not (isinstance(ci, list | tuple) and len(ci) == 2):
        return "N/A"
    lo, hi = ci
    if not isinstance(lo, int | float) or not isinstance(hi, int | float):
        return "N/A"
    if str(kind).lower() == "accuracy":
        return f"{float(lo):+.2f} to {float(hi):+.2f} pp"
    return f"{float(lo):.3f} to {float(hi):.3f}"


def _format_baseline_comparison(primary_metric: dict[str, Any]) -> str:
    kind = str(primary_metric.get("kind") or "").lower()
    field = "delta_vs_baseline_pp" if kind == "accuracy" else "ratio_vs_baseline"
    value = primary_metric.get(field)
    if not isinstance(value, int | float):
        return "N/A"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "N/A"
    if kind == "accuracy":
        return f"{numeric:+.2f} pp"
    if kind.startswith("ppl"):
        return f"{numeric:.3f}x"
    return f"{numeric:.3f}"


def _assurance_mode(evaluation_report: dict[str, Any]) -> str:
    """Describe only what the report declares, never independent verification.

    HTML and Markdown render one report object and do not receive a verified
    receipt or a trust anchor.  In particular, a report-authored
    ``verified_assurance_verdict`` cannot turn a renderer into a verifier.
    """

    assurance = _mapping(evaluation_report.get("assurance"))
    mode = str(assurance.get("mode") or "").strip().lower()
    if mode:
        return f"declared in report: {mode}"
    return "no declaration in report"


def _guard_warning_count(evaluation_report: dict[str, Any]) -> int:
    warnings = _mapping(evaluation_report.get("guard_warnings"))
    try:
        return int(warnings.get("warning_count") or 0)
    except _PARSE_EXCEPTIONS:
        return 0


def _baseline_summary(evaluation_report: dict[str, Any]) -> str:
    baseline_ref = _mapping(evaluation_report.get("baseline_ref"))
    provenance = _mapping(evaluation_report.get("provenance"))
    baseline_provenance = _mapping(provenance.get("baseline"))
    model_id = str(baseline_ref.get("model_id") or "").strip()
    run_id = str(
        baseline_ref.get("run_id") or baseline_provenance.get("run_id") or ""
    ).strip()
    if model_id and run_id:
        return f"{model_id} · run {_short_digest(run_id)}"
    if model_id:
        return model_id
    if run_id:
        return f"run {_short_digest(run_id)}"
    return "unknown"


def _build_decision_section(evaluation_report: dict[str, Any]) -> ReportSection:
    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    status_value, _status = _status_bool(overall_pass)
    meta = _mapping(evaluation_report.get("meta"))
    primary_metric = _mapping(evaluation_report.get("primary_metric"))
    warning_count = _guard_warning_count(evaluation_report)
    warning_evidence_present = isinstance(evaluation_report.get("guard_warnings"), dict)
    facts = (
        ReportFact(
            "Report-local Gates",
            status_value,
            "info",
            source="validation",
        ),
        ReportFact(
            "Independent Verification",
            "NOT EMBEDDED",
            "warn",
            source="renderer",
        ),
        ReportFact(
            "Declared Assurance Mode",
            _assurance_mode(evaluation_report),
            source="assurance",
        ),
        ReportFact(
            "Model", str(meta.get("model_id") or "unknown"), source="meta.model_id"
        ),
        ReportFact(
            "Baseline",
            _baseline_summary(evaluation_report),
            source="baseline_ref",
        ),
        ReportFact(
            "Adapter", str(meta.get("adapter") or "unknown"), source="meta.adapter"
        ),
        ReportFact(
            "Edit",
            str(evaluation_report.get("edit_name") or "unknown"),
            source="edit_name",
        ),
        ReportFact(
            "Primary Metric",
            str(primary_metric.get("kind") or "unknown"),
            source="primary_metric.kind",
        ),
        ReportFact(
            "Guard Warnings",
            str(warning_count) if warning_evidence_present else "N/A",
            "warn" if warning_count else "pass" if warning_evidence_present else "info",
            source="guard_warnings.warning_count",
        ),
    )
    return ReportSection(
        key="decision",
        title="Decision",
        summary="Policy verdict, evidence mode, subject/baseline identity, edit, and warning count.",
        priority="summary",
        source_blocks=(
            "validation",
            "assurance",
            "meta",
            "baseline_ref",
            "primary_metric",
            "guard_warnings",
        ),
        facts=facts,
    )


def _build_primary_metric_section(evaluation_report: dict[str, Any]) -> ReportSection:
    primary_metric = _mapping(evaluation_report.get("primary_metric"))
    validation = _mapping(evaluation_report.get("validation"))
    kind = str(primary_metric.get("kind") or "unknown")
    pm_status, pm_tone = _status_bool(validation.get("primary_metric_acceptable"))
    tail = _mapping(evaluation_report.get("primary_metric_tail"))
    tail_fact = ReportFact("Tail Gate", "N/A", source="primary_metric_tail")
    if tail:
        if bool(tail.get("evaluated", False)):
            tail_status, tail_tone = _status_bool(tail.get("passed"))
            tail_fact = ReportFact(
                "Tail Gate",
                tail_status,
                tail_tone,
                source="primary_metric_tail.passed",
            )
        else:
            tail_fact = ReportFact(
                "Tail Gate", "not evaluated", source="primary_metric_tail"
            )

    summary = "Task metric and baseline-relative movement."
    if kind.lower() == "accuracy":
        summary = "Task accuracy and baseline-relative delta."
    elif kind.lower().startswith("ppl"):
        summary = "Perplexity and baseline-relative ratio."

    facts = (
        ReportFact("Metric", kind, source="primary_metric.kind"),
        ReportFact(
            "Preview",
            _fmt_by_kind(primary_metric.get("preview"), kind),
            source="primary_metric.preview",
        ),
        ReportFact(
            "Final",
            _fmt_by_kind(primary_metric.get("final"), kind),
            source="primary_metric.final",
        ),
        ReportFact(
            "Baseline Comparison",
            _format_baseline_comparison(primary_metric),
            source=(
                "primary_metric.delta_vs_baseline_pp"
                if kind.lower() == "accuracy"
                else "primary_metric.ratio_vs_baseline"
            ),
        ),
        ReportFact(
            "CI", _format_ci(primary_metric, kind), source="primary_metric.display_ci"
        ),
        ReportFact(
            "Policy Gate",
            pm_status,
            pm_tone,
            source="validation.primary_metric_acceptable",
        ),
        tail_fact,
    )
    return ReportSection(
        key="primary_metric",
        title="Primary Metric",
        summary=summary,
        priority="summary",
        source_blocks=("primary_metric", "primary_metric_tail", "validation"),
        facts=facts,
    )


def _build_policy_gates_section(evaluation_report: dict[str, Any]) -> ReportSection:
    summary = build_quality_gates_summary(evaluation_report)
    facts = tuple(
        ReportFact(
            row.label,
            f"{_plain_gate_status(row.status)} | {row.measured} vs {row.threshold}",
            "pass"
            if "PASS" in row.status
            else "fail"
            if "FAIL" in row.status
            else "info",
            detail=f"{row.description}; basis={row.basis}",
            source="validation",
        )
        for row in summary.rows
    )
    return ReportSection(
        key="policy_gates",
        title="Policy Gates",
        summary="Hard policy gates and thresholds used by verify.",
        priority="review",
        source_blocks=("validation", "policy_digest", "resolved_policy"),
        facts=facts,
    )


def _build_guard_signals_section(evaluation_report: dict[str, Any]) -> ReportSection:
    validation = _mapping(evaluation_report.get("validation"))
    spectral = _mapping(evaluation_report.get("spectral"))
    rmt = _mapping(evaluation_report.get("rmt"))
    variance = _mapping(evaluation_report.get("variance"))
    moe = _mapping(evaluation_report.get("moe"))
    warning_count = _guard_warning_count(evaluation_report)
    warning_evidence_present = isinstance(evaluation_report.get("guard_warnings"), dict)
    spectral_value = "N/A"
    if spectral:
        caps = spectral.get("caps_applied")
        max_caps = spectral.get("max_caps")
        spectral_value = (
            f"{caps}/{max_caps} caps"
            if caps is not None and max_caps is not None
            else str(spectral.get("status") or "recorded")
        )
    rmt_value = str(rmt.get("status") or "recorded") if rmt else "N/A"
    variance_value = "recorded" if variance else "N/A"
    moe_value = "observed" if moe else "N/A"

    facts = (
        ReportFact(
            "Guard Warnings",
            str(warning_count) if warning_evidence_present else "N/A",
            "warn" if warning_count else "pass" if warning_evidence_present else "info",
            source="guard_warnings",
        ),
        ReportFact(
            "Invariants",
            _status_bool(validation.get("invariants_pass"))[0],
            _status_bool(validation.get("invariants_pass"))[1],
            source="validation.invariants_pass",
        ),
        ReportFact(
            "Spectral",
            spectral_value,
            _status_bool(validation.get("spectral_stable"))[1],
            source="spectral",
        ),
        ReportFact(
            "RMT",
            rmt_value,
            _status_bool(validation.get("rmt_stable"))[1],
            source="rmt",
        ),
        ReportFact("Variance", variance_value, source="variance"),
        ReportFact("MoE", moe_value, source="moe"),
    )
    return ReportSection(
        key="guard_signals",
        title="Guard Signals",
        summary="Guard observations separated from hard policy failure semantics.",
        priority="review",
        source_blocks=(
            "guard_warnings",
            "invariants",
            "spectral",
            "rmt",
            "variance",
            "moe",
        ),
        facts=facts,
    )


def _benchmark_block(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    for key in ("benchmark_comparison", "benchmark", "guard_effect_benchmark"):
        block = evaluation_report.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _build_benchmark_section(evaluation_report: dict[str, Any]) -> ReportSection | None:
    block = _benchmark_block(evaluation_report)
    scenarios = block.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return None

    total = len(scenarios)
    skipped = 0
    passed = 0
    pm_impacts: list[float] = []
    time_overheads: list[float] = []
    mem_overheads: list[float] = []
    rmt_pairs: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if bool(scenario.get("skip") or scenario.get("skipped")):
            skipped += 1
            continue
        pass_block = scenario.get("pass") or scenario.get("gates")
        if isinstance(pass_block, dict) and pass_block and all(pass_block.values()):
            passed += 1
        for key, target in (
            ("guard_primary_metric_impact", pm_impacts),
            ("guard_runtime_overhead", time_overheads),
            ("guard_memory_overhead", mem_overheads),
        ):
            value = scenario.get(key)
            if isinstance(value, int | float) and math.isfinite(float(value)):
                target.append(float(value))
        bare = scenario.get("rmt_outliers_bare")
        guarded = scenario.get("rmt_outliers_guarded")
        if isinstance(bare, int | float) and isinstance(guarded, int | float):
            rmt_pairs.append(f"{int(bare)}->{int(guarded)}")

    unique_rmt = sorted(set(rmt_pairs))
    facts = (
        ReportFact(
            "Profile",
            str(block.get("profile") or "unknown"),
            source="benchmark.profile",
        ),
        ReportFact(
            "Scenarios",
            f"{total} total, {passed} passed, {skipped} skipped",
            "pass" if passed + skipped == total else "fail",
            source="benchmark.scenarios",
        ),
        ReportFact(
            "Primary Metric Impact",
            _format_percent_range(pm_impacts),
            source="benchmark.scenarios.guard_primary_metric_impact",
        ),
        ReportFact(
            "Time Overhead",
            _format_percent_range(time_overheads),
            source="benchmark.scenarios.guard_runtime_overhead",
        ),
        ReportFact(
            "Memory Overhead",
            _format_percent_range(mem_overheads),
            source="benchmark.scenarios.guard_memory_overhead",
        ),
        ReportFact(
            "RMT Outliers",
            ", ".join(unique_rmt) if unique_rmt else "N/A",
            source="benchmark.scenarios.rmt_outliers",
        ),
    )
    return ReportSection(
        key="benchmark_comparison",
        title="Benchmark Comparison",
        summary="Bare-vs-guarded benchmark deltas and scenario gates.",
        priority="review",
        source_blocks=("benchmark_comparison", "benchmark", "guard_effect_benchmark"),
        facts=facts,
    )


def _build_evidence_provenance_section(
    evaluation_report: dict[str, Any],
) -> ReportSection:
    dataset = _mapping(evaluation_report.get("dataset"))
    windows = _mapping(dataset.get("windows"))
    hashes = _mapping(dataset.get("hash"))
    provenance = _mapping(evaluation_report.get("provenance"))
    provider_digest = _mapping(provenance.get("provider_digest"))
    policy_digest = _mapping(evaluation_report.get("policy_digest"))
    meta = _mapping(evaluation_report.get("meta"))
    window_count = "N/A"
    if windows:
        window_count = f"{windows.get('preview', 'N/A')} preview, {windows.get('final', 'N/A')} final"
    facts = (
        ReportFact(
            "Dataset",
            str(dataset.get("provider") or "unknown"),
            source="dataset.provider",
        ),
        ReportFact("Windows", window_count, source="dataset.windows"),
        ReportFact(
            "Hash Source",
            str(hashes.get("source") or "unknown"),
            source="dataset.hash.source",
        ),
        ReportFact(
            "Provider Digest",
            "present" if provider_digest else "missing",
            "pass" if provider_digest else "warn",
            source="provenance.provider_digest",
        ),
        ReportFact(
            "Policy Digest",
            _short_digest(str(policy_digest.get("thresholds_hash") or "missing")),
            source="policy_digest.thresholds_hash",
        ),
        ReportFact(
            "Device", str(meta.get("device") or "unknown"), source="meta.device"
        ),
        ReportFact("Seed", str(meta.get("seed") or "unknown"), source="meta.seed"),
    )
    return ReportSection(
        key="evidence_provenance",
        title="Evidence And Provenance",
        summary="Dataset, pairing, runtime, policy, and digest evidence needed for audit.",
        priority="audit",
        source_blocks=("dataset", "provenance", "policy_digest", "meta", "artifacts"),
        facts=facts,
    )


def _build_technical_appendix_section(
    evaluation_report: dict[str, Any],
) -> ReportSection:
    appendix_blocks = [
        key
        for key in (
            "plugins",
            "resolved_policy",
            "policy_provenance",
            "system_overhead",
            "classification",
            "structure",
            "compression_diagnostics",
            "artifacts",
        )
        if evaluation_report.get(key) not in ({}, [], None)
    ]
    facts = (
        ReportFact(
            "Raw Blocks",
            ", ".join(appendix_blocks) if appendix_blocks else "none",
            source="top-level",
        ),
    )
    return ReportSection(
        key="technical_appendix",
        title="Technical Appendix",
        summary="Verbose raw measurements, policies, plugin provenance, and artifacts.",
        priority="appendix",
        source_blocks=tuple(appendix_blocks),
        facts=facts,
    )


def _build_edit_provenance_section(
    evaluation_report: dict[str, Any],
) -> ReportSection | None:
    edit = _mapping(evaluation_report.get("edit"))
    provenance = _mapping(edit.get("edit_provenance"))
    impact = _mapping(edit.get("edit_impact"))
    topology = _mapping(edit.get("edit_topology"))
    delta_privacy = _mapping(edit.get("delta_privacy"))
    if not provenance and not impact and not topology and not delta_privacy:
        return None

    scenario_types = impact.get("scenario_types")
    scenario_display = (
        ", ".join(str(item) for item in scenario_types)
        if isinstance(scenario_types, list)
        else "N/A"
    )
    facts = (
        ReportFact(
            "Edit Family",
            str(provenance.get("edit_family") or "N/A"),
            source="edit.edit_provenance.edit_family",
        ),
        ReportFact(
            "Edit Method",
            str(provenance.get("edit_method") or "N/A"),
            source="edit.edit_provenance.edit_method",
        ),
        ReportFact(
            "Edit Count",
            str(provenance.get("edit_count") or "N/A"),
            source="edit.edit_provenance.edit_count",
        ),
        ReportFact(
            "Dynamic Runtime",
            str(provenance.get("dynamic_runtime_required", "N/A")),
            source="edit.edit_provenance.dynamic_runtime_required",
        ),
        ReportFact(
            "Scenario Types",
            scenario_display,
            source="edit.edit_impact.scenario_types",
        ),
        ReportFact(
            "Artifact Kind",
            str(topology.get("artifact_kind") or "N/A"),
            source="edit.edit_topology.artifact_kind",
        ),
        ReportFact(
            "Runtime Activation",
            str(topology.get("runtime_activation_policy") or "N/A"),
            source="edit.edit_topology.runtime_activation_policy",
        ),
        ReportFact(
            "Delta Availability",
            str(delta_privacy.get("delta_available") or "N/A"),
            source="edit.delta_privacy.delta_available",
        ),
        ReportFact(
            "Privacy Sensitivity",
            str(delta_privacy.get("privacy_sensitivity") or "N/A"),
            source="edit.delta_privacy.privacy_sensitivity",
        ),
    )
    return ReportSection(
        key="edit_provenance",
        title="Edit Provenance",
        summary=(
            "Optional descriptive metadata about the upstream subject-generation "
            "workflow."
        ),
        priority="review",
        source_blocks=("edit",),
        facts=facts,
    )


def _build_evaluation_realism_section(
    evaluation_report: dict[str, Any],
) -> ReportSection | None:
    realism = _mapping(evaluation_report.get("evaluation_realism"))
    if not realism:
        return None

    warning = realism.get("proxy_metric_warning")
    facts = (
        ReportFact(
            "Mode",
            str(realism.get("mode") or "N/A"),
            source="evaluation_realism.mode",
        ),
        ReportFact(
            "Generation Realistic",
            str(realism.get("metric_is_generation_realistic", "N/A")),
            source="evaluation_realism.metric_is_generation_realistic",
        ),
        ReportFact(
            "Task",
            str(realism.get("dataset_or_task_id") or "N/A"),
            source="evaluation_realism.dataset_or_task_id",
        ),
        ReportFact(
            "Max Tokens",
            str(realism.get("max_tokens", "N/A")),
            source="evaluation_realism.max_tokens",
        ),
        ReportFact(
            "Truncation",
            str(realism.get("truncation_policy") or "N/A"),
            source="evaluation_realism.truncation_policy",
        ),
        ReportFact(
            "Proxy Warning",
            str(warning or "none"),
            source="evaluation_realism.proxy_metric_warning",
        ),
    )
    return ReportSection(
        key="evaluation_realism",
        title="Evaluation Realism",
        summary=(
            "Optional context describing whether the metric reflects live generation "
            "behavior or a proxy evaluation setup."
        ),
        priority="review",
        source_blocks=("evaluation_realism",),
        facts=facts,
    )


def build_evaluation_report_outline(
    evaluation_report: dict[str, Any],
) -> EvaluationReportOutline:
    """Build a renderer-neutral report outline for modern evaluation reports."""

    block = compute_console_validation_block(evaluation_report)
    overall_pass = bool(block.get("overall_pass"))
    sections: list[ReportSection] = [
        _build_decision_section(evaluation_report),
        _build_primary_metric_section(evaluation_report),
        _build_policy_gates_section(evaluation_report),
        _build_guard_signals_section(evaluation_report),
    ]
    benchmark = _build_benchmark_section(evaluation_report)
    if benchmark is not None:
        sections.append(benchmark)
    evaluation_realism = _build_evaluation_realism_section(evaluation_report)
    if evaluation_realism is not None:
        sections.append(evaluation_realism)
    edit_provenance = _build_edit_provenance_section(evaluation_report)
    if edit_provenance is not None:
        sections.append(edit_provenance)
    sections.extend(
        [
            _build_evidence_provenance_section(evaluation_report),
            _build_technical_appendix_section(evaluation_report),
        ]
    )
    return EvaluationReportOutline(
        title="InvarLock Evaluation Report",
        report_kind="evaluation",
        overall_status="PASS" if overall_pass else "FAIL",
        sections=tuple(sections),
    )


__all__ = [
    "EvaluationReportOutline",
    "ReportFact",
    "ReportSection",
    "build_evaluation_report_outline",
]
