"""Central strict-assurance contract for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.core.guard_evidence import GuardEvidence

CANONICAL_GUARD_CHAIN = (
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "invariants",
)
ASSURANCE_CLAIM_SET = "invarlock-weight-edit-regression-v1"
ASSURANCE_MODES = {"strict", "off"}
VERIFY_ASSURANCE_MODES = {"report", "strict", "off"}
STRICT_ASSURANCE_PROFILES = {"ci", "release"}
STRICT_ASSURANCE_TIERS = {"balanced", "conservative"}
REPORT_BUILD_EVENT_CATEGORIES = (
    "synthesized_fields",
    "repaired_fields",
    "fallback_fields",
)


@dataclass(frozen=True)
class AssuranceVerdict:
    mode: str
    profile: str
    tier: str
    verdict: str
    report_local_verdict: str
    verified_assurance_verdict: str
    blocking_reasons: tuple[str, ...]
    canonical_guard_chain_enforced: bool
    fallback_fields_used: bool
    runtime_provenance_verified: bool
    runtime_provenance_declared: str
    runtime_provenance_verification_status: str

    def as_report_section(self, *, observed_guard_chain: list[str]) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "tier": self.tier,
            "claim_set": ASSURANCE_CLAIM_SET,
            "canonical_guard_chain": list(CANONICAL_GUARD_CHAIN),
            "guard_chain_observed": list(observed_guard_chain),
            "canonical_guard_chain_enforced": self.canonical_guard_chain_enforced,
            "fallback_fields_used": self.fallback_fields_used,
            "runtime_provenance_verified": self.runtime_provenance_verified,
            "runtime_provenance_declared": self.runtime_provenance_declared,
            "runtime_provenance_verification_status": (
                self.runtime_provenance_verification_status
            ),
            "verdict": self.verdict,
            "report_local_verdict": self.report_local_verdict,
            "verified_assurance_verdict": self.verified_assurance_verdict,
            "blocking_reasons": list(self.blocking_reasons),
        }


def normalize_assurance_mode(mode: object, *, default: str = "strict") -> str:
    text = str(mode if mode is not None else default).strip().lower()
    if text not in ASSURANCE_MODES:
        raise ValueError("Assurance mode must be one of: strict, off.")
    return text


def normalize_verify_assurance_mode(mode: object, *, default: str = "report") -> str:
    text = str(mode if mode is not None else default).strip().lower()
    if text not in VERIFY_ASSURANCE_MODES:
        raise ValueError("Verify assurance mode must be one of: report, strict, off.")
    return text


def is_canonical_guard_chain(guard_chain: Any) -> bool:
    return (
        isinstance(guard_chain, list | tuple)
        and tuple(guard_chain) == CANONICAL_GUARD_CHAIN
    )


def strict_evaluate_policy_errors(
    *,
    assurance_mode: str,
    profile: str,
    tier: str,
    guards_order: list[str],
    execution_mode: str = "container",
    allow_unverified_provenance: bool = False,
) -> list[str]:
    if assurance_mode == "off":
        return []
    errors: list[str] = []
    profile_name = str(profile or "").strip().lower()
    tier_name = str(tier or "").strip().lower()
    execution_name = str(execution_mode or "").strip().lower()
    if profile_name not in STRICT_ASSURANCE_PROFILES:
        errors.append("strict assurance requires profile ci or release.")
    if tier_name not in STRICT_ASSURANCE_TIERS:
        errors.append("strict assurance requires tier balanced or conservative.")
    if not is_canonical_guard_chain(guards_order):
        errors.append("strict assurance requires the canonical guard chain.")
    if execution_name != "container" or allow_unverified_provenance:
        errors.append("strict assurance requires verified container provenance.")
    return errors


def resolve_report_assurance_mode(report: dict[str, Any]) -> str:
    section = report.get("assurance")
    if isinstance(section, dict):
        mode = section.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip().lower()
    context = report.get("context")
    if isinstance(context, dict):
        assurance = context.get("assurance")
        if isinstance(assurance, dict):
            mode = assurance.get("mode")
            if isinstance(mode, str) and mode.strip():
                return mode.strip().lower()
    return "off"


def observed_guard_chain_from_report(report: dict[str, Any]) -> list[str]:
    assurance = report.get("assurance")
    if isinstance(assurance, dict):
        observed = assurance.get("guard_chain_observed")
        if isinstance(observed, list) and all(
            isinstance(item, str) for item in observed
        ):
            return list(observed)
    context = report.get("context")
    if isinstance(context, dict):
        observed = context.get("guard_chain_observed")
        if isinstance(observed, list) and all(
            isinstance(item, str) for item in observed
        ):
            return list(observed)
    guards = report.get("guards")
    if isinstance(guards, list):
        names: list[str] = []
        for guard in guards:
            if isinstance(guard, dict) and isinstance(guard.get("name"), str):
                names.append(str(guard["name"]))
        if names:
            return names
    plugins = report.get("plugins")
    plugin_guards = plugins.get("guards") if isinstance(plugins, dict) else None
    if isinstance(plugin_guards, list) and all(
        isinstance(item, str) for item in plugin_guards
    ):
        return list(plugin_guards)
    return []


def _report_profile(report: dict[str, Any]) -> str:
    assurance = report.get("assurance")
    assurance_profile = (
        assurance.get("profile") if isinstance(assurance, dict) else None
    )
    if isinstance(assurance_profile, str):
        return assurance_profile.strip().lower()
    context = report.get("context")
    context_profile = context.get("profile") if isinstance(context, dict) else None
    if isinstance(context_profile, str):
        return context_profile.strip().lower()
    return ""


def _report_tier(report: dict[str, Any]) -> str:
    assurance = report.get("assurance")
    assurance_tier = assurance.get("tier") if isinstance(assurance, dict) else None
    if isinstance(assurance_tier, str):
        return assurance_tier.strip().lower()
    auto = report.get("auto")
    auto_tier = auto.get("tier") if isinstance(auto, dict) else None
    if isinstance(auto_tier, str):
        return auto_tier.strip().lower()
    context = report.get("context")
    if isinstance(context, dict):
        auto_context = context.get("auto")
        auto_context_tier = (
            auto_context.get("tier") if isinstance(auto_context, dict) else None
        )
        if isinstance(auto_context_tier, str):
            return auto_context_tier.strip().lower()
        context_tier = context.get("tier")
        if isinstance(context_tier, str):
            return context_tier.strip().lower()
    return ""


def _nonblocking_report_build_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    return (
        event.get("field") == "primary_metric.display_ci"
        and event.get("reason") == "computed_from_primary_metric_ci"
    )


def report_build_has_blocking_evidence_events(report: dict[str, Any]) -> bool:
    section = report.get("report_build")
    if not isinstance(section, dict):
        return False
    for category in REPORT_BUILD_EVENT_CATEGORIES:
        events = section.get(category)
        if not isinstance(events, list):
            continue
        for event in events:
            if not _nonblocking_report_build_event(event):
                return True
    return False


def _strict_guard_blocking_reasons(guard_name: str, block: Any) -> tuple[str, ...]:
    evidence = GuardEvidence.from_report_block(guard_name, block)
    if evidence is None:
        return ()
    return evidence.strict_blocking_reasons()


def resolve_report_runtime_provenance_declared(
    report: dict[str, Any], *, default: str = "unknown"
) -> str:
    context = report.get("context")
    candidates: list[Any] = []
    if isinstance(context, dict):
        runtime = context.get("runtime")
        if isinstance(runtime, dict):
            candidates.append(runtime.get("execution_mode"))
            candidates.append(runtime.get("runtime_provenance_declared"))
        candidates.append(context.get("execution_mode"))
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        runtime = provenance.get("runtime")
        if isinstance(runtime, dict):
            candidates.append(runtime.get("execution_mode"))
        candidates.append(provenance.get("execution_mode"))
    meta = report.get("meta")
    if isinstance(meta, dict):
        candidates.append(meta.get("execution_mode"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return default


def build_assurance_section(
    report: dict[str, Any],
    *,
    mode: str | None = None,
    fallback_fields_used: bool = False,
    runtime_provenance_verified: bool | None = None,
    runtime_provenance_declared: str | None = None,
    runtime_provenance_verification_status: str | None = None,
) -> dict[str, Any]:
    assurance_mode = normalize_assurance_mode(
        mode or resolve_report_assurance_mode(report), default="off"
    )
    observed = observed_guard_chain_from_report(report)
    fallback_fields_used = bool(
        fallback_fields_used or report_build_has_blocking_evidence_events(report)
    )
    profile = _report_profile(report)
    tier = _report_tier(report)
    declared_provenance = (
        str(runtime_provenance_declared).strip().lower()
        if runtime_provenance_declared is not None
        else resolve_report_runtime_provenance_declared(report)
    )
    provenance_status = runtime_provenance_verification_status or (
        "verified" if runtime_provenance_verified is True else "pending"
    )
    provenance_verified = runtime_provenance_verified is True
    reasons: list[str] = []
    if assurance_mode == "strict":
        reasons.extend(
            strict_evaluate_policy_errors(
                assurance_mode=assurance_mode,
                profile=profile,
                tier=tier,
                guards_order=observed,
                execution_mode=declared_provenance,
                allow_unverified_provenance=False,
            )
        )
        if fallback_fields_used:
            reasons.append("strict assurance forbids synthesized or repaired fields.")
        if provenance_status not in {"pending", "verified"}:
            reasons.append("strict assurance requires verified runtime provenance.")
        for guard_name in ("spectral", "rmt", "variance", "invariants"):
            reasons.extend(
                _strict_guard_blocking_reasons(guard_name, report.get(guard_name))
            )
    report_local_verdict = "pass" if not reasons else "fail"
    if reasons:
        verdict = "fail"
        verified_assurance_verdict = "fail"
    elif assurance_mode == "strict" and provenance_status == "pending":
        verdict = "pending_verifier"
        verified_assurance_verdict = "pending"
    else:
        verdict = "pass"
        verified_assurance_verdict = "pass"
    return AssuranceVerdict(
        mode=assurance_mode,
        profile=profile,
        tier=tier,
        verdict=verdict,
        report_local_verdict=report_local_verdict,
        verified_assurance_verdict=verified_assurance_verdict,
        blocking_reasons=tuple(reasons),
        canonical_guard_chain_enforced=is_canonical_guard_chain(observed),
        fallback_fields_used=bool(fallback_fields_used),
        runtime_provenance_verified=provenance_verified,
        runtime_provenance_declared=declared_provenance,
        runtime_provenance_verification_status=str(provenance_status or ""),
    ).as_report_section(observed_guard_chain=observed)


def strict_report_policy_errors(
    report: dict[str, Any],
    *,
    require_strict: bool,
    fallback_fields_used: bool | None = None,
    runtime_provenance_verified: bool | None = None,
) -> list[str]:
    if not require_strict:
        return []
    errors: list[str] = []
    assurance = report.get("assurance")
    if not isinstance(assurance, dict):
        errors.append("strict assurance report missing assurance section.")
    else:
        if assurance.get("claim_set") != ASSURANCE_CLAIM_SET:
            errors.append("strict assurance report has unknown claim_set.")
        if assurance.get("mode") != "strict":
            errors.append("strict assurance report mode must be strict.")
        verdict = assurance.get("verdict")
        if verdict not in {"pass", "pending_verifier"}:
            errors.append("strict assurance verdict must be pass or pending_verifier.")
        if (
            verdict == "pending_verifier"
            and runtime_provenance_verified is True
            and assurance.get("report_local_verdict") not in {None, "pass"}
        ):
            errors.append("strict assurance report-local verdict must be pass.")
        if assurance.get("canonical_guard_chain_enforced") is not True:
            errors.append(
                "strict assurance requires canonical_guard_chain_enforced=true."
            )
        if assurance.get("fallback_fields_used") is True:
            errors.append("strict assurance forbids synthesized or repaired fields.")
        blocking = assurance.get("blocking_reasons")
        if isinstance(blocking, list) and blocking:
            errors.extend(str(item) for item in blocking)
    profile = _report_profile(report)
    tier = _report_tier(report)
    if profile not in STRICT_ASSURANCE_PROFILES:
        errors.append("strict assurance requires report profile ci or release.")
    if tier not in STRICT_ASSURANCE_TIERS:
        errors.append("strict assurance requires report tier balanced or conservative.")
    if not is_canonical_guard_chain(observed_guard_chain_from_report(report)):
        errors.append("strict assurance requires canonical guard chain evidence.")
    if fallback_fields_used is True:
        errors.append("strict assurance forbids synthesized or repaired fields.")
    if report_build_has_blocking_evidence_events(report):
        errors.append("strict assurance forbids synthesized or repaired fields.")
    if runtime_provenance_verified is False:
        errors.append("strict assurance requires verified runtime provenance.")
    for guard_name in ("spectral", "rmt", "variance", "invariants"):
        block = report.get(guard_name)
        if not isinstance(block, dict) or not block:
            errors.append(f"strict assurance missing {guard_name} guard evidence.")
            continue
        errors.extend(_strict_guard_blocking_reasons(guard_name, block))
    return _dedupe(errors)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "ASSURANCE_CLAIM_SET",
    "ASSURANCE_MODES",
    "CANONICAL_GUARD_CHAIN",
    "STRICT_ASSURANCE_PROFILES",
    "STRICT_ASSURANCE_TIERS",
    "VERIFY_ASSURANCE_MODES",
    "AssuranceVerdict",
    "build_assurance_section",
    "is_canonical_guard_chain",
    "normalize_assurance_mode",
    "normalize_verify_assurance_mode",
    "observed_guard_chain_from_report",
    "report_build_has_blocking_evidence_events",
    "resolve_report_assurance_mode",
    "resolve_report_runtime_provenance_declared",
    "strict_evaluate_policy_errors",
    "strict_report_policy_errors",
]
