"""Central strict-assurance contract for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class AssuranceVerdict:
    mode: str
    profile: str
    tier: str
    verdict: str
    blocking_reasons: tuple[str, ...]
    canonical_guard_chain_enforced: bool
    fallback_fields_used: bool
    runtime_provenance_verified: bool

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
            "verdict": self.verdict,
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
    if isinstance(assurance, dict) and isinstance(assurance.get("profile"), str):
        return assurance["profile"].strip().lower()
    context = report.get("context")
    if isinstance(context, dict) and isinstance(context.get("profile"), str):
        return context["profile"].strip().lower()
    return ""


def _report_tier(report: dict[str, Any]) -> str:
    assurance = report.get("assurance")
    if isinstance(assurance, dict) and isinstance(assurance.get("tier"), str):
        return assurance["tier"].strip().lower()
    auto = report.get("auto")
    if isinstance(auto, dict) and isinstance(auto.get("tier"), str):
        return auto["tier"].strip().lower()
    context = report.get("context")
    if isinstance(context, dict):
        auto_context = context.get("auto")
        if isinstance(auto_context, dict) and isinstance(auto_context.get("tier"), str):
            return auto_context["tier"].strip().lower()
        if isinstance(context.get("tier"), str):
            return context["tier"].strip().lower()
    return ""


def build_assurance_section(
    report: dict[str, Any],
    *,
    mode: str | None = None,
    fallback_fields_used: bool = False,
    runtime_provenance_verified: bool = True,
) -> dict[str, Any]:
    assurance_mode = normalize_assurance_mode(
        mode or resolve_report_assurance_mode(report), default="off"
    )
    observed = observed_guard_chain_from_report(report)
    profile = _report_profile(report)
    tier = _report_tier(report)
    reasons: list[str] = []
    if assurance_mode == "strict":
        reasons.extend(
            strict_evaluate_policy_errors(
                assurance_mode=assurance_mode,
                profile=profile,
                tier=tier,
                guards_order=observed,
                execution_mode="container",
                allow_unverified_provenance=not runtime_provenance_verified,
            )
        )
        if fallback_fields_used:
            reasons.append("strict assurance forbids synthesized or repaired fields.")
        for guard_name in ("spectral", "rmt", "variance", "invariants"):
            block = report.get(guard_name)
            if isinstance(block, dict) and block.get("supported") is False:
                if block.get("assurance_blocking") is True:
                    reason = block.get("reason") or "unsupported"
                    reasons.append(
                        f"{guard_name} unsupported for strict assurance: {reason}."
                    )
    return AssuranceVerdict(
        mode=assurance_mode,
        profile=profile,
        tier=tier,
        verdict="pass" if not reasons else "fail",
        blocking_reasons=tuple(reasons),
        canonical_guard_chain_enforced=is_canonical_guard_chain(observed),
        fallback_fields_used=bool(fallback_fields_used),
        runtime_provenance_verified=bool(runtime_provenance_verified),
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
        if assurance.get("verdict") != "pass":
            errors.append("strict assurance verdict must be pass.")
        if assurance.get("canonical_guard_chain_enforced") is not True:
            errors.append(
                "strict assurance requires canonical_guard_chain_enforced=true."
            )
        if assurance.get("fallback_fields_used") is True:
            errors.append("strict assurance forbids synthesized or repaired fields.")
        if assurance.get("runtime_provenance_verified") is not True:
            errors.append("strict assurance requires verified runtime provenance.")
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
    if runtime_provenance_verified is False:
        errors.append("strict assurance requires verified runtime provenance.")
    for guard_name in ("spectral", "rmt", "variance", "invariants"):
        block = report.get(guard_name)
        if isinstance(block, dict):
            if (
                block.get("supported") is False
                and block.get("assurance_blocking") is True
            ):
                reason = block.get("reason") or "unsupported"
                errors.append(
                    f"{guard_name} unsupported for strict assurance: {reason}."
                )
            if str(block.get("status", "")).lower() in {
                "degraded",
                "monitor_only",
                "monitor-only",
            }:
                errors.append(
                    f"{guard_name} is degraded/monitor-only under strict assurance."
                )
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
    "resolve_report_assurance_mode",
    "strict_evaluate_policy_errors",
    "strict_report_policy_errors",
]
