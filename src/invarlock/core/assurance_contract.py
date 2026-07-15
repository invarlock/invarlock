"""Central strict-assurance contract for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invarlock.core.assurance_guard_validation import (
    guard_evidence_policy_errors,
    strict_guard_chain_errors,
)
from invarlock.core.assurance_plugin_validation import (
    strict_plugin_provenance_errors,
)
from invarlock.guards.authority import resolved_guard_authority

CANONICAL_GUARD_CHAIN = (
    "invariants",
    "spectral",
    "rmt",
    "variance",
    "invariants",
)
LEGACY_ASSURANCE_CLAIM_SET = "invarlock-weight-edit-regression-v1"
ASSURANCE_CLAIM_SET = "invarlock-weight-edit-regression-v2"
ASSURANCE_CLAIM_SET_V2 = ASSURANCE_CLAIM_SET
ASSURANCE_MODES = {"strict", "off"}
VERIFY_ASSURANCE_MODES = {"report", "strict", "off"}
STRICT_ASSURANCE_PROFILES = {"ci", "release"}
STRICT_ASSURANCE_TIERS = {"balanced", "conservative"}
REPORT_BUILD_EVENT_CATEGORIES = (
    "synthesized_fields",
    "repaired_fields",
    "fallback_fields",
)


def report_tiny_relax_enabled(report: dict[str, Any] | None) -> bool:
    """Return whether a report declares the development-only tiny-relax mode."""

    if not isinstance(report, dict):
        return False

    def _coerce(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value) if value in {0, 1} else None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    context = report.get("context")
    if isinstance(context, dict):
        for owner in ("run", "eval"):
            owner_context = context.get(owner)
            if isinstance(owner_context, dict):
                resolved = _coerce(owner_context.get("tiny_relax"))
                if resolved is not None:
                    return resolved

    auto = report.get("auto")
    if isinstance(auto, dict):
        resolved = _coerce(auto.get("tiny_relax"))
        if resolved is not None:
            return resolved

    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        flags = provenance.get("flags")
        if isinstance(flags, list):
            return "tiny_relax" in {str(flag).strip().lower() for flag in flags}
    return False


@dataclass(frozen=True)
class AssuranceVerdict:
    claim_set: str
    guard_authority: dict[str, str] | None
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
        section = {
            "mode": self.mode,
            "profile": self.profile,
            "tier": self.tier,
            "claim_set": self.claim_set,
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
        if self.guard_authority is not None:
            section["guard_authority"] = dict(self.guard_authority)
        return section


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
        errors.append(
            "strict assurance requires container execution mode and fail-closed "
            "runtime provenance checks."
        )
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
    guard_authority, authority_errors, has_v2_authority = resolved_guard_authority(
        report
    )
    reasons: list[str] = []
    if assurance_mode == "strict":
        reasons.extend(authority_errors)
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
            reasons.append(
                "strict assurance requires report/manifest binding plus a "
                "independently supplied runtime image digest."
            )
        reasons.extend(
            strict_guard_chain_errors(
                report,
                canonical_chain=CANONICAL_GUARD_CHAIN,
                require_assurance=False,
            )
        )
        reasons.extend(guard_evidence_policy_errors(report, require_complete=True))
        reasons.extend(
            strict_plugin_provenance_errors(
                report,
                canonical_guard_chain=CANONICAL_GUARD_CHAIN,
            )
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
        claim_set=(
            ASSURANCE_CLAIM_SET_V2 if has_v2_authority else LEGACY_ASSURANCE_CLAIM_SET
        ),
        guard_authority=(guard_authority if has_v2_authority else None),
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
    verifier_profile: str | None = None,
) -> list[str]:
    if not require_strict:
        return []
    errors: list[str] = []
    guard_authority, authority_errors, has_v2_authority = resolved_guard_authority(
        report
    )
    errors.extend(authority_errors)
    if report_tiny_relax_enabled(report):
        errors.append("strict assurance forbids development-only tiny_relax policy.")
    assurance = report.get("assurance")
    if not isinstance(assurance, dict):
        errors.append("strict assurance report missing assurance section.")
    else:
        expected_claim_set = (
            ASSURANCE_CLAIM_SET_V2 if has_v2_authority else LEGACY_ASSURANCE_CLAIM_SET
        )
        if assurance.get("claim_set") != expected_claim_set:
            errors.append("strict assurance report has unknown claim_set.")
        submitted_authority = assurance.get("guard_authority")
        if has_v2_authority:
            if submitted_authority != guard_authority:
                errors.append(
                    "strict assurance.guard_authority must exactly match "
                    "resolved_policy.guard_authority."
                )
        elif "guard_authority" in assurance:
            errors.append("legacy strict assurance cannot declare guard_authority.")
        if assurance.get("mode") != "strict":
            errors.append("strict assurance report mode must be strict.")
        if assurance.get("verdict") != "pending_verifier":
            errors.append(
                "strict assurance.verdict must be pending_verifier in submitted evidence."
            )
        if assurance.get("report_local_verdict") != "pass":
            errors.append("strict assurance.report_local_verdict must be pass.")
        if assurance.get("verified_assurance_verdict") != "pending":
            errors.append(
                "strict assurance.verified_assurance_verdict must be pending."
            )
        if assurance.get("canonical_guard_chain_enforced") is not True:
            errors.append(
                "strict assurance requires canonical_guard_chain_enforced=true."
            )
        if assurance.get("fallback_fields_used") is not False:
            errors.append("strict assurance.fallback_fields_used must be false.")
        if assurance.get("runtime_provenance_verified") is not False:
            errors.append(
                "strict assurance.runtime_provenance_verified must be false in "
                "submitted evidence."
            )
        if assurance.get("runtime_provenance_declared") != "container":
            errors.append(
                "strict assurance.runtime_provenance_declared must be container."
            )
        if assurance.get("runtime_provenance_verification_status") != "pending":
            errors.append(
                "strict assurance.runtime_provenance_verification_status must be pending."
            )
        blocking = assurance.get("blocking_reasons")
        if not isinstance(blocking, list):
            errors.append("strict assurance.blocking_reasons must be an array.")
        elif blocking:
            errors.append("strict assurance.blocking_reasons must be empty.")
            errors.extend(str(item) for item in blocking)
        assurance_profile = assurance.get("profile")
        if not isinstance(assurance_profile, str):
            errors.append("strict assurance.profile must be a string.")
        assurance_tier = assurance.get("tier")
        if not isinstance(assurance_tier, str):
            errors.append("strict assurance.tier must be a string.")
        if verifier_profile is not None:
            caller_profile = str(verifier_profile).strip().lower()
            if caller_profile not in STRICT_ASSURANCE_PROFILES:
                errors.append(
                    "strict assurance verifier caller profile must be ci or release."
                )
            elif assurance_profile != caller_profile:
                errors.append(
                    "strict assurance verifier caller profile must exactly match "
                    "assurance.profile."
                )
    profile = _report_profile(report)
    tier = _report_tier(report)
    if profile not in STRICT_ASSURANCE_PROFILES:
        errors.append("strict assurance requires report profile ci or release.")
    if tier not in STRICT_ASSURANCE_TIERS:
        errors.append("strict assurance requires report tier balanced or conservative.")
    policy_provenance = report.get("policy_provenance")
    if not isinstance(policy_provenance, dict):
        errors.append("strict assurance requires runtime-bound policy provenance.")
    else:
        if policy_provenance.get("source") != "runtime":
            errors.append("strict assurance policy provenance source must be runtime.")
    errors.extend(
        strict_guard_chain_errors(report, canonical_chain=CANONICAL_GUARD_CHAIN)
    )
    context = report.get("context")
    if isinstance(context, dict):
        context_profile = context.get("profile")
        if context_profile is not None and context_profile != profile:
            errors.append(
                "strict assurance.profile must match context.profile exactly."
            )
    meta = report.get("meta")
    if isinstance(meta, dict):
        meta_profile = meta.get("profile")
        if meta_profile is not None and meta_profile != profile:
            errors.append("strict assurance.profile must match meta.profile exactly.")
    auto = report.get("auto")
    if isinstance(auto, dict):
        auto_tier = auto.get("tier")
        if auto_tier is not None and auto_tier != tier:
            errors.append("strict assurance.tier must match auto.tier exactly.")
    if fallback_fields_used is True:
        errors.append("strict assurance forbids synthesized or repaired fields.")
    if report_build_has_blocking_evidence_events(report):
        errors.append("strict assurance forbids synthesized or repaired fields.")
    if runtime_provenance_verified is False:
        errors.append(
            "strict assurance requires report/manifest binding plus a "
            "independently supplied runtime image digest."
        )
    errors.extend(guard_evidence_policy_errors(report, require_complete=True))
    errors.extend(
        strict_plugin_provenance_errors(
            report,
            canonical_guard_chain=CANONICAL_GUARD_CHAIN,
        )
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
    "ASSURANCE_CLAIM_SET_V2",
    "LEGACY_ASSURANCE_CLAIM_SET",
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
    "report_tiny_relax_enabled",
    "resolve_report_assurance_mode",
    "resolve_report_runtime_provenance_declared",
    "strict_evaluate_policy_errors",
    "strict_report_policy_errors",
]
