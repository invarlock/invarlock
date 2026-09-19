"""Bounded presentation of retained judge evidence without recipient acceptance."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from invarlock.engine import run_digest
from invarlock.evaluation_record_contracts.contracts import digest as record_digest
from invarlock.evidence_reporting import EvidenceReportError, EvidenceReportV2
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.record_reporting import (
    _captured_context,
    _captured_identities,
    _configuration_preview,
    _short_context,
)
from invarlock.report_presentation import (
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    number,
    render_html,
    render_markdown,
    xml_text,
)
from invarlock.report_publication import (
    ReportPublicationError,
    publish_report_outputs,
    validate_report_destinations,
)

CASE_DETAIL_LIMIT = 50
TEXT_DETAIL_LIMIT = 2000


def _count_label(value: int, singular: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {singular}{suffix}"


@dataclass(frozen=True)
class JudgeEvidenceReport(EvidenceReportV2):
    facts: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return canonical_payload(
            {
                "format_version": "invarlock/judge-evidence-report-v1",
                "kind": "judge",
                "ok": not self.errors,
                "evidence_digest": self.pack_manifest_digest,
                "requested_outputs": self.requested_outputs,
                "written_outputs": self.written_outputs,
                "errors": list(self.errors),
                **self.facts,
            }
        ).decode()


def is_judge_evidence(path: Path) -> bool:
    """Recognize the additive envelope filename without falling back on errors."""
    candidate = Path(path) / "envelope.json"
    return candidate.exists() or candidate.is_symlink()


def _snapshot(path: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    from invarlock.judge_measurements.evidence import (
        object_sha256,
        read_object,
        replay_judge_evidence,
    )

    publication = replay_judge_evidence(path)
    artifacts: dict[str, dict[str, Any]] = {}
    retained = [
        ("plan", "plan_sha256"),
        ("measurements", "measurements_sha256"),
        ("analysis_policy", "analysis_policy_sha256"),
        ("baseline_run", "baseline_run_sha256"),
        ("subject_run", "subject_run_sha256"),
    ]
    if "native_capture_sha256" in publication.envelope["bindings"]:
        retained.append(("native_capture", "native_capture_sha256"))
    for name, digest_field in retained:
        value = read_object(Path(path) / f"{name}.json")
        digest = run_digest(value) if name.endswith("_run") else object_sha256(value)
        if (
            digest
            != cast(dict[str, str], publication.envelope["bindings"])[digest_field]
        ):
            raise EvidenceReportError(
                "judge evidence changed between replay and rendering"
            )
        artifacts[name] = value
    return publication, artifacts


def _baseline_mean(plan: dict[str, Any], measurements: dict[str, Any]) -> str:
    """Describe a complete replay with the same equal-unit weighting as analysis."""
    case_units = {
        case["case_id"]: case["unit_id"] for case in plan["sampling"]["case_units"]
    }
    case_scores: dict[str, list[Fraction]] = {case_id: [] for case_id in case_units}
    for trial in measurements["trials"]:
        if trial["side"] == "baseline":
            case_scores[trial["case_id"]].append(Fraction(trial["parse"]["value"]))
    unit_scores: dict[str, list[Fraction]] = {}
    for case_id, scores in case_scores.items():
        unit_scores.setdefault(case_units[case_id], []).append(
            sum(scores, Fraction()) / len(scores)
        )
    mean = sum(
        (sum(scores, Fraction()) / len(scores) for scores in unit_scores.values()),
        Fraction(),
    ) / len(unit_scores)
    # Display-only rounding is independent of the caller's Decimal context.
    with localcontext(Context(prec=100, rounding=ROUND_HALF_EVEN)):
        return str(
            (Decimal(mean.numerator) / Decimal(mean.denominator)).quantize(
                Decimal("0.000000000000001")
            )
        )


def _selected_case_ids(
    requested: tuple[str, ...], available: tuple[str, ...]
) -> tuple[str, ...]:
    if len(requested) > CASE_DETAIL_LIMIT:
        raise EvidenceReportError(
            f"at most {CASE_DETAIL_LIMIT} --case-id values may be requested"
        )
    if len(set(requested)) != len(requested):
        raise EvidenceReportError("--case-id values must be unique")
    missing = sorted(set(requested) - set(available))
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise EvidenceReportError(
            f"requested case IDs are not in the judge plan: {preview}{suffix}"
        )
    return requested or available[:CASE_DETAIL_LIMIT]


def _captured_record_context(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source settings and projection identities without source prompts."""
    settings: dict[str, Any] = {}
    settings_count = 0
    projections: dict[str, dict[str, str]] = {}
    projection_count = 0
    for record in records:
        context = record.get("context")
        if not isinstance(context, dict):
            continue
        if "settings" in context:
            settings_count += 1
            identity = record_digest(context["settings"])
            if identity not in settings:
                settings[identity] = context["settings"]
        projection = context.get("input_projection")
        if isinstance(projection, dict):
            projection_count += 1
            identity = projection["configuration_digest"]
            projections[identity] = {
                "configuration_digest": identity,
                "kind": projection["configuration"]["kind"],
                "pointer": _short_context(projection["configuration"]["pointer"]),
            }
    result: dict[str, Any] = {}
    if settings_count:
        result["settings"] = {
            "status": "common"
            if len(settings) == 1 and settings_count == len(records)
            else "mixed_or_incomplete",
            "present_records": settings_count,
            "included_records": len(records),
        }
        if result["settings"]["status"] == "common":
            identity, value = next(iter(settings.items()))
            result["settings"].update(
                digest=identity,
                preview=_short_context(
                    _configuration_preview(value)["text"], TEXT_DETAIL_LIMIT
                ),
            )
    if projection_count:
        result["input_projection"] = {
            "present_records": projection_count,
            "included_records": len(records),
            "configuration_count": len(projections),
            "configurations": [projections[key] for key in sorted(projections)[:8]],
            "note": "At most eight mapping identities are shown; original structured inputs remain in retained records.",
        }
    return result


def _interval_observed(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "Unavailable; incomplete planned schedule"
    return (
        f"mean {interval['mean']}; lower {interval['lower']}; upper {interval['upper']}"
    )


def _policy_checks(
    analysis: dict[str, Any], policy: dict[str, Any]
) -> tuple[CheckView, ...]:
    """Expose numerical requirements without recomputing inference decisions."""
    counts = analysis["counts"]
    role = policy["decision_role"]
    complete = counts["incomplete_trials"] == 0
    checks = [
        CheckView(
            name="Schedule completeness",
            observed=f"{counts['completed_trials']}/{counts['expected_trials']} completed trials",
            required=f"{counts['expected_trials']}/{counts['expected_trials']} completed trials ({role})",
            passed=True if complete else None,
        ),
        CheckView(
            name="Independent units",
            observed=f"{counts['complete_units']} complete / {counts['scheduled_units']} scheduled",
            required=f"at least {policy['minimum_units']} complete independent units ({role})",
            passed=True
            if complete and counts["complete_units"] >= policy["minimum_units"]
            else None,
        ),
    ]
    gates = {gate["name"]: gate for gate in analysis["gates"]}
    names = ["paired_effect"]
    if policy["subject_bound"] is not None:
        names.append("subject_bound")
    for name in names:
        interval = analysis[
            "effect_interval" if name == "paired_effect" else "subject_interval"
        ]
        gate = gates.get(name)
        decision = gate["decision"] if gate else "insufficient_evidence"
        width = None
        if interval is not None:
            with localcontext(Context(prec=100)):
                width = Decimal(interval["upper"]) - Decimal(interval["lower"])
        checks.append(
            CheckView(
                name=f"{name} precision",
                observed=str(width) if width is not None else "Unavailable",
                required=f"interval width <= {policy['maximum_interval_width']} ({role})",
                passed=True
                if width is not None
                and width <= Decimal(policy["maximum_interval_width"])
                else None,
            )
        )
        threshold = Decimal(
            policy["allowed_degradation"]
            if name == "paired_effect"
            else policy["subject_bound"]
        )
        if name == "paired_effect" and policy["direction"] == "higher":
            threshold = threshold.copy_negate()
        requirement = (
            f"lower >= {threshold}; adverse if upper < {threshold}"
            if policy["direction"] == "higher"
            else f"upper <= {threshold}; adverse if lower > {threshold}"
        )
        outcome = {
            "pass": "satisfied",
            "regression": "adverse",
            "insufficient_evidence": "inconclusive",
        }[decision]
        checks.append(
            CheckView(
                name=name,
                observed=f"{_interval_observed(interval)}; {outcome}",
                required=f"{requirement} ({role})",
                passed=True
                if decision == "pass"
                else False
                if decision == "regression"
                else None,
                explanation=", ".join(gate["reasons"] if gate else analysis["reasons"]),
            )
        )
    return tuple(checks)


def _view(
    publication: Any,
    artifacts: dict[str, dict[str, Any]],
    *,
    case_ids: tuple[str, ...] = (),
) -> tuple[ReportView, dict[str, Any]]:
    plan = artifacts["plan"]
    per_case_references = plan["prompt"].get("reference_mode") == "per_case"
    policy = artifacts["analysis_policy"]
    analysis = publication.analysis_result.to_dict()
    effect = analysis["effect_interval"]
    subject = analysis["subject_interval"]
    counts = analysis["counts"]
    role = policy["decision_role"]
    required = role == "required"
    explanation = {
        "pass": f"All declared {role} bounds are satisfied.",
        "regression": f"At least one declared {role} bound is violated.",
        "insufficient_evidence": f"The declared {role} bounds are not established by the available evidence.",
    }[analysis["decision"]]
    if analysis["reasons"]:
        explanation += " " + ", ".join(analysis["reasons"])
    checks = _policy_checks(analysis, policy)
    # Floats are display geometry only. Decision arithmetic and exact strings are retained.
    interval = None
    if effect is not None:
        threshold = Decimal(policy["allowed_degradation"])
        if policy["direction"] == "higher":
            threshold = -threshold
        interval = IntervalView(
            lower=float(effect["lower"]),
            upper=float(effect["upper"]),
            estimate=float(effect["mean"]),
            threshold=float(threshold),
            label="Paired independent-unit effect interval",
            unit="normalized rating",
            threshold_direction="minimum"
            if policy["direction"] == "higher"
            else "maximum",
            neutral=0.0,
        )
    baseline_mean = (
        _baseline_mean(plan, artifacts["measurements"]) if subject is not None else None
    )
    with localcontext(Context(prec=100)):
        confidence = (
            format((1 - Decimal(policy["alpha"])) * 100, "f").rstrip("0").rstrip(".")
        )
    metric = MetricView(
        name=policy["metric_name"],
        scope="Fixed benchmark; equal independent-unit weights"
        if required
        else "Advisory metric; fixed benchmark; equal independent-unit weights",
        decision=analysis["decision"],
        baseline=number(float(baseline_mean))
        if baseline_mean is not None
        else "Unavailable",
        candidate=number(float(subject["mean"]))
        if subject is not None
        else "Unavailable",
        change=number(float(effect["mean"])) if effect is not None else "Unavailable",
        count=_count_label(counts["scheduled_cases"], "case"),
        explanation=explanation,
        checks=checks,
        interval=interval,
        notes=(
            f"{_count_label(counts['scheduled_units'], 'independent unit')}; "
            f"{counts['completed_trials']}/{counts['expected_trials']} completed "
            f"{'trial' if counts['expected_trials'] == 1 else 'trials'}.",
            "Decision role: required."
            if required
            else "Decision role: advisory; this metric does not gate required decisions.",
            f"Allowed degradation: {policy['allowed_degradation']} ({policy['direction']} is better).",
            f"Minimum units: {policy['minimum_units']}; maximum interval width: {policy['maximum_interval_width']}.",
            f"Two-sided Hoeffding intervals ({analysis['method']}); family confidence at least {confidence}% "
            f"(alpha {policy['alpha']}; comparison family size {policy['comparison_family_size']}); "
            "Bonferroni error allocation alpha / comparison family size per interval.",
            "Effect is subject minus baseline; tabulated interval endpoints and widths retain analysis precision.",
            "Inconclusive checks do not establish a bound: completeness, minimum units and precision must be met before an interval can establish satisfaction or an adverse result.",
            *(
                (
                    f"Subject interval (descriptive; no subject bound configured): {_interval_observed(subject)}.",
                )
                if policy["subject_bound"] is None
                else ()
            ),
            "Baseline and subject means describe the same complete schedule with equal independent-unit weights.",
            "Repetitions do not increase the number of independent units.",
        ),
    )
    baseline_rows = {row["id"]: row for row in artifacts["baseline_run"]["records"]}
    subject_rows = {row["id"]: row for row in artifacts["subject_run"]["records"]}
    available_case_ids = tuple(
        sorted(item["case_id"] for item in plan["sampling"]["case_units"])
    )
    selected_case_ids = _selected_case_ids(case_ids, available_case_ids)
    trials_by_case: dict[str, list[dict[str, Any]]] = {
        case_id: [] for case_id in selected_case_ids
    }
    for trial in artifacts["measurements"]["trials"]:
        if trial["case_id"] in trials_by_case:
            trials_by_case[trial["case_id"]].append(
                {
                    "side": trial["side"],
                    "repetition": trial["repetition"],
                    "status": trial["status"],
                    "parse": trial["parse"],
                    "attempts": [
                        {
                            "status": attempt["status"],
                            "resolved_model": attempt["resolved_model"],
                            "response_excerpt": attempt["response"]["text"][
                                :TEXT_DETAIL_LIMIT
                            ]
                            if attempt["response"] is not None
                            else None,
                            "error": attempt["error"],
                        }
                        for attempt in trial["attempts"]
                    ],
                }
            )
    details = []
    for case_id in selected_case_ids:
        details.append(
            (
                case_id,
                {
                    "input_excerpt": str(baseline_rows[case_id]["input"])[
                        :TEXT_DETAIL_LIMIT
                    ],
                    "baseline_answer_excerpt": str(baseline_rows[case_id]["output"])[
                        :TEXT_DETAIL_LIMIT
                    ],
                    "subject_answer_excerpt": str(subject_rows[case_id]["output"])[
                        :TEXT_DETAIL_LIMIT
                    ],
                    "trials": trials_by_case[case_id],
                    **(
                        {
                            "reference_excerpt": baseline_rows[case_id]["expected"][
                                :TEXT_DETAIL_LIMIT
                            ]
                        }
                        if per_case_references
                        else {}
                    ),
                },
            )
        )
    signed = publication.envelope["signature"] is not None
    signer = publication.envelope["signer"]
    assurance = {
        "authentication": "signature_present_not_recipient_authorized"
        if signed
        else "unsigned",
        "replay": "completed_without_recipient_authorization",
        "policy_decision": analysis["decision"],
        "decision_role": role,
        "recipient_acceptance": "not_performed",
        "signer_identity": signer["identity"] if signer is not None else None,
        "signer_public_key_sha256": signer["public_key_sha256"]
        if signer is not None
        else None,
    }
    resolved_models = sorted(
        {
            attempt["resolved_model"]
            for trial in artifacts["measurements"]["trials"]
            for attempt in trial["attempts"]
            if attempt["resolved_model"] is not None
        }
    )
    facts = {
        "assurance": assurance,
        "baseline": artifacts["baseline_run"]["run_id"],
        "subject": artifacts["subject_run"]["run_id"],
        "comparison": {
            "baseline": {
                "run_id": artifacts["baseline_run"]["run_id"],
                "artifact_digest": artifacts["baseline_run"]["artifact_digest"],
                "source": artifacts["baseline_run"]["source"],
            },
            "subject": {
                "run_id": artifacts["subject_run"]["run_id"],
                "artifact_digest": artifacts["subject_run"]["artifact_digest"],
                "source": artifacts["subject_run"]["source"],
            },
        },
        "judge": plan["judge"],
        "prompt": {
            **({"reference_mode": "per_case"} if per_case_references else {}),
            "system_excerpt": plan["prompt"]["system"][:TEXT_DETAIL_LIMIT],
            "template_excerpt": plan["prompt"]["template"][:TEXT_DETAIL_LIMIT],
            "demonstrations": len(plan["prompt"]["demonstrations"]),
            "references": [
                {"id": item["id"], "sha256": item["sha256"]}
                for item in plan["prompt"]["references"]
            ],
        },
        "resolved_models": resolved_models,
        "rubric": {
            "sha256": plan["rubric"]["sha256"],
            "excerpt": plan["rubric"]["text"][:TEXT_DETAIL_LIMIT],
        },
        "scale": plan["scale"],
        "analysis": analysis,
        "descriptive_means": {
            "baseline": baseline_mean,
            "subject": subject["mean"] if subject is not None else None,
            "change": effect["mean"] if effect is not None else None,
            "weighting": "equal_independent_units",
        },
        "detail_limits": {
            "shown_cases": len(details),
            "total_cases": len(available_case_ids),
            "text_excerpt_characters": TEXT_DETAIL_LIMIT,
            "selection": "requested" if case_ids else "first_by_case_id",
            "case_ids": list(selected_case_ids),
        },
    }
    if per_case_references:
        facts["per_case_references"] = [
            {
                "case_id": case_id,
                "reference_excerpt": baseline_rows[case_id]["expected"][
                    :TEXT_DETAIL_LIMIT
                ],
            }
            for case_id in selected_case_ids
        ]
    native = artifacts.get("native_capture")
    native_context: list[tuple[str, str]] = []
    native_assurance: tuple[tuple[str, str], ...] = ()
    native_identity: list[tuple[str, str]] = []
    captured_subjects: tuple[tuple[str, str], ...] | None = None
    if native is not None:
        requested = native["normalized_request"]["comparison"]
        same_artifact = (
            artifacts["baseline_run"]["artifact_digest"]
            == artifacts["subject_run"]["artifact_digest"]
        )
        same_settings = (
            requested["baseline"]["runtime"] == requested["subject"]["runtime"]
        )
        intent = (
            ("Same artifacts" if same_artifact else "Different artifacts")
            + "; "
            + (
                "same runtime settings"
                if same_settings
                else "different runtime settings"
            )
        )
        observations = native["normalized_request"].get("observations", [])
        facts["native_capture"] = {
            "sha256": publication.envelope["bindings"]["native_capture_sha256"],
            "execution_mode": native["normalized_request"]["execution"]["mode"],
            "same_artifact": same_artifact,
            "same_runtime_settings": same_settings,
            "comparison_intent": intent,
            "observations": observations,
        }
        native_context.append(("Comparison inputs", intent))
        for side in ("baseline", "subject"):
            model_id = requested[side]["artifact"]["model_id"]
            facts[side] = model_id
            runtime = requested[side]["runtime"]
            runtime_digest = artifacts[f"{side}_run"]["records"][0]["context"][
                "runtime_digest"
            ]
            facts["comparison"][side].update(
                model_id=model_id,
                runtime=runtime,
                runtime_digest=runtime_digest,
            )
            native_context.append((f"{side.title()} provider", runtime["provider"]))
            native_context.append(
                (
                    f"{side.title()} runtime settings",
                    canonical_payload(runtime["settings"]).decode("utf-8"),
                )
            )
            native_identity.extend(
                (
                    (f"{side.title()} runtime", runtime_digest),
                    (f"{side.title()} run", artifacts[f"{side}_run"]["run_id"]),
                )
            )
        native_context.append(
            (
                "Retained observations",
                "; ".join(
                    f"{item['id']} ({item['scope']}, {item['kind']})"
                    for item in observations
                )
                or "None",
            )
        )
        native_assurance = (
            (
                "Native runtime provenance",
                "Replayed offline from retained provider reports, runtime manifests, configurations, receipts, artifact identities, and scoring observations.",
            ),
        )
        native_identity.append(("Native capture", facts["native_capture"]["sha256"]))
    else:
        inputs = {side: artifacts[f"{side}_run"] for side in ("baseline", "subject")}
        captured_subjects, context, changes, context_details = _captured_context(inputs)
        native_context.extend(context)
        native_context.extend(("Captured comparison", change) for change in changes)
        native_identity.extend(_captured_identities(inputs))
        details.extend(context_details)
        captured_facts: dict[str, Any] = {"context": dict(context), "changes": changes}
        for side, run in inputs.items():
            facts["comparison"][side].update(
                run_digest=run_digest(run), source_digest=run["source_digest"]
            )
            if "service_identity" in run:
                service = run["service_identity"]
                facts["comparison"][side]["service_identity"] = {
                    key: value
                    for key, value in service.items()
                    if key != "configuration"
                }
            native_identity.append(
                (
                    side.title() + " source digest",
                    run["source_digest"] or "Unavailable in captured run",
                )
            )
            recorded = _captured_record_context(run["records"])
            captured_facts[side] = recorded
            for label, value in recorded.items():
                details.append(
                    (side.title() + " recorded " + label.replace("_", " "), value)
                )
                heading = side.title() + " recorded " + label.replace("_", " ")
                if label == "settings":
                    native_context.append(
                        (
                            heading,
                            value.get("preview", "Mixed or incomplete across records"),
                        )
                    )
                    if "digest" in value:
                        native_identity.append((heading, value["digest"]))
                else:
                    native_context.append(
                        (
                            heading,
                            f"{value['present_records']} of {value['included_records']} records; {value['configuration_count']} mapping configurations",
                        )
                    )
                    for configuration in value["configurations"]:
                        native_context.append(
                            (
                                side.title() + " projection pointer",
                                configuration["pointer"],
                            )
                        )
                        native_identity.append(
                            (
                                side.title() + " projection configuration",
                                configuration["configuration_digest"],
                            )
                        )
        facts["captured_context"] = captured_facts
        native_assurance = (
            (
                "Captured answer provenance",
                "Evaluator, artifact, configuration and model labels are source assertions retained with the captured runs; runtime execution has not been independently established.",
            ),
        )
    view = ReportView(
        title="InvarLock bounded judge report",
        family="Bounded judge measurement evidence",
        decision=analysis["decision"],
        summary="Comparison of repeated judgments of frozen baseline and subject answers on the declared benchmark."
        + (
            ""
            if required
            else " This metric is advisory and does not gate required decisions."
        ),
        metrics=(metric,),
        assurance=native_assurance
        + (
            (
                "Authentication",
                "Signature present; recipient signer authorization has not been performed."
                if signed
                else "Unsigned evidence; no signer authentication.",
            ),
            (
                "Measurement and analysis replay",
                "Completed offline against the retained plan and source records.",
            ),
            ("Policy result", analysis["decision"]),
            ("Recipient acceptance", "Not performed by report."),
        ),
        subjects=captured_subjects
        if captured_subjects is not None
        else (("Baseline", facts["baseline"]), ("Subject", facts["subject"])),
        context=tuple(native_context)
        + (
            (
                (
                    "Per-case references",
                    "Enabled; each judge request includes its case reference separately from the model input.",
                ),
            )
            if per_case_references
            else ()
        )
        + (
            ("Judge provider", plan["judge"]["provider"]),
            ("Requested judge", plan["judge"]["requested_model"]),
            ("Resolved judges", ", ".join(resolved_models) or "Unavailable"),
            (
                "Reasoning effort",
                plan["judge"]["config"]["reasoning_effort"] or "Not configured",
            ),
            ("Rubric", plan["rubric"]["text"][:TEXT_DETAIL_LIMIT]),
            ("Judge prompt", plan["prompt"]["template"][:TEXT_DETAIL_LIMIT]),
            ("Prompt demonstrations", str(len(plan["prompt"]["demonstrations"]))),
            ("Prompt references", str(len(plan["prompt"]["references"]))),
            (
                "Scale",
                ", ".join(
                    f"{rating['label']} = {rating['value']}"
                    for rating in plan["scale"]["ratings"]
                ),
            ),
            ("Coverage", metric.count),
        ),
        identity=tuple(native_identity)
        + tuple(
            (side.title() + " artifact", artifacts[f"{side}_run"]["artifact_digest"])
            for side in ("baseline", "subject")
            if artifacts[f"{side}_run"]["artifact_digest"] is not None
        )
        + (
            ("Plan", publication.envelope["bindings"]["plan_sha256"]),
            ("Intended subject", publication.envelope["intended_subject"]),
            (
                "Signer",
                signer["identity"] if signer is not None else "Unsigned",
            ),
            (
                "Signer key",
                signer["public_key_sha256"] if signer is not None else "Unavailable",
            ),
        ),
        next_steps=(
            ()
            if signed
            else (
                "For recipient verification, republish the same retained inputs to a new evidence destination with evaluate --signing-key and the declared signer identity. Do not modify this evidence bundle.",
            )
        )
        + (
            "Use verify with an independently maintained judge recipient policy before relying on this result.",
        ),
        limitations=(
            analysis["estimand"],
            (
                f"Details show the {len(selected_case_ids)} requested cases and at most {TEXT_DETAIL_LIMIT} characters per text excerpt; complete data remains in evidence."
                if case_ids
                else f"Details show the first {min(CASE_DETAIL_LIMIT, len(available_case_ids))} cases by case ID and at most {TEXT_DETAIL_LIMIT} characters per text excerpt; use report --case-id to select any retained case."
            ),
            (
                "Native runtime provenance binds the captured model answers. Judge measurements do not establish immunity to prompt injection."
                if native is not None
                else "A retained judgment does not establish model execution or immunity to prompt injection."
            ),
            "Verification replays retained measurements offline; it does not remeasure the evaluated service or independently establish that every judge rating is correct.",
        ),
        details=tuple(details),
        technical={
            **(
                {"native_capture": facts["native_capture"]}
                if native is not None
                else {"captured_context": facts["captured_context"]}
            ),
            "method": analysis["method"],
            "assumptions": analysis["assumptions"],
            "policy": policy,
            "effect_interval": effect,
            "subject_interval": subject,
        },
    )
    return view, facts


def render_judge_evidence(
    evidence: Path,
    *,
    html_path: Path | None = None,
    markdown_path: Path | None = None,
    junit_path: Path | None = None,
    explain: bool = False,
    case_ids: tuple[str, ...] = (),
) -> JudgeEvidenceReport:
    from invarlock.judge_measurements.evidence import object_sha256

    requested = {
        name: str(path)
        for name, path in (
            ("html", html_path),
            ("markdown", markdown_path),
            ("junit", junit_path),
        )
        if path is not None
    }
    written: dict[str, str] = {}
    failed: str | None = None
    try:
        validate_report_destinations(requested, evidence=evidence)
        publication, artifacts = _snapshot(evidence)
        view, facts = _view(publication, artifacts, case_ids=case_ids)
        text = render_markdown(view, include_details=explain)
        required = facts["assurance"]["decision_role"] == "required"

        def render_junit() -> bytes:
            suite = Element(
                "testsuite",
                name="InvarLock bounded judge policy",
                tests="1",
                failures=str(int(required and view.decision == "regression")),
                errors=str(int(required and view.decision == "insufficient_evidence")),
                skipped=str(int(not required)),
            )
            properties = SubElement(suite, "properties")
            SubElement(
                properties,
                "property",
                name="decision_role",
                value="required" if required else "advisory",
            )
            SubElement(
                properties, "property", name="policy_decision", value=view.decision
            )
            case = SubElement(
                suite,
                "testcase",
                name=xml_text(view.metrics[0].name),
                classname="bounded-judge-fixed-benchmark-v1",
            )
            if not required:
                SubElement(
                    case,
                    "skipped",
                    message=f"Advisory metric: {view.decision}; required decisions are not gated.",
                )
            elif view.decision != "pass":
                SubElement(
                    case,
                    "failure" if view.decision == "regression" else "error",
                    message=xml_text(view.metrics[0].explanation),
                )
            SubElement(
                case, "system-out"
            ).text = "Offline replay completed. Recipient signer authorization and acceptance were not performed."
            return cast(bytes, tostring(suite, encoding="utf-8", xml_declaration=True))

        written.update(
            publish_report_outputs(
                requested,
                {
                    "html": lambda: render_html(view).encode(),
                    "markdown": lambda: text.encode(),
                    "junit": render_junit,
                },
                evidence=evidence,
            )
        )
        return JudgeEvidenceReport(
            text=text,
            kind="judge",
            pack_manifest_digest=object_sha256(publication.envelope),
            requested_outputs=requested,
            written_outputs=written,
            facts=facts,
        )
    except ReportPublicationError as exc:
        written.update(exc.written_outputs)
        failed = exc.failed_output
        payload = {
            "format_version": "invarlock/judge-evidence-report-v1",
            "kind": "judge",
            "ok": False,
            "errors": [str(exc)[:1024]],
            "requested_outputs": requested,
            "written_outputs": written,
            "failed_output": failed,
        }
        raise EvidenceReportError(str(exc), payload=payload) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {
            "format_version": "invarlock/judge-evidence-report-v1",
            "kind": "judge",
            "ok": False,
            "errors": [str(exc)[:1024]],
            "requested_outputs": requested,
            "written_outputs": written,
            "failed_output": failed,
        }
        raise EvidenceReportError(str(exc), payload=payload) from exc
