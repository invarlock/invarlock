"""Bounded presentation of retained judge evidence without recipient acceptance."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from invarlock.captured_contracts import atomic_write
from invarlock.engine import run_digest
from invarlock.evidence_reporting import EvidenceReportError, EvidenceReportV2
from invarlock.judge_measurements.contracts import canonical_payload
from invarlock.report_presentation import (
    CheckView,
    IntervalView,
    MetricView,
    ReportView,
    render_html,
    render_markdown,
    xml_text,
)

CASE_DETAIL_LIMIT = 50
TEXT_DETAIL_LIMIT = 2000


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
    for name, digest_field in (
        ("plan", "plan_sha256"),
        ("measurements", "measurements_sha256"),
        ("analysis_policy", "analysis_policy_sha256"),
        ("baseline_run", "baseline_run_sha256"),
        ("subject_run", "subject_run_sha256"),
    ):
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


def _view(
    publication: Any, artifacts: dict[str, dict[str, Any]]
) -> tuple[ReportView, dict[str, Any]]:
    plan = artifacts["plan"]
    policy = artifacts["analysis_policy"]
    analysis = publication.analysis_result.to_dict()
    effect = analysis["effect_interval"]
    subject = analysis["subject_interval"]
    counts = analysis["counts"]
    checks = tuple(
        CheckView(
            name=gate["name"],
            observed=gate["decision"],
            required="pass",
            passed=True
            if gate["decision"] == "pass"
            else False
            if gate["decision"] == "regression"
            else None,
            explanation=", ".join(gate["reasons"]),
        )
        for gate in analysis["gates"]
    )
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
        )
    metric = MetricView(
        name=policy["metric_name"],
        scope="Fixed benchmark; equal independent-unit weights",
        decision=analysis["decision"],
        baseline="Frozen baseline answers",
        candidate=subject["mean"] if subject is not None else "Unavailable",
        change=effect["mean"] if effect is not None else "Unavailable",
        count=f"{counts['scheduled_cases']} cases; {counts['scheduled_units']} independent units; {counts['completed_trials']}/{counts['expected_trials']} completed trials",
        explanation=", ".join(analysis["reasons"])
        or "All declared required bounds are satisfied.",
        checks=checks,
        interval=interval,
        notes=(
            f"Allowed degradation: {policy['allowed_degradation']} ({policy['direction']} is better).",
            f"Minimum units: {policy['minimum_units']}; maximum interval width: {policy['maximum_interval_width']}.",
            "Repetitions do not increase the number of independent units.",
        ),
    )
    baseline_rows = {row["id"]: row for row in artifacts["baseline_run"]["records"]}
    subject_rows = {row["id"]: row for row in artifacts["subject_run"]["records"]}
    details = []
    for case_id in sorted(baseline_rows)[:CASE_DETAIL_LIMIT]:
        trials = []
        for trial in artifacts["measurements"]["trials"]:
            if trial["case_id"] == case_id:
                trials.append(
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
                    "trials": trials,
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
        "detail_limits": {
            "shown_cases": len(details),
            "total_cases": len(baseline_rows),
            "text_excerpt_characters": TEXT_DETAIL_LIMIT,
        },
    }
    view = ReportView(
        title="InvarLock bounded judge report",
        family="Bounded judge measurement evidence",
        decision=analysis["decision"],
        summary="Comparison of repeated judgments of frozen baseline and subject answers on the declared benchmark.",
        metrics=(metric,),
        assurance=(
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
        subjects=(("Baseline", facts["baseline"]), ("Subject", facts["subject"])),
        context=(
            ("Judge provider", plan["judge"]["provider"]),
            ("Requested judge", plan["judge"]["requested_model"]),
            ("Resolved judges", ", ".join(resolved_models) or "Unavailable"),
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
        identity=(
            ("Baseline artifact", artifacts["baseline_run"]["artifact_digest"]),
            ("Subject artifact", artifacts["subject_run"]["artifact_digest"]),
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
            "Use verify with an independently maintained judge recipient policy before relying on this result.",
        ),
        limitations=(
            analysis["estimand"],
            f"Details show at most {CASE_DETAIL_LIMIT} cases and {TEXT_DETAIL_LIMIT} characters per text excerpt; complete data remains in evidence.",
            "A retained judgment does not establish model execution or immunity to prompt injection.",
        ),
        details=tuple(details),
        technical={
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
        destinations: set[Path] = set()
        for value in requested.values():
            destination = Path(value).absolute()
            canonical = destination.resolve()
            if canonical.is_relative_to(
                Path(evidence).resolve()
            ) or destination.is_relative_to(Path(evidence).absolute()):
                raise EvidenceReportError(
                    "report destination must remain outside immutable evidence"
                )
            if any(
                canonical.is_relative_to(other) or other.is_relative_to(canonical)
                for other in destinations
            ):
                raise EvidenceReportError("report destinations collide")
            if destination.exists() or destination.is_symlink():
                raise EvidenceReportError("report destination already exists")
            for parent in destination.parents:
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise EvidenceReportError(
                        "report destination parent must be a real directory"
                    )
            destinations.add(canonical)
        publication, artifacts = _snapshot(evidence)
        view, facts = _view(publication, artifacts)
        text = render_markdown(view, include_details=explain)
        rendered = {"html": render_html(view).encode(), "markdown": text.encode()}
        suite = Element(
            "testsuite",
            name="InvarLock bounded judge policy",
            tests="1",
            failures=str(int(view.decision == "regression")),
            errors=str(int(view.decision == "insufficient_evidence")),
        )
        case = SubElement(
            suite,
            "testcase",
            name=xml_text(view.metrics[0].name),
            classname="bounded-judge-fixed-benchmark-v1",
        )
        if view.decision != "pass":
            SubElement(
                case,
                "failure" if view.decision == "regression" else "error",
                message=xml_text(view.metrics[0].explanation),
            )
        SubElement(
            case, "system-out"
        ).text = "Offline replay completed. Recipient signer authorization and acceptance were not performed."
        rendered["junit"] = tostring(suite, encoding="utf-8", xml_declaration=True)
        for name, destination_text in requested.items():
            failed = name
            atomic_write(Path(destination_text), rendered[name])
            written[name] = destination_text
        return JudgeEvidenceReport(
            text=text,
            kind="judge",
            pack_manifest_digest=object_sha256(publication.envelope),
            requested_outputs=requested,
            written_outputs=written,
            facts=facts,
        )
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
