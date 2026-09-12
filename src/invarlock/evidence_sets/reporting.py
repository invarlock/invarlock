"""One comparison view with distinct component methods and trust scopes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from invarlock.captured_contracts import atomic_write, read_file, secure_directory, sha
from invarlock.captured_reporting import _load as captured_snapshot
from invarlock.captured_reporting import _view as captured_view
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_reporting import EvidenceReportError, EvidenceReportV2
from invarlock.evidence_sets.contracts import (
    CONTROL_LIMIT,
    INDEX_FILE,
    STATISTICAL_SCOPE,
    EvidenceSetError,
    check_statements,
    load_index,
    member_path,
    read_object,
    require_external,
)
from invarlock.evidence_sets.verification import (
    require_deterministic_policy,
    shared_captured_inputs,
)
from invarlock.judge_measurements.reporting import _snapshot as judge_snapshot
from invarlock.judge_measurements.reporting import _view as judge_view
from invarlock.report_presentation import (
    ReportView,
    render_html,
    render_markdown,
    xml_text,
)


@dataclass(frozen=True)
class EvidenceSetReport(EvidenceReportV2):
    facts: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return canonical_json_bytes(
            {
                "format_version": "invarlock/evidence-set-report-v1",
                "kind": "evidence_set",
                "ok": not self.errors,
                "index_sha256": self.pack_manifest_digest,
                "requested_outputs": self.requested_outputs,
                "written_outputs": self.written_outputs,
                "errors": list(self.errors),
                **self.facts,
            }
        ).decode()


def build_evidence_set_view(
    root: Path, *, case_ids: tuple[str, ...] = ()
) -> tuple[ReportView, dict[str, Any], str]:
    """Check shared retained inputs; rendering never authorizes recipients."""
    root = Path(root).absolute()
    with secure_directory(root):
        index, raw = load_index(root)
        check_statements(root, index)
        captured_path = member_path(root, index["members"]["deterministic"]["path"])
        judge_path = member_path(root, index["members"]["judge"]["path"])
        manifest, payloads, signer, _ = captured_snapshot(captured_path)
        if (
            sha(canonical_json_bytes(manifest))
            != index["members"]["deterministic"]["statement_sha256"]
        ):
            raise EvidenceSetError(
                "captured statement changed during report preparation"
            )
        require_deterministic_policy(payloads["policy"])
        shared = shared_captured_inputs(payloads)
        expected_envelope, envelope_raw = read_object(judge_path / "envelope.json")
        if sha(envelope_raw) != index["members"]["judge"]["statement_sha256"]:
            raise EvidenceSetError("judge statement changed before rendering")
        publication, artifacts = judge_snapshot(judge_path)
        if canonical_json_bytes(publication.envelope) != canonical_json_bytes(
            expected_envelope
        ):
            raise EvidenceSetError("judge statement changed during rendering")
        for field in ("baseline_run_sha256", "subject_run_sha256", "case_set_sha256"):
            if publication.envelope["bindings"][field] != shared[field]:
                raise EvidenceSetError("component runs or case sets differ")
        if (
            publication.envelope["intended_subject"]
            != shared["subject_artifact_sha256"]
        ):
            raise EvidenceSetError("component subject artifacts differ")
        if artifacts["analysis_policy"]["decision_role"] != "required":
            raise EvidenceSetError("evidence set requires a required judge policy")
        first = captured_view(manifest, payloads, signer)
        second, judge_facts = judge_view(publication, artifacts, case_ids=case_ids)
        check_statements(root, index)
        if read_file(root / INDEX_FILE, CONTROL_LIMIT) != raw:
            raise EvidenceSetError("evidence set index changed during rendering")
    decisions = {first.decision, second.decision}
    decision = (
        "regression"
        if "regression" in decisions
        else "insufficient_evidence"
        if "insufficient_evidence" in decisions
        else "pass"
    )
    metrics = tuple(
        replace(
            metric,
            name=f"Deterministic · {metric.name}",
            notes=(
                *metric.notes,
                "Captured component: original cases; marginal component intervals.",
            ),
        )
        for metric in first.metrics
    )
    metrics += tuple(
        replace(metric, name=f"Judge · {metric.name}") for metric in second.metrics
    )
    assurance = (
        (
            "Deterministic authentication",
            "Signed manifest verified."
            if manifest["authentication"] == "signed"
            else "Unsigned local evidence.",
        ),
        (
            "Deterministic scoring replay",
            "Not performed by report; recorded component result shown.",
        ),
        (
            "Judge authentication",
            "Signature present; recipient authorization not performed."
            if publication.envelope["signature"] is not None
            else "Unsigned local evidence.",
        ),
        ("Judge measurement replay", "Recomputed offline from retained measurements."),
        (
            "Shared inputs",
            "Same complete baseline and subject runs, original case set, and subject artifact.",
        ),
        ("Recipient acceptance", "Not performed by report."),
    )
    facts = {
        "decision": decision,
        "statistical_scope": STATISTICAL_SCOPE,
        "recipient_acceptance": "not_performed",
        "shared_inputs": shared,
        "component_decisions": {
            "deterministic": first.decision,
            "judge": second.decision,
        },
        "judge": judge_facts["judge"],
    }
    view = ReportView(
        title="InvarLock comparison report",
        family="Deterministic and bounded judge evidence",
        decision=decision,
        summary="Both required components compare the same frozen baseline and subject answers. Each component retains its own statistical method and acceptance requirements.",
        metrics=metrics,
        assurance=assurance,
        subjects=first.subjects,
        context=(*first.context, *second.context),
        changes=first.changes,
        identity=(
            ("Evidence set", sha(raw)),
            ("Baseline run", shared["baseline_run_sha256"]),
            ("Subject run", shared["subject_run_sha256"]),
            ("Original case set", shared["case_set_sha256"]),
            ("Subject artifact", shared["subject_artifact_sha256"]),
        ),
        next_steps=(
            "Verify the evidence set with an independently maintained composition recipient policy before accepting the combined result.",
        ),
        limitations=(
            "The combined decision is a conjunction of component decisions. It provides no joint confidence guarantee.",
            "Deterministic metrics count original cases. Judge repetitions do not increase that count or the number of independent units.",
            *first.limitations,
            *second.limitations,
        ),
        details=tuple(
            (f"Deterministic · {name}", detail) for name, detail in first.details
        )
        + tuple((f"Judge · {name}", detail) for name, detail in second.details),
        technical={
            "statistical_scope": STATISTICAL_SCOPE,
            "deterministic": first.technical,
            "judge": second.technical,
        },
    )
    return view, facts, sha(raw)


def render_evidence_set(
    evidence: Path,
    *,
    html_path: Path | None = None,
    markdown_path: Path | None = None,
    junit_path: Path | None = None,
    explain: bool = False,
    case_ids: tuple[str, ...] = (),
) -> EvidenceSetReport:
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
    failed = None
    try:
        destinations: set[Path] = set()
        for value in requested.values():
            path = Path(value).absolute()
            require_external(path, evidence)
            resolved = path.resolve()
            if any(
                resolved.is_relative_to(other) or other.is_relative_to(resolved)
                for other in destinations
            ):
                raise EvidenceSetError("report destinations collide")
            if path.exists() or path.is_symlink():
                raise EvidenceSetError("report destination already exists")
            for parent in path.parents:
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise EvidenceSetError(
                        "report destination parent must be a real directory"
                    )
            destinations.add(resolved)
        view, facts, digest = build_evidence_set_view(evidence, case_ids=case_ids)
        text = render_markdown(view, include_details=explain)
        suite = Element(
            "testsuite",
            name="InvarLock combined component policy",
            tests=str(len(view.metrics)),
            failures=str(sum(m.decision == "regression" for m in view.metrics)),
            errors=str(
                sum(m.decision == "insufficient_evidence" for m in view.metrics)
            ),
        )
        for metric in view.metrics:
            case = SubElement(
                suite,
                "testcase",
                name=xml_text(metric.name),
                classname="same-answer-component-conjunction-v1",
            )
            if metric.decision != "pass":
                SubElement(
                    case,
                    "failure" if metric.decision == "regression" else "error",
                    message=xml_text(metric.explanation),
                )
            SubElement(
                case, "system-out"
            ).text = "Component policy result only; recipient acceptance not performed. No joint confidence guarantee."
        rendered = {
            "html": render_html(view).encode(),
            "markdown": text.encode(),
            "junit": tostring(suite, encoding="utf-8", xml_declaration=True),
        }
        for name, destination in requested.items():
            failed = name
            atomic_write(Path(destination), rendered[name])
            written[name] = destination
        return EvidenceSetReport(
            text=text,
            kind="evidence_set",
            pack_manifest_digest=digest,
            requested_outputs=requested,
            written_outputs=written,
            facts=facts,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceReportError(
            str(exc),
            payload={
                "format_version": "invarlock/evidence-set-report-v1",
                "kind": "evidence_set",
                "ok": False,
                "errors": [str(exc)[:1024]],
                "requested_outputs": requested,
                "written_outputs": written,
                "failed_output": failed,
            },
        ) from exc
