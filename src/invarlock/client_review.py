"""Build attributed client handoffs without modifying or approving evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from invarlock import __version__
from invarlock.captured_contracts import read_file
from invarlock.client_review_files import copy_evidence, file_hash, require_unchanged
from invarlock.evidence_pack_json import parse_json_bytes
from invarlock.filesystem.paths import pinned_directory
from invarlock.filesystem.staged_directory import staged_directory
from invarlock.report_presentation import (
    AuthoredReview,
    ReportView,
    render_html,
    render_markdown,
)
from invarlock.report_publication import validate_report_destinations

CONTEXT_FORMAT = "invarlock/client-review-context-v1"
CONTEXT_FORMAT_V2 = "invarlock/client-review-context-v2"
CONTEXT_LIMIT = 64 * 1024
RECEIPT_LIMIT = 8 * 1024**2
_RESERVED = {
    "brief.html",
    "brief.md",
    "review.html",
    "review.md",
    "review-context.json",
    "technical-report.html",
    "technical-report.md",
    "verification.html",
    "verification.md",
    "verification.receipt.json",
    "verification-snapshot.json",
    "original-file-hashes.json",
    "package-manifest.json",
    "readme.txt",
    "handoff.zip",
    "evidence",
}


class ClientReviewError(ValueError):
    """The handoff could not be prepared without weakening its boundaries."""


def load_review_context(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read a closed, bounded commentary document; absent decisions remain pending."""
    raw = read_file(path, CONTEXT_LIMIT)
    value = parse_json_bytes(raw, label="review context")
    required = {"format", "title", "review_date", "byline", "context", "next_action"}
    optional = {"label", "recommendation", "decision", "basis", "summary"}
    structured = isinstance(value, dict) and value.get("format") == CONTEXT_FORMAT_V2
    if structured:
        required |= {
            "intended_use",
            "excluded_uses",
            "decision_conditions",
            "decision_state",
        }
        optional |= {"exception_rationale"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise ClientReviewError("review context has missing or unsupported fields")
    if value["format"] not in {CONTEXT_FORMAT, CONTEXT_FORMAT_V2}:
        raise ClientReviewError("unsupported review context format")
    for key in (required | optional) - {"context", "basis", "summary"}:
        if key not in value:
            continue
        field = value[key]
        if (
            not isinstance(field, str)
            or not field.strip()
            or len(field) > 4000
            or any(ord(c) < 32 and c not in "\n\t" for c in field)
        ):
            raise ClientReviewError(
                f"review context {key} must be nonempty text of at most 4000 characters"
            )
    if structured and value["decision_state"] not in {
        "pending",
        "accepted",
        "rejected",
        "conditional",
        "exception",
    }:
        raise ClientReviewError("unknown owner decision_state")
    if (
        structured
        and value["decision_state"] == "exception"
        and not value.get("exception_rationale")
    ):
        raise ClientReviewError("an exception requires exception_rationale")
    for key in ("context", "basis", "summary"):
        rows = value.get(key, [])
        row_limit = 4 if key == "summary" else 32
        text_limit = 600 if key == "summary" else 4000
        if not isinstance(rows, list) or len(rows) > row_limit:
            raise ClientReviewError(
                f"review context {key} must contain at most {row_limit} labeled rows"
            )
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or any(
                    not isinstance(x, str)
                    or not x.strip()
                    or len(x) > text_limit
                    or any(ord(c) < 32 and c not in "\n\t" for c in x)
                    for x in row
                )
            ):
                raise ClientReviewError(
                    f"review context {key} rows require two nonempty strings of at most {text_limit} characters"
                )
    if not value["context"]:
        raise ClientReviewError(
            "review context requires intended-use or decision context"
        )
    try:
        if date.fromisoformat(value["review_date"]).isoformat() != value["review_date"]:
            raise ValueError
    except ValueError as exc:
        raise ClientReviewError("review_date must be YYYY-MM-DD") from exc
    return value, raw


def _view(evidence: Path) -> tuple[ReportView, str, str]:
    """Reuse each family's reporting validation and its exact statement identity."""
    from invarlock.captured_contracts import sha
    from invarlock.captured_reporting import _load, is_captured_manifest
    from invarlock.captured_reporting import _view as captured_view
    from invarlock.evidence_pack_contract import canonical_json_bytes
    from invarlock.evidence_reporting import _render_native_evidence
    from invarlock.evidence_sets.contracts import is_evidence_set
    from invarlock.evidence_sets.reporting import build_evidence_set_view
    from invarlock.judge_measurements.evidence import object_sha256
    from invarlock.judge_measurements.reporting import _snapshot, is_judge_evidence
    from invarlock.judge_measurements.reporting import _view as judge_view

    if is_evidence_set(evidence):
        view, _, digest = build_evidence_set_view(evidence)
        return view, "evidence_set", digest
    if is_judge_evidence(evidence):
        publication, artifacts = _snapshot(evidence)
        view, _ = judge_view(publication, artifacts)
        return view, "judge", object_sha256(publication.envelope)
    if is_captured_manifest(evidence):
        manifest, values, signer, _ = _load(evidence)
        return (
            captured_view(manifest, values, signer),
            "captured",
            sha(canonical_json_bytes(manifest)),
        )
    view, digest = _render_native_evidence(evidence)
    return view, "runtime", digest


def _receipt(path: Path, kind: str, digest: str) -> bytes:
    raw = read_file(path, RECEIPT_LIMIT)
    value = parse_json_bytes(raw, label="historical receipt")
    if not isinstance(value, dict):
        raise ClientReviewError("historical receipt must be a JSON object")
    statement = value.get("statement", {})
    if not isinstance(statement, dict):
        raise ClientReviewError("historical receipt statement must be an object")
    if kind == "evidence_set":
        claimed = value.get("index_sha256")
    elif kind == "judge":
        result = statement.get("result", {})
        claimed = result.get("envelope_sha256") if isinstance(result, dict) else None
    else:
        claimed = statement.get("pack_manifest_digest")
    if not isinstance(claimed, str) or claimed.removeprefix(
        "sha256:"
    ) != digest.removeprefix("sha256:"):
        raise ClientReviewError(
            "historical receipt does not bind this evidence statement"
        )
    return raw


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _supporting_files(files: Mapping[str, Path]) -> dict[str, bytes]:
    if len(files) > 32:
        raise ClientReviewError("at most 32 supporting files are allowed")
    names: set[str] = set()
    result: dict[str, bytes] = {}
    for name, path in files.items():
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or any(ord(c) < 32 for c in name)
            or name.casefold() in _RESERVED | names
        ):
            raise ClientReviewError(
                "supporting file name collides or is not a safe basename"
            )
        names.add(name.casefold())
        raw = read_file(Path(path), 32 * 1024**2)
        if b"PRIVATE KEY-----" in raw:
            raise ClientReviewError(
                "private keys must not be included in public attachments"
            )
        result[name] = raw
    return result


def build_client_review(
    evidence: Path,
    *,
    context_path: Path,
    output: Path,
    retained_receipt: Path | None = None,
    supporting_files: Mapping[str, Path] | None = None,
    receipt_expectations: Path | None = None,
) -> dict[str, Any]:
    """Publish a fresh directory and its ZIP, retaining all source evidence bytes.

    Supporting files are explicit public attachments, not trusted verifier inputs.
    Without receipt_expectations, a historical receipt is an unverified attachment.
    With independent expectations, the existing family-specific contract checks
    the receipt and binds a snapshot; current authorization is not established.
    """
    from invarlock.client_review_presentation import brief, verification_guide
    from invarlock.report_provenance import identity, renderer_identity, write_archive

    evidence, output = Path(evidence).absolute(), Path(output).absolute()
    try:
        validate_report_destinations({"package": output}, evidence=evidence)
        context, raw_context = load_review_context(Path(context_path))
        if receipt_expectations is not None and retained_receipt is None:
            raise ClientReviewError("receipt expectations require a retained receipt")
        attachments = _supporting_files(supporting_files or {})
        # Finish source binding checks before publishing the completed package.
        with staged_directory(
            output, prefix=".client-review-", create_parents=True
        ) as stage:
            with pinned_directory(evidence) as source:
                root = stage.path
                source_state, originals = copy_evidence(source, root / "evidence")
                view, kind, digest = _view(root / "evidence")
                original_hashes = {
                    "evidence/" + name: value for name, value in originals.items()
                }
                if retained_receipt is not None:
                    attachments["verification.receipt.json"] = _receipt(
                        Path(retained_receipt), kind, digest
                    )
                for name, raw in attachments.items():
                    (root / name).write_bytes(raw)
                    original_hashes[name] = file_hash(root / name)
                snapshot = None
                if receipt_expectations is not None:
                    from invarlock.report_verification import authenticate_receipt

                    snapshot = authenticate_receipt(
                        root / "verification.receipt.json",
                        root / "evidence",
                        inputs=Path(receipt_expectations),
                        kind=kind,
                        digest=digest,
                        submitted_evidence=evidence,
                    )
                    _json(root / "verification-snapshot.json", snapshot)
                (root / "review-context.json").write_bytes(raw_context)
                _json(root / "original-file-hashes.json", original_hashes)
                links: tuple[tuple[str, str], ...] = (
                    ("Decision brief", "brief.html"),
                    ("Full review", "review.html"),
                    ("Technical appendix", "technical-report.html"),
                    ("Verify this package", "verification.html"),
                )
                if retained_receipt is not None:
                    links += (
                        (
                            "Authenticated receipt snapshot"
                            if snapshot
                            else "Historical receipt (unverified attachment)",
                            "verification.receipt.json",
                        ),
                    )
                technical = replace(
                    view,
                    handoff_links=links,
                    metrics=tuple(
                        replace(m, evidence_source="evidence/" + m.evidence_source)
                        for m in view.metrics
                    ),
                )
                from invarlock.report_facts import generated_facts, owner_context

                owner_rows = owner_context(context, view)
                technical = replace(
                    technical, critical_facts=generated_facts(view) + owner_rows
                )
                if snapshot:
                    from invarlock.report_verification import verification_facts

                    technical = replace(
                        technical,
                        critical_facts=tuple(
                            (k, v)
                            for k, v in technical.critical_facts
                            if k not in {"Recipient verification", "Verification time"}
                        )
                        + verification_facts(snapshot),
                        assurance=technical.assurance + verification_facts(snapshot),
                    )
                sources = renderer_identity()
                renderer_digest = identity(sources)
                snapshot_digest = identity(
                    {
                        "renderer": renderer_digest,
                        "evidence": digest,
                        "context": file_hash(root / "review-context.json"),
                        "verification": snapshot,
                        "configuration": {
                            "delivery": "extracted-package",
                            "format": "client-review-v2",
                        },
                    }
                )
                technical = replace(
                    technical,
                    critical_facts=technical.critical_facts
                    + (("Report snapshot identity", snapshot_digest),),
                )
                authored = AuthoredReview(
                    label=context.get("label", "Authored client review"),
                    byline=f"{context['byline']} · {context['review_date']}",
                    context=owner_rows
                    + tuple(tuple(row) for row in context["context"]),
                    recommendation=context.get(
                        "recommendation", "No owner recommendation supplied."
                    ),
                    decision=context.get("decision", "Pending owner decision"),
                    basis=tuple(tuple(row) for row in context.get("basis", [])),
                    next_action=context["next_action"],
                    summary=tuple(tuple(row) for row in context.get("summary", [])),
                )
                review = replace(
                    technical,
                    title=context["title"],
                    review=authored,
                    metrics=tuple(
                        replace(m, evidence_source="evidence/" + m.evidence_source)
                        for m in view.metrics
                    ),
                )
                for name, item in (("technical-report", technical), ("review", review)):
                    (root / f"{name}.html").write_text(
                        render_html(item), encoding="utf-8"
                    )
                    (root / f"{name}.md").write_text(
                        render_markdown(item), encoding="utf-8"
                    )
                for suffix in ("html", "md"):
                    (root / f"brief.{suffix}").write_text(
                        brief(
                            review,
                            digest=digest,
                            receipt=retained_receipt is not None,
                            html=suffix == "html",
                        ),
                        encoding="utf-8",
                    )
                    (root / f"verification.{suffix}").write_text(
                        verification_guide(
                            kind,
                            version=__version__,
                            decision=view.decision,
                            html=suffix == "html",
                        ),
                        encoding="utf-8",
                    )
                (root / "README.txt").write_text(
                    "Start with brief.html (decision brief), then review.html and technical-report.html.\n"
                    "Follow verification.html or verification.md from this extracted directory.\n"
                    "Obtain approved trust inputs and your signing key separately, outside this package.\n"
                    "Read verification-snapshot.json when present; otherwise the receipt is an unverified historical attachment.\n"
                    "review-context.json is authored commentary; it is not signed evidence or an approval mechanism.\n"
                    "original-file-hashes.json records copied originals. package-manifest.json inventories delivered files.\n"
                    "These unsigned hashes detect changes relative to the inventory; they do not establish authority.\n"
                    "handoff.zip contains these files except itself. Neither inventory hashes itself.\n"
                    "Preserved supporting source documents may retain links to their original repository location.\n",
                    encoding="utf-8",
                )
                files = {
                    str(p.relative_to(root)): file_hash(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                }
                _json(
                    root / "package-manifest.json",
                    {
                        "format": "invarlock/client-review-package-v1",
                        "generator_version": __version__,
                        "generator_source_sha256": renderer_digest,
                        "renderer_sources": sources,
                        "snapshot_sha256": snapshot_digest,
                        "reproducibility": "HTML, Markdown and ZIP are deterministic for identical source, context, evidence and verification snapshot. A new receipt check includes a new local check time. PDF bytes may differ by browser and metadata; delivery records exact file hashes.",
                        "evidence_kind": kind,
                        "evidence_statement_sha256": digest,
                        "recipient_verification": snapshot["state"]
                        if snapshot
                        else "not_performed",
                        "historical_receipt": "authenticated_snapshot"
                        if snapshot
                        else "unverified_attachment"
                        if retained_receipt
                        else "not_supplied",
                        "inventory_scope": "All delivered files except this inventory and handoff.zip; unsigned identity record, not acceptance.",
                        "files": files,
                    },
                )
                write_archive(
                    root, root / "handoff.zip", [*files, "package-manifest.json"]
                )
                # Validate copied originals after rendering; hashes do not authorize a signer.
                delivered_evidence = {
                    str(path.relative_to(root / "evidence"))
                    for path in (root / "evidence").rglob("*")
                    if path.is_file()
                }
                if delivered_evidence != set(originals) or any(
                    file_hash(root / name) != expected
                    for name, expected in original_hashes.items()
                ):
                    raise ClientReviewError("copied originals changed during packaging")
                require_unchanged(source, source_state)
            stage.publish()
            stage.require_published_binding()
        return {
            "format_version": "invarlock/client-review-package-v1",
            "ok": True,
            "kind": kind,
            "output": str(output),
            "brief": str(output / "brief.html"),
            "archive": str(output / "handoff.zip"),
            "evidence_statement_sha256": digest,
            "recipient_verification": snapshot["state"]
            if snapshot
            else "not_performed",
        }
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        raise ClientReviewError(str(exc)) from exc
