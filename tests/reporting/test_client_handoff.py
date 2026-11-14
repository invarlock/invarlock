"""Client packages preserve originals and keep recipient authority external."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.client_review import ClientReviewError, build_client_review

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/captured-results/references/mistral-7b-likelihood"


def context(tmp_path, **changes):
    value = {
        "format": "invarlock/client-review-context-v1",
        "title": "Editorial checkpoint replacement",
        "review_date": "2026-10-27",
        "label": "Client-authored review",
        "byline": "Casey, evaluation lead",
        "context": [
            ["Intended use", "Plain-text completion"],
            ["Decision owner", "Sam"],
        ],
        "recommendation": "Retain the baseline pending review.",
        "next_action": "Sam will review the failed requirement before release.",
        "basis": [
            ["Scope", "This fixed subset does not establish production quality."]
        ],
    }
    value.update(changes)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(value))
    return path


def inventory(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_package_retains_evidence_receipt_context_and_separate_pending_decision(
    tmp_path,
):
    source = REFERENCE / "evidence"
    before = inventory(source)
    ctx = context(tmp_path)
    output = tmp_path / "handoff"
    result = build_client_review(
        source,
        context_path=ctx,
        output=output,
        retained_receipt=REFERENCE / "verification.receipt.json",
    )
    assert result["kind"] == "captured"
    assert inventory(source) == before == inventory(output / "evidence")
    assert (output / "review-context.json").read_bytes() == ctx.read_bytes()
    assert (output / "verification.receipt.json").read_bytes() == (
        REFERENCE / "verification.receipt.json"
    ).read_bytes()
    brief = (output / "brief.html").read_text()
    assert "Pending owner decision" in brief
    assert "Policy not met" in brief
    assert 'href="verification.html"' in brief
    assert 'href="verification.receipt.json"' in brief
    assert "Not authenticated by packaging" in brief
    assert "Not performed for this recipient" in brief
    assert "not authenticated it against a recipient-owned key" in brief
    assert "An application-specific interpretation was not supplied" in brief
    assert 'href="brief.html"' in (output / "review.html").read_text()
    manifest = json.loads((output / "package-manifest.json").read_text())
    assert "review-context.json" in manifest["files"]
    assert "brief.html" in manifest["files"]
    with ZipFile(output / "handoff.zip") as archive:
        assert "handoff.zip" not in archive.namelist()
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    guide = (output / "verification.md").read_text()
    assert "invarlock verify ./evidence" in guide
    assert '--trust-profile "$TRUST_PROFILE"' in guide
    assert "outside this package" in guide
    assert "Exit 7" in guide and "rejection" in guide
    assert "harness_model_handoff.py" not in guide


@pytest.mark.parametrize("family", ["native", "judge", "combined"])
def test_other_families_use_existing_report_validators(tmp_path, family):
    if family == "native":
        source = ROOT / "examples/acceptance-handoff/golden/evidence"
    elif family == "judge":
        from tests.judge_measurements.test_evidence_acceptance import _publish

        source = _publish(tmp_path)[0].path
    else:
        from tests.evidence_sets.test_verification import fixture

        source = fixture(tmp_path)[0]
    before = inventory(source)
    output = tmp_path / "handoff"
    result = build_client_review(source, context_path=context(tmp_path), output=output)
    assert (
        result["kind"]
        == {"native": "runtime", "judge": "judge", "combined": "evidence_set"}[family]
    )
    assert inventory(source) == before == inventory(output / "evidence")
    digest = result["evidence_statement_sha256"]
    reference = (
        {"index_sha256": digest}
        if family == "combined"
        else {"statement": {"result": {"envelope_sha256": digest}}}
        if family == "judge"
        else {"statement": {"pack_manifest_digest": digest}}
    )
    receipt = tmp_path / "unauthenticated-reference.json"
    receipt.write_text(json.dumps(reference))
    with_receipt = tmp_path / "with-receipt"
    build_client_review(
        source,
        context_path=context(tmp_path),
        output=with_receipt,
        retained_receipt=receipt,
    )
    assert (
        with_receipt / "verification.receipt.json"
    ).read_bytes() == receipt.read_bytes()
    assert "Not authenticated by packaging" in (with_receipt / "brief.html").read_text()
    guide = (output / "verification.md").read_text()
    assert ("--verifier-signing-key" in guide) == (family == "judge")
    if family == "combined":
        assert "not a new signed attestation" in guide


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": True},
        {"title": ""},
        {"decision": None},
        {"review_date": "yesterday"},
        {"context": [["bad"]]},
        {"byline": "x" * 5000},
    ],
)
def test_invalid_context_fails_without_output(tmp_path, change):
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path, **change),
            output=tmp_path / "handoff",
        )
    assert not (tmp_path / "handoff").exists()


def test_wrong_receipt_is_not_presented_as_related(tmp_path):
    receipt = json.loads((REFERENCE / "verification.receipt.json").read_text())
    receipt["statement"]["pack_manifest_digest"] = "sha256:" + "0" * 64
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ClientReviewError, match="receipt.*bind"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "handoff",
            retained_receipt=path,
        )
    assert not (tmp_path / "handoff").exists()


def test_rejects_existing_or_inside_evidence_destination(tmp_path):
    ctx = context(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("untouched")
    for output in (existing, REFERENCE / "evidence" / "review"):
        with pytest.raises((ClientReviewError, FileExistsError)):
            build_client_review(REFERENCE / "evidence", context_path=ctx, output=output)
    assert (existing / "keep").read_text() == "untouched"


def test_rejects_symlink_evidence_and_escapes_authored_text(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(REFERENCE / "evidence", target_is_directory=True)
    ctx = context(tmp_path, title="<script>alert(1)</script>")
    with pytest.raises(ClientReviewError):
        build_client_review(alias, context_path=ctx, output=tmp_path / "unsafe")
    build_client_review(
        REFERENCE / "evidence", context_path=ctx, output=tmp_path / "safe"
    )
    for name in ("brief.html", "review.html"):
        text = (tmp_path / "safe" / name).read_text()
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text


def test_cli_requires_package_and_context_together(tmp_path):
    runner = CliRunner()
    for args in (
        ["--package", str(tmp_path / "out")],
        ["--review-context", str(context(tmp_path))],
    ):
        result = runner.invoke(
            app, ["report", str(REFERENCE / "evidence"), *args, "--json"]
        )
        assert result.exit_code == 2
        assert "together" in result.stdout
    result = runner.invoke(
        app,
        [
            "report",
            str(REFERENCE / "evidence"),
            "--review-context",
            str(context(tmp_path)),
            "--package",
            str(tmp_path / "out"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (
        json.loads(result.stdout)["format_version"]
        == "invarlock/client-review-package-v1"
    )


def test_invalid_signed_pack_and_copy_changes_never_publish(tmp_path, monkeypatch):
    import shutil

    import invarlock.client_review as module

    source = tmp_path / "source"
    shutil.copytree(REFERENCE / "evidence", source)
    original_view = module._view

    def changed_view(evidence):
        view = original_view(evidence)
        (source / "checksums.sha256").write_text("changed concurrently")
        return view

    monkeypatch.setattr(module, "_view", changed_view)
    with pytest.raises(ClientReviewError, match="changed"):
        build_client_review(
            source, context_path=context(tmp_path), output=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".client-review-*"))
    monkeypatch.setattr(module, "_view", original_view)
    with pytest.raises(ClientReviewError):
        build_client_review(
            source, context_path=context(tmp_path), output=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_does_not_promote_a_forged_receipt_to_authenticated_status(tmp_path):
    receipt = json.loads((REFERENCE / "verification.receipt.json").read_text())
    receipt["signature"]["value"] = "not a real signature"
    receipt["statement"]["verdict"]["ok"] = True
    path = tmp_path / "unverified.json"
    path.write_text(json.dumps(receipt))
    build_client_review(
        REFERENCE / "evidence",
        context_path=context(tmp_path),
        output=tmp_path / "out",
        retained_receipt=path,
    )
    brief = (tmp_path / "out/brief.html").read_text()
    assert "Not authenticated by packaging" in brief
    assert "Policy not met" in brief
    assert "verified rejection" not in brief
    assert (
        tmp_path / "out/verification.receipt.json"
    ).read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    "name",
    ["../escape", "evidence", "HANDOFF.ZIP", "review.html", "bad\\path", "bad\nname"],
)
def test_supporting_files_cannot_replace_deliverables(tmp_path, name):
    with pytest.raises(ClientReviewError, match="supporting file"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "out",
            supporting_files={name: REFERENCE / "README.md"},
        )
    assert not (tmp_path / "out").exists()


def test_duplicate_json_fields_and_symlink_members_are_rejected(tmp_path):
    import shutil

    path = context(tmp_path)
    raw = path.read_text().replace('"format":', '"format": "duplicate", "format":', 1)
    path.write_text(raw)
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence", context_path=path, output=tmp_path / "out"
        )
    source = tmp_path / "source"
    shutil.copytree(REFERENCE / "evidence", source)
    (source / "outside").symlink_to(REFERENCE / "README.md")
    with pytest.raises(ClientReviewError, match="regular"):
        build_client_review(
            source, context_path=context(tmp_path), output=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_commentary_is_bounded_and_brief_preserves_full_text(tmp_path):
    value = "Detailed review sentence. " * 80
    ctx = context(tmp_path, recommendation=value)
    build_client_review(
        REFERENCE / "evidence", context_path=ctx, output=tmp_path / "out"
    )
    assert value in (tmp_path / "out/brief.html").read_text()
    assert value in (tmp_path / "out/review.html").read_text()
    ctx.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ClientReviewError, match="limit"):
        build_client_review(
            REFERENCE / "evidence", context_path=ctx, output=tmp_path / "large"
        )


def test_plain_language_summary_is_attributed_escaped_and_separate_from_evidence(
    tmp_path,
):
    from html import unescape

    summary = [
        ["What was tested", "A fixed test set <not production traffic>."],
        ["What happened", "The replacement did not meet the recorded requirement."],
        ["What this means", "Keep the current model pending review."],
        ["What remains unknown", "Application suitability has not been established."],
    ]
    ctx = context(tmp_path, summary=summary)
    output = tmp_path / "handoff"
    before = inventory(REFERENCE / "evidence")
    build_client_review(REFERENCE / "evidence", context_path=ctx, output=output)
    assert inventory(output / "evidence") == before == inventory(REFERENCE / "evidence")
    assert (output / "review-context.json").read_bytes() == ctx.read_bytes()
    for suffix in ("html", "md"):
        for name in ("brief", "review"):
            text = (output / f"{name}.{suffix}").read_text()
            assert "Summary by the review author" in text
            assert "Pending owner decision" in text
            assert "Policy not met" in text
            assert "<not production traffic>" not in text
            assert all(value in unescape(text) for _, value in summary)
        brief = (output / f"brief.{suffix}").read_text()
        assert brief.index("Next action") < brief.index("In plain language")
        assert brief.index("Results at a glance") < brief.index("In plain language")
        technical = (output / f"technical-report.{suffix}").read_text()
        assert summary[1][1] not in technical
        assert "1.091" in technical and "1.05" in technical
        assert "paired" in technical


@pytest.mark.parametrize(
    "summary",
    [
        "not rows",
        [["Label", "text"]] * 5,
        [["Label", "x" * 601]],
        [["Label", ""]],
        [["Label", "value", "extra"]],
        [["Label", "bad\x00text"]],
    ],
)
def test_plain_language_summary_rejects_invalid_or_oversized_input(tmp_path, summary):
    with pytest.raises(ClientReviewError, match="summary"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path, summary=summary),
            output=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("family", ["required", "advisory", "combined"])
def test_authored_summary_cannot_hide_judge_role_or_combined_requirements(
    tmp_path, family
):
    if family == "combined":
        from tests.evidence_sets.test_verification import fixture

        source = fixture(tmp_path)[0]
        expected = (
            "Direct-score comparison:",
            "Judge comparison:",
            "Both comparisons must meet",
        )
    else:
        from tests.judge_measurements.test_evidence_acceptance import _publish

        source = _publish(tmp_path, role=family)[0].path
        expected = (
            ("Advisory judge result", "cannot satisfy a required release check")
            if family == "advisory"
            else (
                "Required judge comparison",
                "covers only the recorded judge requirements",
            )
        )
    output = tmp_path / "handoff"
    build_client_review(
        source,
        output=output,
        context_path=context(
            tmp_path,
            summary=[["What this means", "Review this result before deciding."]],
        ),
    )
    for suffix in ("html", "md"):
        brief = (output / f"brief.{suffix}").read_text()
        assert "Pending owner decision" in brief
        for text in expected:
            assert text in brief
            assert brief.index(text) < brief.index("In plain language")


def test_cli_rejects_mixed_output_modes_before_writing(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "report",
            str(REFERENCE / "evidence"),
            "--package",
            str(tmp_path / "out"),
            "--review-context",
            str(context(tmp_path)),
            "--html",
            str(tmp_path / "old.html"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "cannot be combined" in result.stdout
    assert not (tmp_path / "out").exists() and not (tmp_path / "old.html").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"format": "unknown"},
        {"context": []},
        {"context": [["label", "bad\x01text"]]},
        {"basis": "not rows"},
        {"context": [["label"]]},
    ],
)
def test_invalid_context_never_publishes(tmp_path, changes):
    with pytest.raises(ClientReviewError):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path, **changes),
            output=tmp_path / "handoff",
        )
    assert not (tmp_path / "handoff").exists()


@pytest.mark.parametrize("value", [[], {"statement": []}])
def test_malformed_historical_receipt_is_not_attached(tmp_path, value):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ClientReviewError, match="must be.*object"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "handoff",
            retained_receipt=receipt,
        )
    assert not (tmp_path / "handoff").exists()


def test_supporting_file_count_is_bounded(tmp_path):
    with pytest.raises(ClientReviewError, match="32 supporting"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "handoff",
            supporting_files={str(n): tmp_path / str(n) for n in range(33)},
        )


@pytest.mark.parametrize("member", ["checksums.sha256", "unexpected.json"])
def test_changed_copy_cannot_publish(tmp_path, monkeypatch, member):
    import invarlock.client_review_presentation as presentation

    real = presentation.brief

    def corrupt(*args, **kwargs):
        for stage in tmp_path.glob(".client-review-*"):
            (stage / "evidence" / member).write_text("changed")
        return real(*args, **kwargs)

    monkeypatch.setattr(presentation, "brief", corrupt)
    with pytest.raises(ClientReviewError, match="copied originals changed"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "handoff",
        )
    assert not (tmp_path / "handoff").exists()
    assert not list(tmp_path.glob(".client-review-*"))


def test_markdown_commentary_cannot_inject_html_or_links(tmp_path):
    text = (
        "[approve](https://invalid.example)\n# false heading <script>x</script> `code`"
    )
    build_client_review(
        REFERENCE / "evidence",
        context_path=context(tmp_path, recommendation=text),
        output=tmp_path / "handoff",
    )
    brief = (tmp_path / "handoff/brief.md").read_text()
    assert "<script>" not in brief and "[approve]" not in brief
    assert "\n# false heading" not in brief and "`code`" not in brief
    assert "[Verify this package](verification.html)" in brief
