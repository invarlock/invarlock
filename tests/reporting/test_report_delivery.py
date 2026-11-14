"""Deliveries identify renderer helpers and reject incomplete or changed inputs."""

import json
from pathlib import Path

import pytest

from invarlock.client_review import build_client_review
from tests.reporting.test_client_handoff import REFERENCE, context


def test_renderer_identity_covers_helpers_and_package_is_repeatable(tmp_path):
    from invarlock.report_provenance import renderer_identity

    source = tmp_path / "renderer"
    source.mkdir()
    (source / "builder.py").write_text("version = 1")
    helper = source / "helper.py"
    helper.write_text('color = "blue"')
    first = renderer_identity(source)
    helper.write_text('color = "green"')
    assert renderer_identity(source) != first
    ctx = context(tmp_path)
    for name in ["one", "two"]:
        build_client_review(
            REFERENCE / "evidence", context_path=ctx, output=tmp_path / name
        )
    assert (tmp_path / "one/handoff.zip").read_bytes() == (
        tmp_path / "two/handoff.zip"
    ).read_bytes()
    manifest = json.loads((tmp_path / "one/package-manifest.json").read_text())
    assert "client_review_presentation.py" in manifest["renderer_sources"]
    assert "report_reading.py" in manifest["renderer_sources"]
    assert manifest["snapshot_sha256"] in (tmp_path / "one/brief.html").read_text()


def test_export_failure_never_publishes_complete_delivery(tmp_path, monkeypatch):
    import invarlock.report_delivery as module
    from invarlock.report_delivery import export_client_review

    package = tmp_path / "package"
    build_client_review(
        REFERENCE / "evidence", context_path=context(tmp_path), output=package
    )

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted export")

    monkeypatch.setattr(module, "_print", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        export_client_review(
            package, output=tmp_path / "delivery", browser_module=Path("/unused")
        )
    assert not (tmp_path / "delivery").exists()
    assert not list(tmp_path.glob(".report-delivery-*"))


def test_private_key_attachment_rejected_and_sensitive_output_not_in_brief(tmp_path):
    from invarlock.client_review import ClientReviewError

    key = tmp_path / "secret.pem"
    key.write_text(
        "-----BEGIN PRIVATE KEY-----\nSYNTHETIC_SECRET\n-----END PRIVATE KEY-----"
    )
    with pytest.raises(ClientReviewError, match="private keys"):
        build_client_review(
            REFERENCE / "evidence",
            context_path=context(tmp_path),
            output=tmp_path / "unsafe",
            supporting_files={"secret.pem": key},
        )
    from invarlock.evaluation_records.templates import example_project
    from tests._evaluation_support import build_pack, write_snapshot

    baseline, candidate, policy = example_project("classification")
    candidate["records"][0]["output"] = "SENSITIVE_OUTPUT_MARKER"
    source = tmp_path / "evidence"
    write_snapshot(source, build_pack(baseline, candidate, policy))
    out = tmp_path / "safe"
    build_client_review(source, context_path=context(tmp_path), output=out)
    assert "SENSITIVE_OUTPUT_MARKER" not in (out / "brief.html").read_text()
    assert (
        "SENSITIVE_OUTPUT_MARKER" in (out / "evidence/records/subject.json").read_text()
    )


def test_stale_delivery_inventory_is_rejected(tmp_path):
    from invarlock.report_delivery import export_client_review

    package = tmp_path / "package"
    build_client_review(
        REFERENCE / "evidence", context_path=context(tmp_path), output=package
    )
    (package / "brief.html").write_text("changed")
    with pytest.raises(ValueError, match="stale or incomplete"):
        export_client_review(
            package, output=tmp_path / "delivery", browser_module=Path("/unused")
        )
    assert not (tmp_path / "delivery").exists()
