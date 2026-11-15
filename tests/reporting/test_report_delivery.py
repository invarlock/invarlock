"""Deliveries identify renderer helpers and reject incomplete or changed inputs."""

import json
import runpy
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.client_review import build_client_review
from tests.reporting.test_client_handoff import REFERENCE, context, inventory


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
    key.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
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


@pytest.mark.parametrize("paper", ["A4", "Letter"])
def test_export_publishes_complete_archives_and_preserves_source(
    tmp_path, monkeypatch, capsys, paper
):
    import invarlock.report_delivery as module

    package = tmp_path / "package"
    build_client_review(
        REFERENCE / "evidence", context_path=context(tmp_path), output=package
    )
    before = inventory(package)
    browser = tmp_path / "browser-module"

    def render(command, **options):
        # Substitute only the external renderer; exercise publication and archive I/O.
        assert command[0] == "node"
        assert command[2] == str(browser)
        assert command[4] == paper
        assert options["check"] and options["timeout"] == 120
        root = Path(command[3])
        for name in ("brief", "review", "technical-report"):
            (root / f"{name}.pdf").write_bytes(f"fixture PDF: {name}, {paper}".encode())

    monkeypatch.setattr(module.subprocess, "run", render)
    output = tmp_path / "delivery"
    command = Path(__file__).resolve().parents[2] / "examples/client-review/export.py"
    runpy.run_path(str(command))
    assert not output.exists()
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(
        sys,
        "argv",
        [str(command), str(package), str(output), "--browser-module", str(browser)]
        + ([] if paper == "A4" else ["--paper", paper]),
    )
    runpy.run_path(str(command), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["output"] == str(output)
    assert inventory(package) == before
    delivered = inventory(output)
    manifest = json.loads((output / "delivery-inventory.json").read_text())
    assert manifest["snapshot_sha256"] == result["snapshot_sha256"]
    assert manifest["files"] == {
        name: digest
        for name, digest in delivered.items()
        if name not in {"delivery-inventory.json", "delivery.zip"}
    }
    with ZipFile(result["archive"]) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(delivered) - {"delivery.zip"}
        for name in archive.namelist():
            assert archive.read(name) == (output / name).read_bytes()
    with ZipFile(result["summary"]) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "package/brief.pdf",
            "summary-inventory.json",
        }
        summary = json.loads(archive.read("summary-inventory.json"))
        assert summary["files"] == {"package/brief.pdf": delivered["package/brief.pdf"]}
    assert not list(tmp_path.glob(".report-delivery-*"))


@pytest.mark.parametrize("name", ["/outside", "../outside", "nested/../../outside"])
def test_export_rejects_inventory_path_escape_before_rendering(tmp_path, name):
    from invarlock.report_delivery import export_client_review

    package = tmp_path / "package"
    build_client_review(
        REFERENCE / "evidence", context_path=context(tmp_path), output=package
    )
    path = package / "package-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe package inventory"):
        export_client_review(
            package, output=tmp_path / "delivery", browser_module=Path("/unused")
        )
    assert not (tmp_path / "delivery").exists()
    assert not list(tmp_path.glob(".report-delivery-*"))


@pytest.mark.parametrize("target", ["staged-package", "source-package"])
def test_renderer_cannot_publish_after_original_input_changes(
    tmp_path, monkeypatch, target
):
    import invarlock.report_delivery as module

    package = tmp_path / "package"
    build_client_review(
        REFERENCE / "evidence", context_path=context(tmp_path), output=package
    )

    def render(root, *_args):
        (root / "brief.pdf").write_bytes(b"fixture PDF")
        changed = root if target == "staged-package" else package
        (changed / "brief.html").write_text("modified during rendering")

    monkeypatch.setattr(module, "_print", render)
    with pytest.raises(ValueError, match="changed during"):
        module.export_client_review(
            package, output=tmp_path / "delivery", browser_module=Path("/unused")
        )
    assert not (tmp_path / "delivery").exists()
    assert not list(tmp_path.glob(".report-delivery-*"))


def test_unsupported_paper_never_creates_a_delivery(tmp_path):
    from invarlock.report_delivery import export_client_review

    with pytest.raises(ValueError, match="supported paper sizes"):
        export_client_review(
            tmp_path / "missing-package",
            output=tmp_path / "delivery",
            browser_module=Path("/unused"),
            paper="A3",
        )
    assert list(tmp_path.iterdir()) == []
