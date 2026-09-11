"""Stored captured views use one authenticated snapshot and shared publication."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
from xml.etree.ElementTree import fromstring

import pytest

from invarlock import (
    captured_contracts,
    captured_reporting,
    evidence_reporting,
    record_reporting,
)
from invarlock.captured_evaluation import evaluate_captured_request
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests.core.test_captured_contract_freeze import (
    CORPUS,
    _digest,
    _pack,
    _pem,
    _rebind,
    _value,
)
from tests.core.test_captured_evaluation import _request
from tests.core.test_captured_sdk_omissions import _inputs


@pytest.fixture
def no_replay(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("report performed replay or native fallback")
    )
    for name in (
        "invarlock.captured_verification.compare_runs",
        "invarlock.captured_evaluation.compare_runs",
        "invarlock.evaluation_comparison.comparison.score",
        "invarlock.core.scoring.score",
        "invarlock.evidence_reporting._render_native_evidence",
    ):
        monkeypatch.setattr(name, forbidden)
    return forbidden


@pytest.mark.parametrize("name", ["pass", "regression", "insufficient"])
@pytest.mark.parametrize("unsigned", [False, True])
def test_shared_render_preserves_frozen_values_labels_and_outputs(
    tmp_path, no_replay, name, unsigned
):
    pack = _pack(tmp_path, name, unsigned=unsigned)
    before = {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    manifest, payloads, signer, raw = captured_reporting._load(pack)
    view = captured_reporting._view(manifest, payloads, signer)
    stored = _value(f"comparison-{name}.json")
    assert view.technical == stored
    assert view.decision == stored["decision"]
    assert raw == {
        name: before[Path(name)] for name in captured_contracts.PAYLOADS.values()
    }
    with captured_contracts.captured_snapshot(pack) as snapshot:
        sdk_view = record_reporting._view(stored, snapshot)
    assert view.metrics == sdk_view.metrics
    by_scope = {(m["name"], m["slice"]): m for m in stored["metrics"]}
    for rendered in view.metrics:
        metric = by_scope[(rendered.name, rendered.scope)]
        assert (rendered.name, rendered.scope, rendered.decision) == (
            metric["name"],
            metric["slice"],
            metric["decision"],
        )
        assert rendered.count == str(metric["count"] - len(metric["missing_ids"]))
        if metric["baseline_mean"] is None:
            assert rendered.baseline == "Unavailable"
        else:
            assert rendered.baseline != "Unavailable"
        assert (rendered.interval is not None) == (metric["interval"] is not None)
        assert rendered.checks[0].passed is (not metric["missing_ids"])
        assert rendered.checks[1].passed is (
            metric["count"] >= payloads["policy"]["metrics"][0]["minimum_count"]
        )
        assert rendered.explanation
        if rendered.interval is not None:
            assert any(check.name == "Allowed change" for check in rendered.checks)
            assert any(check.name == "Interval width" for check in rendered.checks)
    paths = {
        name: tmp_path / f"report.{suffix}"
        for name, suffix in (("html", "html"), ("markdown", "md"), ("junit", "xml"))
    }
    result = evidence_reporting.render_evidence(
        pack,
        html_path=paths["html"],
        markdown_path=paths["markdown"],
        junit_path=paths["junit"],
        explain=True,
    )
    assert isinstance(result, evidence_reporting.EvidenceReportV2)
    assert (
        result.requested_outputs
        == result.written_outputs
        == {k: str(v) for k, v in paths.items()}
    )
    assert result.pack_manifest_digest == _digest(
        f"manifest-{name}{'-unsigned' if unsigned else ''}.json"
    )
    assert paths["markdown"].read_text() == result.text
    for text in (result.text, paths["html"].read_text()):
        assert "Not performed by report." in text
        assert "Scoring and replay were not performed by report." in text
        assert (
            "Unsigned local evidence; no signer authentication."
            if unsigned
            else "Signed manifest verified."
        ) in text
        assert ("Signed manifest verified." not in text) is unsigned
        if not unsigned:
            assert CORPUS["test_keys"]["signer"]["fingerprint"] in text
    suite = fromstring(paths["junit"].read_bytes())
    assert suite.get("tests") == str(len(stored["metrics"]))
    assert suite.get("failures") == str(
        sum(m["decision"] == "regression" for m in stored["metrics"])
    )
    assert suite.get("errors") == str(
        sum(m["decision"] == "insufficient_evidence" for m in stored["metrics"])
    )
    assert [(case.get("name"), case.get("classname")) for case in suite] == [
        (m["name"], m["slice"]) for m in stored["metrics"]
    ]
    assert before == {
        p.relative_to(pack): p.read_bytes() for p in pack.rglob("*") if p.is_file()
    }
    no_replay.assert_not_called()


def test_junit_escapes_xml_forbidden_metric_characters(tmp_path):
    _, baseline, subject, policy = _inputs(tmp_path)
    policy["metrics"][0]["name"] = "quality\ufffevisible"
    request = _request(tmp_path, baseline, subject, policy)
    result = evaluate_captured_request(request, signing_key_path=None, unsigned=True)
    destination = tmp_path / "junit.xml"
    rendered = evidence_reporting.render_evidence(
        result.evidence_path, junit_path=destination
    )
    assert rendered.written_outputs["junit"] == str(destination)
    suite = fromstring(destination.read_bytes())
    case = next(iter(suite))
    assert case.get("name") == "quality\\ufffevisible"


def test_captured_report_includes_mandatory_run_identities(tmp_path):
    pack = _pack(tmp_path)
    manifest, payloads, signer, _ = captured_reporting._load(pack)
    view = captured_reporting._view(manifest, payloads, signer)
    identities = dict(view.identity)
    for role in ("Baseline", "Subject"):
        run = payloads[role.lower()]
        assert identities[f"{role} run"] == run["run_id"]
        assert identities[f"{role} attributed artifact"] == run["artifact_digest"]
        assert identities[f"{role} evaluator"] == (
            f"{run['source']['name']} {run['source']['version']}"
        )
        assert identities[f"{role} run digest"].startswith("sha256:")


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"null",
        b"{}",
        b'{"format":"invarlock/evidence-pack-v2","kind":"runtime"}',
        b'{"format":"obsolete"}',
    ],
)
def test_manifest_detector_refuses_unsupported_discriminators(tmp_path, raw):
    (tmp_path / "manifest.json").write_bytes(raw)
    with pytest.raises(captured_reporting.CapturedReportError, match="unsupported"):
        captured_reporting.is_captured_manifest(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [b"{", b'{"format":1,"format":2}', b"x" * (captured_contracts.DETECTOR_LIMIT + 1)],
)
def test_manifest_detector_refuses_ambiguous_or_oversized_bytes(tmp_path, raw):
    (tmp_path / "manifest.json").write_bytes(raw)
    with pytest.raises(
        captured_reporting.CapturedReportError, match="identified safely"
    ):
        captured_reporting.is_captured_manifest(tmp_path)


def test_missing_manifest_and_large_native_manifest_retain_native_routing(tmp_path):
    assert captured_reporting.is_captured_manifest(tmp_path) is False
    (tmp_path / "manifest.json").write_bytes(
        b'{"format":"invarlock/evidence-pack-v1"}' + b" " * (100 * 1024)
    )
    assert captured_reporting.is_captured_manifest(tmp_path) is False


@pytest.fixture
def scoped_pack(tmp_path):
    _, baseline, subject, policy = _inputs(tmp_path)
    policy["metrics"].append({**policy["metrics"][0], "name": "second accuracy"})
    policy["slices"] = [{"name": "west", "where": {"region": "west"}}]
    for run in (baseline, subject):
        for row in run["records"]:
            row["metadata"] = {"region": "west"}
    request = _request(tmp_path, baseline, subject, policy)
    key = tmp_path / "signer.pem"
    key.write_bytes(_pem("signer"))
    pack = evaluate_captured_request(request, signing_key_path=key).evidence_path
    for path in pack.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    return pack


def test_every_configured_metric_and_overlapping_slice_is_rendered(scoped_pack):
    manifest, values, signer, _ = captured_reporting._load(scoped_pack)
    view = captured_reporting._view(manifest, values, signer)
    assert {(metric.name, metric.scope) for metric in view.metrics} == {
        (name, scope)
        for name in ("accuracy", "second accuracy")
        for scope in ("overall", "west")
    }
    assert len(view.metrics) == 4
    for metric in view.metrics:
        assert metric.count == "4"
        assert metric.checks[0].observed == "4 of 4"
        assert metric.checks[1].required == ">= 2"
        assert any(
            "overlapping slices must not be added together" in note
            for note in metric.notes
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "omits configured"),
        ("duplicate", "duplicated"),
        ("name", "not in policy"),
        ("slice", "not in policy"),
        ("kind", "contradicts policy"),
        ("direction", "contradicts policy"),
        ("unit", "contradicts policy"),
        ("scoring_assurance", "contradicts policy"),
        ("empty", "contains no metrics"),
    ],
)
def test_rebound_report_requires_complete_policy_metric_scope_inventory(
    scoped_pack, tmp_path, monkeypatch, mutation, message
):
    report_path = scoped_pack / "reports/evaluation.report.json"
    report = json.loads(report_path.read_bytes())
    assert len(report["metrics"]) == 4
    if mutation == "missing":
        report["metrics"].pop()
    elif mutation == "duplicate":
        report["metrics"][-1] = deepcopy(report["metrics"][0])
    elif mutation == "empty":
        report["metrics"] = []
    else:
        report["metrics"][0][mutation] = {
            "direction": "lower",
            "scoring_assurance": "recorded",
        }.get(mutation, "unapproved")
    report_path.chmod(0o600)
    report_path.write_bytes(canonical_json_bytes(report))
    # Re-sign all transport bindings so refusal must come from report/policy linkage.
    _rebind(scoped_pack)
    forbidden = Mock(
        side_effect=AssertionError("unbound report reached presentation or writing")
    )
    monkeypatch.setattr(evidence_reporting, "render_report_markdown", forbidden)
    monkeypatch.setattr(captured_contracts, "atomic_write", forbidden)
    with pytest.raises(evidence_reporting.EvidenceReportError, match=message) as caught:
        evidence_reporting.render_evidence(
            scoped_pack, markdown_path=tmp_path / "out.md"
        )
    assert caught.value.written_outputs == {}
    assert not (tmp_path / "out.md").exists()
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("entry", "metrics are invalid"),
        ("name", "decision is invalid"),
        ("decision", "decision is invalid"),
        ("missing_ids", "inventory is invalid"),
        ("slice", "not in policy"),
    ],
)
def test_view_defensive_checks_refuse_invalid_in_memory_values(mutation, message):
    payloads = {
        "report": _value("comparison-pass.json"),
        "policy": _value("policy.json"),
    }
    if mutation == "entry":
        payloads["report"]["metrics"][0] = None
    else:
        payloads["report"]["metrics"][0][mutation] = None
    with pytest.raises(captured_reporting.CapturedReportError, match=message):
        captured_reporting._view(_value("manifest-pass.json"), payloads, "unused")


@pytest.mark.parametrize(
    "mutation", ["signature", "ledger", "inventory", "request", "binding"]
)
def test_invalid_snapshot_never_reaches_view_or_output(tmp_path, monkeypatch, mutation):
    pack = _pack(tmp_path)
    if mutation == "signature":
        path = pack / "manifest.signature.json"
        value = json.loads(path.read_bytes())
        signature = bytearray(base64.b64decode(value["signature"]["value"]))
        signature[0] ^= 1
        value["signature"]["value"] = base64.b64encode(signature).decode()
        path.write_bytes(canonical_json_bytes(value))
    elif mutation == "ledger":
        (pack / "checksums.sha256").write_bytes(b"invalid ledger\n")
    elif mutation == "inventory":
        (pack / "receipt.json").write_bytes(b"{}\n")
    else:
        name = (
            "request.json"
            if mutation == "request"
            else "reports/evaluation.report.json"
        )
        path = pack / name
        value = json.loads(path.read_bytes())
        if mutation == "request":
            value["comparison"]["baseline"]["path"] = "/outside"
        else:
            value["bindings"]["subject"] = "sha256:" + "0" * 64
        path.write_bytes(canonical_json_bytes(value))
        _rebind(pack)
    forbidden = Mock(
        side_effect=AssertionError("invalid snapshot reached presentation")
    )
    monkeypatch.setattr(captured_reporting, "_view", forbidden)
    monkeypatch.setattr(captured_contracts, "atomic_write", forbidden)
    with pytest.raises(evidence_reporting.EvidenceReportError) as caught:
        evidence_reporting.render_evidence(pack, html_path=tmp_path / "new/report.html")
    assert caught.value.written_outputs == {}
    assert not (tmp_path / "new").exists()
    forbidden.assert_not_called()


def test_snapshot_exit_stability_check_precedes_presentation_and_output(
    tmp_path, monkeypatch
):
    pack = _pack(tmp_path)
    load = captured_reporting.load_payloads

    def mutate_after_load(snapshot):
        result = load(snapshot)
        (pack / "records/subject.json").write_bytes(b"replaced after validation")
        return result

    monkeypatch.setattr(captured_reporting, "load_payloads", mutate_after_load)
    forbidden = Mock(
        side_effect=AssertionError("unstable snapshot reached presentation")
    )
    monkeypatch.setattr(captured_reporting, "_view", forbidden)
    monkeypatch.setattr(captured_contracts, "atomic_write", forbidden)
    with pytest.raises(
        evidence_reporting.EvidenceReportError, match="changed after capture"
    ):
        evidence_reporting.render_evidence(pack, html_path=tmp_path / "out.html")
    forbidden.assert_not_called()
    assert not (tmp_path / "out.html").exists()


def test_rendering_uses_acquired_values_not_reopened_evidence(tmp_path, monkeypatch):
    pack = _pack(tmp_path)
    original = captured_reporting._view
    calls = []

    def mutate_after_capture(manifest, values, signer):
        calls.append(1)
        for path in pack.rglob("*"):
            if path.is_file():
                path.unlink()
        return original(manifest, values, signer)

    monkeypatch.setattr(captured_reporting, "_view", mutate_after_capture)
    result = evidence_reporting.render_evidence(pack, markdown_path=tmp_path / "out.md")
    assert result.pack_manifest_digest == _digest("manifest-pass.json")
    assert result.text == (tmp_path / "out.md").read_text()
    assert "Signed manifest verified." in result.text
    assert calls == [1]


def test_render_failure_writes_nothing_even_when_html_was_already_built(
    tmp_path, monkeypatch
):
    pack = _pack(tmp_path)
    html = Mock(wraps=evidence_reporting.render_report_html)
    monkeypatch.setattr(evidence_reporting, "render_report_html", html)
    monkeypatch.setattr(
        evidence_reporting,
        "tostring",
        Mock(side_effect=ValueError("cannot encode JUnit")),
    )
    writer = Mock(
        side_effect=AssertionError("incomplete rendering reached publication")
    )
    monkeypatch.setattr(captured_contracts, "atomic_write", writer)
    with pytest.raises(
        evidence_reporting.EvidenceReportError, match="cannot encode JUnit"
    ) as caught:
        evidence_reporting.render_evidence(
            pack, html_path=tmp_path / "out.html", junit_path=tmp_path / "out.xml"
        )
    assert caught.value.written_outputs == {}
    assert caught.value.failed_output is None
    html.assert_called_once()
    writer.assert_not_called()


@pytest.mark.parametrize("output", ["html", "markdown", "junit"])
@pytest.mark.parametrize("via_symlink", [False, True])
def test_shared_output_preflight_cannot_add_files_to_pack(
    tmp_path, monkeypatch, output, via_symlink
):
    pack = _pack(tmp_path)
    parent = pack
    if via_symlink:
        parent = tmp_path / "pack-link"
        parent.symlink_to(pack, target_is_directory=True)
    forbidden = Mock(side_effect=AssertionError("invalid destination reached snapshot"))
    monkeypatch.setattr(captured_reporting, "_load", forbidden)
    paths = {
        name: tmp_path / f"new/out.{name}" for name in ("html", "markdown", "junit")
    }
    paths[output] = parent / "out"
    with pytest.raises(
        evidence_reporting.EvidenceReportError,
        match="outside the immutable evidence pack",
    ) as caught:
        evidence_reporting.render_evidence(
            pack,
            html_path=paths["html"],
            markdown_path=paths["markdown"],
            junit_path=paths["junit"],
        )
    assert caught.value.written_outputs == {}
    assert not (tmp_path / "new").exists()
    assert not (pack / "out").exists()
    forbidden.assert_not_called()


@pytest.mark.parametrize("failed", ["html", "markdown", "junit"])
@pytest.mark.parametrize("failed_flush", [1, 2], ids=["file", "directory"])
def test_fsync_failure_is_atomic_per_file_and_reports_prior_outputs(
    tmp_path, monkeypatch, failed, failed_flush
):
    pack = _pack(tmp_path)
    paths = {
        name: tmp_path / f"out.{suffix}"
        for name, suffix in (("html", "html"), ("markdown", "md"), ("junit", "xml"))
    }
    fsync = captured_contracts.os.fsync
    writes = captured_contracts.atomic_write
    current = None
    flush_count = 0

    def write(path, raw):
        nonlocal current, flush_count
        current = path
        flush_count = 0
        return writes(path, raw)

    def fail_fsync(fd):
        nonlocal flush_count
        flush_count += 1
        if current == paths[failed] and flush_count == failed_flush:
            raise OSError("report disk flush failed")
        return fsync(fd)

    monkeypatch.setattr(captured_contracts, "atomic_write", write)
    monkeypatch.setattr(captured_contracts.os, "fsync", fail_fsync)
    with pytest.raises(
        evidence_reporting.EvidenceReportError, match="disk flush failed"
    ) as caught:
        evidence_reporting.render_evidence(
            pack,
            html_path=paths["html"],
            markdown_path=paths["markdown"],
            junit_path=paths["junit"],
        )
    earlier = list(paths)[: list(paths).index(failed)]
    assert caught.value.failed_output == failed
    assert caught.value.written_outputs == {name: str(paths[name]) for name in earlier}
    assert {name for name, path in paths.items() if path.exists()} == set(earlier)
    assert not list(tmp_path.glob(".captured-*"))


def test_retired_captured_output_transaction_is_not_a_compatibility_alias():
    for name in (
        "_read_json",
        "_validate_checksums",
        "_write",
        "render_captured",
        "REPORT_FORMAT",
    ):
        assert not hasattr(captured_reporting, name)
