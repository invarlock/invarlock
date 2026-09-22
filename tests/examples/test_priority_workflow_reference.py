"""Real priority captures replay with original rejection and fail on alteration."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/integrations/evaluator-live/references/priority-workflows"
SPEC = importlib.util.spec_from_file_location(
    "priority_workflow_reference", REFERENCE / "replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_execution_summary_preserves_measured_scope_and_missing_values():
    summary = json.loads((REFERENCE / "execution-summary.json").read_bytes())
    assert summary["format"] == "invarlock/priority-workflow-execution-summary-v1"
    assert "not part of this public reference" in summary["observation_provenance"]
    collection = summary["model_collection"]
    assert collection["requests"] == 576
    assert collection["host_elapsed_seconds_from_boot_to_capture_complete"] == 1008
    assert collection["process_observations"]["sdk_capture_processes"] == 16
    assert collection["process_observations"]["model_worker_processes"] == 4
    recipient = summary["independent_recipient"]
    assert set(recipient) == {
        "em_nll_journeys",
        "verified_journeys",
        "elapsed_seconds",
        "memory_bytes",
        "measurement_status",
    }
    assert (recipient["em_nll_journeys"], recipient["verified_journeys"]) == (32, 32)
    assert recipient["elapsed_seconds"] is None and recipient["memory_bytes"] is None
    assert "not retained" in recipient["measurement_status"]
    judge = summary["judge_collection"]
    assert (
        judge["admitted_calls"],
        judge["completed_ratings"],
        judge["provider_errors"],
    ) == (
        1288,
        1288,
        0,
    )
    assert (
        judge["requested_model"],
        judge["approved_resolved_model"],
        judge["reasoning_effort"],
        judge["maximum_output_tokens_per_call"],
    ) == ("openai/gpt-5.6-luna", "gpt-5.6-luna", "xhigh", 25000)
    assert judge["provider_invoice_usd"] is None and judge["elapsed_seconds"] is None
    assert summary["source_bindings"] == {
        "capture_archive_sha256": "sha256:" + REPLAY.ARCHIVES["captures.zip"],
        "judge_catalog_sha256": "sha256:acf24e0284ee078d0a4eaabd8fef659b873569b873d7f7a7dc278edccb2ad298",
    }


@pytest.fixture(scope="module")
def retained():
    return REPLAY.read_reference(REFERENCE)


@pytest.fixture(scope="module")
def materialized(tmp_path_factory, retained):
    root = tmp_path_factory.mktemp("priority-reference").resolve()
    for name, raw in retained[0].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


def test_real_source_profiles_and_original_signed_results_replay():
    result = REPLAY.replay(REFERENCE)
    assert result["ok"] and result["original_captures_checked"] == 16
    assert result["evaluators"] == list(REPLAY.EVALUATORS)
    assert result["signed_packs_replayed"] == 32
    assert result["new_model_or_judge_calls"] == 0
    assert not result["historical_policy_acceptance"]
    assert len({row["id"] for row in result["results"]}) == 32
    for row in result["results"]:
        assert row["receipt_authenticated"] and row["integrity_ok"]
        assert (row["decision"], row["policy_verdict"], row["verification_status"]) == (
            "regression",
            "fail",
            7,
        )


def test_historical_http_helper_source_matches_frozen_capture(
    retained, tmp_path, monkeypatch
):
    files, reference = retained
    path = reference["source_profiles"]["http"]["protocol"]
    protocol = json.loads(files[path])
    REPLAY.verify_historical_http_helper(protocol, "baseline")
    altered = copy.deepcopy(protocol)
    altered["http_services"]["baseline"]["helper_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="historical HTTP helper source"):
        REPLAY.verify_historical_http_helper(altered, "baseline")
    changed_source = tmp_path / "historical-helper.py.txt"
    changed_source.write_bytes(REPLAY.HISTORICAL_HTTP_HELPER.read_bytes() + b"\n")
    monkeypatch.setattr(REPLAY, "HISTORICAL_HTTP_HELPER", changed_source)
    with pytest.raises(ValueError, match="historical HTTP helper source"):
        REPLAY.verify_historical_http_helper(protocol, "baseline")


def test_catalog_pins_profiles_and_original_failures(retained):
    files, reference = retained
    catalog = json.loads((REFERENCE / "reference.json").read_bytes())
    assert len(reference["packs"]) == 32 and len(reference["captures"]) == 16
    assert reference["related_reference"]["evaluators"] == 19
    assert reference["source_profiles"]["local"]["cases_per_role"] == 64
    assert reference["source_profiles"]["http"]["cases_per_role"] == 8
    for archive in catalog["archives"]:
        raw = (REFERENCE / archive["path"]).read_bytes()
        assert len(raw) == archive["bytes"] < REPLAY.SHARED.ARCHIVE_LIMIT
        assert (
            REPLAY.SHARED.sha(raw)
            == archive["sha256"]
            == REPLAY.ARCHIVES[archive["path"]]
        )
    assert any("runtime-install/output.log" in name for name in files)
    assert any("runtime-recovery.sh" in name for name in files)
    assert any(
        name.startswith("capture/worker/local/") and name.endswith(".request.json")
        for name in files
    )
    assert any(
        name.startswith("capture/captures/http/") and "/http/" in name for name in files
    )
    for name, raw in files.items():
        assert not name.endswith((".pem", ".safetensors", ".gguf", ".pyc"))
        assert not (
            name.startswith("capture/supervision/") and name.endswith("/admission.json")
        )
        private_key_marker = b"-----BEGIN " + b"PRIVATE KEY-----"
        assert private_key_marker not in raw


def test_altered_archive_rejected_before_extracting(tmp_path):
    raw = bytearray((REFERENCE / "captures.zip").read_bytes())
    raw[-1] ^= 1
    (tmp_path / "captures.zip").write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        REPLAY.read_reference(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "format",
        "evaluator",
        "profiles",
        "count",
        "requests",
        "profile_kind",
        "missing_pack",
        "duplicate_pack",
        "missing_capture",
        "wrong_capture",
    ],
)
def test_campaign_scope_is_closed(retained, change):
    reference = copy.deepcopy(retained[1])
    if change == "format":
        reference["format"] = "different"
    elif change == "evaluator":
        reference["evaluators"] = ["other"]
    elif change == "profiles":
        reference["source_profiles"]["cloud"] = {}
    elif change == "count":
        reference["signed_packs"] = 31
    elif change == "requests":
        reference["fresh_model_case_requests"] = 1
    elif change == "profile_kind":
        reference["source_profiles"]["http"]["kind"] = "local_artifact"
    elif change == "missing_pack":
        reference["packs"].pop()
    elif change == "duplicate_pack":
        reference["packs"][-1] = reference["packs"][0]
    elif change == "missing_capture":
        reference["captures"].pop()
    else:
        reference["captures"][0]["profile"] = "wrong"
    with pytest.raises(ValueError, match="reference"):
        REPLAY.validate_reference(reference)


@pytest.mark.parametrize(
    "change", ["manifest_hash", "manifest_schema", "inventory", "member", "overlap"]
)
def test_manifest_and_member_changes_rejected(retained, monkeypatch, change):
    files = dict(retained[0])
    manifest = json.loads(files["archive-manifest.json"])
    if change == "manifest_hash":
        files["archive-manifest.json"] += b" "
    elif change == "manifest_schema":
        manifest["extra"] = True
    elif change == "inventory":
        del manifest["files"][next(iter(manifest["files"]))]
    elif change == "member":
        files["reference.json"] += b" "
    if change in {"manifest_schema", "inventory"}:
        files["archive-manifest.json"] = json.dumps(manifest).encode()
        monkeypatch.setattr(
            REPLAY, "MANIFEST_SHA256", REPLAY.SHARED.sha(files["archive-manifest.json"])
        )
    calls = 0

    def incoming(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return files if calls == 1 or change == "overlap" else {}

    monkeypatch.setattr(REPLAY.SHARED, "read_archive", incoming)
    with pytest.raises(ValueError, match="reference"):
        REPLAY.read_reference(REFERENCE)


def test_source_protocol_mutation_rejected(materialized, retained):
    reference = retained[1]
    path = materialized / reference["source_profiles"]["local"]["protocol"]
    raw = path.read_bytes()
    try:
        path.write_bytes(raw + b" ")
        with pytest.raises(ValueError, match="source protocol hash"):
            REPLAY.verify_sources(materialized, reference)
    finally:
        path.write_bytes(raw)


def test_sdk_payload_mutation_rejected(materialized, retained):
    reference = retained[1]
    row = reference["packs"][0]
    path = materialized / row["directory"] / "baseline.json"
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        value["payload"]["metadata"]["changed_after_capture"] = True
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="original SDK capture"):
            REPLAY.verify_sources(materialized, reference)
    finally:
        path.write_bytes(raw)


def test_normalized_source_must_reproduce_original_signed_digest(
    materialized, retained
):
    reference = copy.deepcopy(retained[1])
    reference["packs"][0]["anchors"]["baseline_run_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="signed run"):
        REPLAY.verify_sources(materialized, reference)


def test_cli_requires_installed_recipient_and_network_guard(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["replay.py", "--directory", str(REFERENCE)])
    monkeypatch.setattr(sys, "addaudithook", lambda value: calls.append(value))
    monkeypatch.setattr(
        REPLAY.SHARED, "require_installed", lambda: calls.append("installed")
    )
    monkeypatch.setattr(
        REPLAY, "replay", lambda directory: {"ok": directory == REFERENCE}
    )
    assert REPLAY.main() == 0
    assert calls == [REPLAY.SHARED.block_network, "installed"]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_changed_http_observed_identity_rejected(materialized, retained):
    directory = materialized / "capture/captures/http/baseline/inspect-ai/http"
    path = next(directory.glob("*.response.json"))
    raw = path.read_bytes()
    try:
        entry = json.loads(raw)
        body = json.loads(base64.b64decode(entry["response_body_base64"]))
        body["observed_model"] = "different-model"
        entry["response_body_base64"] = base64.b64encode(
            json.dumps(body).encode()
        ).decode()
        path.write_text(json.dumps(entry))
        with pytest.raises(ValueError):
            REPLAY.verify_sources(materialized, retained[1])
    finally:
        path.write_bytes(raw)


def test_changed_task_and_generation_result_cannot_detach_from_sdk(
    materialized, retained
):
    directory = materialized / "capture/captures/local/baseline/inspect-ai/tasks"
    path = next(directory.glob("*.response.json"))
    raw = path.read_bytes()
    try:
        response = json.loads(raw)
        result = response["result"]
        result["output"] = "altered task result"
        result["metadata"]["invarlock_model_execution"]["generation_result"] = [
            "altered task result"
        ]
        path.write_text(json.dumps(response))
        with pytest.raises(
            ValueError, match="normalized record differs from the original model task"
        ):
            REPLAY.verify_sources(materialized, retained[1])
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize(
    "change", [None, "duplicate", "missing", "case_metadata", "execution_metadata"]
)
def test_record_context_checks_keep_recovery_and_schedule_bindings(change):
    """Unit coverage of context guards; real task detachment is tested above."""
    from types import SimpleNamespace

    recipient = REPLAY.module(
        "priority_context_checks", REPLAY.HERE.parents[1] / "recipient.py"
    )
    case = {"id": "fixture", "metadata": {"dataset": "unit fixture"}}
    execution = {"kind": "unit fixture"}
    binding = {"kind": "capture fixture"}
    serialization = {"kind": "serialization fixture"}
    transport = {"kind": "transport fixture"}
    result = {
        "output": "observed",
        "metadata": {
            "invarlock_model_execution": execution,
            "invarlock_transport_replay": transport,
        },
    }
    record = {
        "id": "fixture",
        "context": {
            "dataset": "unit fixture",
            "invarlock_model_execution": execution,
            "invarlock_task_outcome": {"output": "observed", "error": None},
            "invarlock_capture_binding": binding,
            "invarlock_serialization_binding": serialization,
            "invarlock_transport_replay": transport,
        },
    }
    calls = []

    def checked(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "metadata": {
                "invarlock_capture_binding": binding,
                "invarlock_serialization_binding": serialization,
            }
        }

    delegated = SimpleNamespace(
        contains=recipient.contains,
        bindings=SimpleNamespace(check_record=checked),
    )
    records = [record]
    if change == "duplicate":
        records.append(record)
    elif change == "missing":
        records.clear()
    elif change == "case_metadata":
        del record["context"]["dataset"]
    elif change == "execution_metadata":
        del record["context"]["invarlock_transport_replay"]
    protocol = {"cases": [case], "versions": {"inspect-ai": "fixture-version"}}
    arguments = (
        delegated,
        {"records": records},
        {"fixture": result},
        protocol,
        "inspect-ai",
        {"kind": "service fixture"},
    )
    if change:
        with pytest.raises(ValueError, match="schedule|metadata"):
            REPLAY.check_records(*arguments)
    else:
        REPLAY.check_records(*arguments)
        assert len(calls) == 1
        assert calls[0][0] == (record, result, case, "inspect-ai", "fixture-version")
        assert calls[0][1] == {
            "cases": [case],
            "service_identity": {"kind": "service fixture"},
        }
