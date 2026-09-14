from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from examples import judge_measurements_pilot_reference as ref
from invarlock.judge_measurements.acceptance import (
    replay_signed_judge_verification_receipt,
    verify_judge_evidence_with_policy,
)
from invarlock.judge_measurements.reporting import render_judge_evidence

ROOT = Path(__file__).parents[2]
REFERENCE = ROOT / "examples/judge-measurements/references/k2-32b-pilot"
ARCHIVE_SHA256 = "44966ad7be58b0f0ebe8ca3054cd92c9321ff366c4801e53032005eef60d2d58"
MANIFEST_SHA256 = "2915ccd99540f75967672bd7127e83b8903b771b8906a1b064d0580034c6097c"
LUNA_REFERENCE = ROOT / "examples/judge-measurements/references/k2-32b-luna-xhigh-pilot"
LUNA_ARCHIVE_SHA256 = "4d8f50e1cba0056d2118695a4dab73cce4a5ab10829320e2f8ea4b0c48d0e766"
LUNA_MANIFEST_SHA256 = (
    "161d7aad0bb9fe83c8e2e100452e86094ca3a780b5de882f65704ecd79ac7b80"
)
HISTORY_SHA256 = {
    "grounded_qa": "02f849fda940ec6436b0b66283f4fbe0329643673535428fa309e9cc83cbbb13",
    "extraction": "4acdeecc5b1f8c2bff6d60844bf4b05e77a9d6a56fe4bd61d30049b26edc1892",
}


@pytest.fixture(scope="module")
def files():
    return ref.read_archive(REFERENCE / "reference.zip", ARCHIVE_SHA256)


@pytest.fixture(scope="module")
def extracted(files, tmp_path_factory):
    root = tmp_path_factory.mktemp("retained-judge-pilot").resolve()
    for name, raw in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return root


@pytest.fixture(scope="module")
def luna_files():
    return ref.read_archive(LUNA_REFERENCE / "reference.zip", LUNA_ARCHIVE_SHA256)


def test_physical_archive_pins_and_public_inventory(files):
    metadata = json.loads((REFERENCE / "archive.json").read_bytes())
    assert metadata["archive"]["sha256"] == ARCHIVE_SHA256
    assert (
        metadata["archive"]["size_bytes"]
        == (REFERENCE / "reference.zip").stat().st_size
    )
    assert metadata["reference_manifest_sha256"] == MANIFEST_SHA256
    assert ref.sha(files["reference.json"]) == MANIFEST_SHA256
    for name, raw in files.items():
        assert not any(
            part in name
            for part in (
                "human_review",
                "selection.json",
                "authorization",
                "private-key",
            )
        )
        for prohibited in (
            b"PRIVATE KEY",
            b"/private/",
            b"/Users/",
            b"/tmp/",
            b"OPENAI_API_KEY",
            b"Authorization:",
            b"Bearer ",
        ):
            assert prohibited not in raw, name
    assert {name for name in files if name.startswith("attribution/")} == {
        "attribution/ATTRIBUTION.md",
        "attribution/SGD-LICENSE.txt",
        "attribution/SGD-README.md",
        "attribution/SQuAD-README.md",
        "attribution/SQuAD-SOFTWARE-LICENSE.txt",
        "attribution/source-attribution.json",
    }


def test_complete_signed_pilot_replays_with_unchanged_negative_outcomes():
    result = ref.validate_reference(REFERENCE / "reference.zip", ARCHIVE_SHA256)
    assert result["human_review"] == "pending"
    assert result["final_plans"] == "not_activated"
    assert result["new_model_calls"] == 0
    assert result["active_result_root"] == "corrected"
    for workflow in ref.WORKFLOWS:
        reviewed = result["workflows"][workflow]
        assert (
            reviewed["verified"] and reviewed["authenticated"] and reviewed["replayed"]
        )
        assert not reviewed["accepted"]
        analysis = reviewed["analysis"]
        assert analysis["decision"] == "insufficient_evidence"
        assert analysis["reasons"] == ["maximum_interval_width_exceeded"]
        assert analysis["policy"]["required"] is False
        assert analysis["policy"]["maximum_interval_width"] == "1"
        assert analysis["policy"]["minimum_units"] == 40
        assert analysis["counts"]["completed_trials"] == 240
        assert analysis["counts"]["incomplete_trials"] == 0
        interval = analysis["effect_interval"]
        assert interval["unit_count"] == 40
        assert Decimal(interval["upper"]) - Decimal(interval["lower"]) == Decimal(
            "1.007489336443204"
        )


def test_luna_archive_replays_signed_advisory_results(luna_files):
    metadata = json.loads((LUNA_REFERENCE / "archive.json").read_bytes())
    assert metadata["archive"]["sha256"] == LUNA_ARCHIVE_SHA256
    assert (
        metadata["archive"]["size_bytes"]
        == (LUNA_REFERENCE / "reference.zip").stat().st_size
    )
    assert metadata["reference_manifest_sha256"] == LUNA_MANIFEST_SHA256
    assert ref.sha(luna_files["reference.json"]) == LUNA_MANIFEST_SHA256

    result = ref.validate_reference(
        LUNA_REFERENCE / "reference.zip", LUNA_ARCHIVE_SHA256
    )
    assert result["active_result_root"] == "pilot"
    assert result["human_review"] == "pending"
    assert result["final_plans"] == "not_activated"
    assert result["new_model_calls"] == 0
    for workflow in ref.WORKFLOWS:
        reviewed = result["workflows"][workflow]
        assert reviewed["verified"]
        assert reviewed["authenticated"]
        assert reviewed["replayed"]
        assert not reviewed["accepted"]
        assert reviewed["decision"] == "insufficient_evidence"
        assert reviewed["analysis"]["reasons"] == ["maximum_interval_width_exceeded"]
        assert reviewed["analysis"]["counts"]["completed_trials"] == 240


def test_luna_archive_has_exact_public_inventory(luna_files):
    attribution = {
        "ATTRIBUTION.md",
        "SGD-LICENSE.txt",
        "SGD-README.md",
        "SQuAD-README.md",
        "SQuAD-SOFTWARE-LICENSE.txt",
        "source-attribution.json",
    }
    workflow_files = {
        "analysis-replayed.json",
        "analysis_policy.json",
        "audit/copied-sol-bound-analysis-policy.json",
        "baseline_run.json",
        "collection.json",
        "evidence/analysis_policy.json",
        "evidence/analysis_result.json",
        "evidence/baseline_run.json",
        "evidence/case_set.json",
        "evidence/envelope.json",
        "evidence/measurements.json",
        "evidence/plan.json",
        "evidence/subject_run.json",
        "measurements-collected.json",
        "plan.json",
        "receipt.json",
        "recipient.json",
        "report.html",
        "report.md",
        "report.xml",
        "subject_run.json",
    }
    expected = {"reference.json"}
    expected.update(f"attribution/{name}" for name in attribution)
    for workflow in ref.WORKFLOWS:
        expected.update(f"pilot/{workflow}/{name}" for name in workflow_files)
    expected.update(
        {
            "provenance/execution.json",
            "provenance/policy-binding-correction.json",
            "provenance/post-collection-integrity.json",
            "provenance/runtime-source-files.json",
            "summaries/comparison-summary.json",
            "summaries/usage-summary.json",
            "summaries/verified-pilot-summary.json",
        }
    )
    assert set(luna_files) == expected
    for name, raw in luna_files.items():
        assert not any(
            part in name.lower()
            for part in (
                "authorization",
                "checkpoint",
                "human_review",
                "final_validation",
                "private-key",
            )
        )
        for prohibited in (
            b"PRIVATE KEY",
            b"/private/",
            b"/Users/",
            b"/tmp/",
            b"OPENAI_API_KEY",
            b"Authorization:",
            b"Bearer ",
            b"approved_ceiling_usd",
        ):
            assert prohibited not in raw, name


@pytest.mark.parametrize("workflow", ref.WORKFLOWS)
def test_luna_policy_correction_changes_only_plan_binding(luna_files, workflow):
    active = json.loads(luna_files[f"pilot/{workflow}/analysis_policy.json"])
    copied = json.loads(
        luna_files[f"pilot/{workflow}/audit/copied-sol-bound-analysis-policy.json"]
    )
    correction = json.loads(luna_files["provenance/policy-binding-correction.json"])[
        "workflows"
    ][workflow]
    assert copied["plan_sha256"] == correction["original_plan_sha256"]
    assert active["plan_sha256"] == correction["luna_plan_sha256"]
    assert (
        ref.sha(luna_files[f"pilot/{workflow}/analysis_policy.json"])
        == correction["derived_policy_sha256"]
    )
    assert {**copied, "plan_sha256": active["plan_sha256"]} == active


@pytest.mark.parametrize("workflow", ref.WORKFLOWS)
def test_current_protocol_and_original_history_are_preserved(files, workflow):
    current = json.loads(files[f"corrected/{workflow}/measurements-collected.json"])
    previous_raw = files[f"historical/{workflow}/measurements-collected.json"]
    assert ref.sha(previous_raw) == HISTORY_SHA256[workflow]
    previous = json.loads(previous_raw)
    assert len(current["trials"]) == len(previous["trials"]) == 240
    assert current["completeness"] == {
        "expected_trials": 240,
        "recorded_trials": 240,
        "completed_trials": 240,
        "status": "complete",
    }
    completed = 203 if workflow == "grounded_qa" else 227
    assert previous["completeness"]["completed_trials"] == completed
    assert previous["completeness"]["status"] == "incomplete"
    assert all(row["status"] == "complete" for row in current["trials"])
    for row in current["trials"]:
        assert len(row["attempts"]) == 1
        attempt = row["attempts"][0]
        assert attempt["error"] is None
        assert attempt["finish_reason"] == "stop"
        assert attempt["resolved_model"] == "gpt-5.6-sol"

    def plan(era):
        return json.loads(files[f"{era}/{workflow}/plan.json"])

    old_plan, new_plan = plan("historical"), plan("corrected")
    assert new_plan["judge"]["config"]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in old_plan["judge"]["config"]
    for key in (
        "rubric",
        "scale",
        "sampling",
        "case_set_sha256",
        "baseline_run_sha256",
        "subject_run_sha256",
    ):
        assert old_plan[key] == new_plan[key]
    for role in ("baseline", "subject"):
        assert (
            files[f"historical/{workflow}/{role}_run.json"]
            == files[f"corrected/{workflow}/{role}_run.json"]
        )
    unsigned = json.loads(files[f"historical/{workflow}/evidence/envelope.json"])
    assert unsigned["signature"] is None and unsigned["signer"] is None
    status = json.loads(files["historical/retention-status.json"])
    assert status["original_producer_source_authenticated"] is False
    assert status["original_artifacts_modified"] is False
    assert (
        status["installed_runtime_source_match"]
        == "08430477203459b46f6f76d99ad3d9d08cd5dbe1"
    )


@pytest.mark.parametrize("workflow", ref.WORKFLOWS)
@pytest.mark.parametrize("changed", ["signer", "plan", "subject", "policy"])
def test_changed_recipient_expectations_are_rejected(
    extracted, tmp_path, workflow, changed
):
    current = extracted / "corrected" / workflow
    policy = json.loads((current / "recipient.json").read_bytes())
    wrong = copy.deepcopy(policy)
    if changed == "signer":
        wrong["trusted_signer"]["public_key_sha256"] = "sha256:" + "0" * 64
    elif changed == "subject":
        wrong["intended_subject"] = "sha256:" + "0" * 64
    else:
        name = "plan_sha256" if changed == "plan" else "analysis_policy_sha256"
        wrong["bindings"][name] = "0" * 64
    destination = tmp_path.resolve() / "wrong-recipient.json"
    destination.write_text(json.dumps(wrong))
    result = verify_judge_evidence_with_policy(current / "evidence", destination)
    assert not result.verified and not result.accepted


@pytest.mark.parametrize("workflow", ref.WORKFLOWS)
def test_changed_verifier_anchor_rejects_receipt(extracted, workflow):
    current = extracted / "corrected" / workflow
    result = replay_signed_judge_verification_receipt(
        current / "receipt.json",
        evidence_path=current / "evidence",
        recipient_policy_path=current / "recipient.json",
        expected_verifier_identity=f"corrected-pilot-{workflow}-verifier",
        expected_verifier_fingerprint="sha256:" + "0" * 64,
    )
    assert not result.verified and not result.accepted


@pytest.mark.parametrize("workflow", ref.WORKFLOWS)
def test_report_replays_the_retained_advisory_result(extracted, tmp_path, workflow):
    current = extracted / "corrected" / workflow
    outputs = {
        name: tmp_path.resolve() / f"report.{extension}"
        for name, extension in (("html", "html"), ("markdown", "md"), ("junit", "xml"))
    }
    report = render_judge_evidence(
        current / "evidence", **{f"{name}_path": path for name, path in outputs.items()}
    )
    assert not report.errors
    assert all(path.stat().st_size for path in outputs.values())
    assert "insufficient_evidence" in outputs["junit"].read_text()
    assert 'decision_role" value="advisory' in outputs["junit"].read_text()


def test_wrong_archive_pin_fails_before_replay():
    with pytest.raises(ValueError, match="expected SHA-256 pin"):
        ref.read_archive(REFERENCE / "reference.zip", "0" * 64)


def test_version_two_declares_active_evidence_and_expected_outcomes():
    expected = {
        workflow: {
            "accepted": False,
            "decision": "insufficient_evidence",
            "analysis_reasons": ["maximum_interval_width_exceeded"],
        }
        for workflow in ref.WORKFLOWS
    }
    root, outcomes = ref.validation_contract(
        {
            "format": "invarlock/judge-pilot-reference-v2",
            "pilot": {"path": "pilot", "workflows": expected},
        }
    )
    assert root == "pilot"
    assert outcomes == expected


def test_version_two_allows_advisory_pass_without_recipient_acceptance():
    outcomes = {
        workflow: {
            "accepted": False,
            "decision": "pass",
            "analysis_reasons": [],
        }
        for workflow in ref.WORKFLOWS
    }
    assert ref.validation_contract(
        {
            "format": "invarlock/judge-pilot-reference-v2",
            "pilot": {"path": "pilot", "workflows": outcomes},
        }
    ) == ("pilot", outcomes)


def test_version_two_rejects_unknown_decision():
    outcomes = {
        workflow: {
            "accepted": False,
            "decision": "fail",
            "analysis_reasons": [],
        }
        for workflow in ref.WORKFLOWS
    }
    with pytest.raises(ValueError, match="outcome expectation differs"):
        ref.validation_contract(
            {
                "format": "invarlock/judge-pilot-reference-v2",
                "pilot": {"path": "pilot", "workflows": outcomes},
            }
        )


@pytest.mark.parametrize("path", ["../pilot", "pilot/results", "/pilot", "pilot\\run"])
def test_version_two_rejects_unsafe_active_evidence_path(path):
    with pytest.raises(ValueError, match="path is invalid"):
        ref.validation_contract(
            {
                "format": "invarlock/judge-pilot-reference-v2",
                "pilot": {
                    "path": path,
                    "workflows": {
                        workflow: {
                            "accepted": False,
                            "decision": "insufficient_evidence",
                            "analysis_reasons": ["maximum_interval_width_exceeded"],
                        }
                        for workflow in ref.WORKFLOWS
                    },
                },
            }
        )


@pytest.mark.parametrize(
    "name", ["../escape.json", "bad\\name.json", "drive:name.json", "bad\nname.json"]
)
def test_archive_rejects_nonportable_members(tmp_path, name):
    import zipfile

    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        entry = zipfile.ZipInfo(name)
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, b"{}")
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        ref.read_archive(path, ref.sha(path.read_bytes()))


def test_archive_rejects_symlink_without_following_target(tmp_path):
    target = tmp_path / "linked.zip"
    target.symlink_to(REFERENCE / "reference.zip")
    with pytest.raises((ValueError, OSError)):
        ref.read_archive(target, ARCHIVE_SHA256)


def _write_archive(path, entries):
    import zipfile

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            entry = zipfile.ZipInfo(name)
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, raw, compress_type=zipfile.ZIP_DEFLATED)
    return ref.sha(path.read_bytes())


def _write_reference(path, files, manifest):
    entries = dict(files)
    manifest = copy.deepcopy(manifest)
    manifest["files"] = {
        name: {"sha256": ref.sha(raw), "size_bytes": len(raw)}
        for name, raw in entries.items()
        if name != "reference.json"
    }
    entries["reference.json"] = json.dumps(manifest).encode()
    return _write_archive(path, entries)


@pytest.mark.parametrize("pin", ["", "0" * 63, "A" * 64, "sha256:" + "0" * 64])
def test_archive_requires_canonical_independent_pin(tmp_path, pin):
    with pytest.raises(ValueError, match="independent archive SHA-256 pin"):
        ref.read_archive(tmp_path / "missing.zip", pin)


@pytest.mark.parametrize("limit", ["entry_count", "expanded_size"])
def test_archive_rejects_excessive_resource_usage(tmp_path, limit):
    entries = (
        {f"entry-{index}.json": b"{}" for index in range(129)}
        if limit == "entry_count"
        else {"oversized.json": b" " * (64 * 1024 * 1024 + 1)}
    )
    path = tmp_path / "excessive.zip"
    pin = _write_archive(path, entries)
    assert path.stat().st_size < 8 * 1024 * 1024
    with pytest.raises(ValueError, match="retained reference limits"):
        ref.read_archive(path, pin)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("format", "unsupported pilot reference"),
        ("missing_inventory", "file inventory differs"),
        ("extra_inventory", "file inventory differs"),
        ("digest", "file pin differs: payload.json"),
        ("size", "file pin differs: payload.json"),
    ],
)
def test_archive_rejects_manifest_integrity_mismatches(tmp_path, changed, message):
    raw = b"{}"
    manifest = {
        "format": "invarlock/judge-pilot-reference-v2",
        "files": {"payload.json": {"sha256": ref.sha(raw), "size_bytes": len(raw)}},
    }
    if changed == "format":
        manifest["format"] = "invarlock/judge-pilot-reference-v3"
    elif changed == "missing_inventory":
        manifest["files"] = {}
    elif changed == "extra_inventory":
        manifest["files"]["absent.json"] = manifest["files"]["payload.json"]
    elif changed == "digest":
        manifest["files"]["payload.json"]["sha256"] = "0" * 64
    else:
        manifest["files"]["payload.json"]["size_bytes"] += 1
    path = tmp_path / "mismatched.zip"
    pin = _write_archive(
        path, {"reference.json": json.dumps(manifest).encode(), "payload.json": raw}
    )
    with pytest.raises(ValueError, match=message):
        ref.read_archive(path, pin)


@pytest.mark.parametrize("pilot", [None, [], {}, {"path": "pilot"}])
def test_version_two_rejects_malformed_pilot_metadata(pilot):
    with pytest.raises(ValueError, match="pilot reference metadata differs"):
        ref.validation_contract(
            {"format": "invarlock/judge-pilot-reference-v2", "pilot": pilot}
        )


@pytest.mark.parametrize("workflows", [None, [], {}, {"grounded_qa": {}}])
def test_version_two_requires_exact_workflow_inventory(workflows):
    with pytest.raises(ValueError, match="pilot workflow inventory differs"):
        ref.validation_contract(
            {
                "format": "invarlock/judge-pilot-reference-v2",
                "pilot": {"path": "pilot", "workflows": workflows},
            }
        )


@pytest.mark.parametrize(
    "outcome",
    [
        None,
        {},
        {**ref.LEGACY_EXPECTATION, "accepted": 0},
        {**ref.LEGACY_EXPECTATION, "analysis_reasons": "reason"},
        {**ref.LEGACY_EXPECTATION, "analysis_reasons": [None]},
        {**ref.LEGACY_EXPECTATION, "analysis_reasons": [""]},
        {**ref.LEGACY_EXPECTATION, "unexpected": True},
    ],
)
def test_version_two_rejects_malformed_outcome(outcome):
    with pytest.raises(
        ValueError, match="grounded_qa: pilot outcome expectation differs"
    ):
        ref.validation_contract(
            {
                "format": "invarlock/judge-pilot-reference-v2",
                "pilot": {
                    "path": "pilot",
                    "workflows": {
                        "grounded_qa": outcome,
                        "extraction": ref.LEGACY_EXPECTATION,
                    },
                },
            }
        )


def test_pinned_archive_cannot_hide_invalid_receipt_signature(luna_files, tmp_path):
    entries = dict(luna_files)
    receipt_name = "pilot/grounded_qa/receipt.json"
    receipt = json.loads(entries[receipt_name])
    receipt["signature"]["value"] = "A" * 86 + "=="
    entries[receipt_name] = json.dumps(receipt).encode()
    path = tmp_path / "invalid-receipt.zip"
    pin = _write_reference(path, entries, json.loads(entries["reference.json"]))
    with pytest.raises(ValueError, match="signed receipt or evidence replay failed"):
        ref.validate_reference(path, pin)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accepted", True, "unexpected recipient outcome"),
        ("decision", "pass", "unexpected recipient outcome"),
        ("analysis_reasons", [], "unexpected analysis reason"),
    ],
)
def test_reference_expectations_must_match_signed_replay(
    luna_files, tmp_path, field, value, message
):
    manifest = json.loads(luna_files["reference.json"])
    manifest["pilot"]["workflows"]["grounded_qa"][field] = value
    path = tmp_path / "wrong-expectation.zip"
    pin = _write_reference(path, luna_files, manifest)
    with pytest.raises(ValueError, match=f"grounded_qa: {message}"):
        ref.validate_reference(path, pin)


def test_reference_cli_prints_replayed_json(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / "examples/judge_measurements_pilot_reference.py"),
            "--bundle",
            str(LUNA_REFERENCE / "reference.zip"),
            "--expected-sha256",
            LUNA_ARCHIVE_SHA256,
        ],
    )
    runpy.run_path(sys.argv[0], run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["archive_sha256"] == LUNA_ARCHIVE_SHA256
    assert result["reference_manifest_sha256"] == LUNA_MANIFEST_SHA256
    assert result["active_result_root"] == "pilot"
    assert result["new_model_calls"] == 0
    for workflow in ref.WORKFLOWS:
        assert result["workflows"][workflow]["verified"]
        assert result["workflows"][workflow]["authenticated"]
        assert result["workflows"][workflow]["replayed"]
        assert result["workflows"][workflow]["accepted"] is False
