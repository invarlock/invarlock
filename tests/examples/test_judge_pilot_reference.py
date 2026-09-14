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
