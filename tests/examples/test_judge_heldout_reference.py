"""Replay the complete held-out reference without calling a judge."""

import json
from pathlib import Path

import pytest

from examples import judge_measurements_pilot_reference as ref
from examples.judge_measurements_review import agreement, reconcile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/judge-measurements/references/k2-32b-luna-xhigh-heldout"
PIN = "19574e2f68f00e06f13432369d99818e625b58d10bf53156bebf86b538409242"
EXPANDED = 216183325


@pytest.fixture(scope="module")
def files():
    return ref.read_archive(
        REFERENCE / "reference.zip", PIN, max_expanded_bytes=EXPANDED
    )


def test_heldout_transport_and_complete_inventory(files):
    meta = json.loads((REFERENCE / "archive.json").read_bytes())
    assert meta["archive"]["sha256"] == PIN
    assert meta["archive"]["size_bytes"] == (REFERENCE / "reference.zip").stat().st_size
    assert meta["reference_manifest_sha256"] == ref.sha(files["reference.json"])
    assert len(files) == meta["member_count"] == 47
    assert sum(map(len, files.values())) == meta["expanded_size_bytes"] == EXPANDED
    summary = json.loads(files["summaries/completed-study.json"])
    assert summary["all_calls_complete"] and summary["distinct_response_ids"] == 10260
    assert summary["invoice_checked"] is False
    assert summary["conservative_ledger_cost_usd"] < 4
    for wf, count in [("grounded_qa", 2532), ("extraction", 7728)]:
        measurements = json.loads(files[f"heldout/{wf}/evidence/measurements.json"])
        assert measurements["completeness"]["completed_trials"] == count
        assert measurements["completeness"]["status"] == "complete"
        assert all(t["status"] == "complete" for t in measurements["trials"])


def test_heldout_review_is_explicit_and_physically_bound(files):
    manifest = json.loads(files["review/manifest.json"])
    assert manifest["reviewer_type"] == "ai"
    assert manifest["human_review_performed"] is False
    assert manifest["ratings_frozen_before_luna_comparison"] is True
    for name, pin in manifest["files"].items():
        assert ref.sha(files[f"review/{name}"]) == pin
    for wf, matches in [("grounded_qa", 334), ("extraction", 463)]:
        comparison = json.loads(files[f"heldout/{wf}/ai-review-comparison.json"])
        assert comparison["exact_matches"] == matches
        labels = reconcile(
            json.loads(files[f"review/{wf}-completed.json"]),
            json.loads(files[f"review/{wf}-selection.json"]),
        )
        recomputed = agreement(
            labels, json.loads(files[f"heldout/{wf}/evidence/measurements.json"])
        )
        assert recomputed["exact_matches"] == matches
        assert recomputed["compared_trials"] == comparison["compared_judge_trials"]
        assert comparison["reviewed_answers"] == 160
        assert sum(c["count"] for c in comparison["confusion"]) == 480
    split = json.loads(files["provenance/split-validation.json"])
    assert all(
        x["pilot_final_case_overlap"] == x["pilot_final_source_cluster_overlap"] == 0
        for x in split.values()
    )


def test_heldout_signed_receipts_and_policy_outcomes_replay():
    result = ref.validate_reference(
        REFERENCE / "reference.zip", PIN, max_expanded_bytes=EXPANDED
    )
    assert result["new_model_calls"] == 0
    assert result["format"] == "invarlock/judge-reference-replay-v2"
    assert "human_review" not in result
    assert result["reference_review"] == {
        "archived_status": "heldout_ai_review_only",
        "label_source": "ai",
    }
    for outcome in result["workflows"].values():
        assert outcome["authenticated"] and outcome["replayed"] and outcome["verified"]
        assert outcome["accepted"] and outcome["decision"] == "pass"


def test_heldout_archive_contains_only_publishable_artifacts(files):
    for name, raw in files.items():
        assert not any(
            x in name for x in ("private-key", "checkpoint", "authorization")
        )
        assert all(
            x not in raw
            for x in (
                b"PRIVATE KEY",
                b"/private/tmp/",
                b"/Users/ospc/",
                b"OPENAI_API_KEY",
                b"Bearer ",
                b"Authorization:",
            )
        )
