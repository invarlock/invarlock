"""Exercise the candidate campaign without weights, network, or a GPU."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from examples.qualification import k2_campaign as campaign


def _tensor_file(path: Path, values: bytes = b"\x80?\x00@"):
    header = json.dumps(
        {"layer.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + values)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob": None,
    }


def test_all_five_plans_are_deterministic_and_honestly_unqualified():
    plans = campaign.draft_plans()
    assert len(plans) == 5
    assert plans == campaign.draft_plans()
    for plan in plans:
        assert plan["status"] == "candidate_not_qualified"
        assert plan["route"] == "external_sglang_pipeline_capture"
        assert plan["runtime"]["dtype"] == "bfloat16"
        assert plan["runtime"]["trust_remote_code"] is False
        assert plan["runtime"]["image_digest"] is None
        assert (
            plan["model"]["baseline"]["revision"]
            != plan["model"]["candidate"]["revision"]
        )
        assert len(plan["cases"]) == 576
        assert set(plan["policies"]) == {"classification", "extraction", "numeric"}


def test_materialization_recomputes_bytes_and_detects_tampering(tmp_path):
    item = _tensor_file(tmp_path / "model.safetensors")
    identity = campaign.measure_snapshot(tmp_path, [item])
    assert identity["tensors"]["layer.weight"]["dtype"] == "BF16"
    (tmp_path / item["path"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="identity"):
        campaign.measure_snapshot(tmp_path, [item])


def test_same_tensor_repackaging_is_not_a_meaningful_checkpoint_change(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    a = campaign.measure_snapshot(left, [_tensor_file(left / "a.safetensors")])
    b = campaign.measure_snapshot(right, [_tensor_file(right / "b.safetensors")])
    assert a["artifact_digest"] != b["artifact_digest"]
    with pytest.raises(ValueError, match="tensor content"):
        campaign.require_changed_tensors(a, b)
    item = _tensor_file(right / "b.safetensors", b"\x00?\x80?")
    campaign.require_changed_tensors(a, campaign.measure_snapshot(right, [item]))


def test_materialization_refuses_symlinks_and_unexpected_files(tmp_path):
    item = _tensor_file(tmp_path / "model.safetensors")
    (tmp_path / "extra.py").write_text("raise RuntimeError('unexpected code')")
    with pytest.raises(ValueError, match="inventory"):
        campaign.measure_snapshot(tmp_path, [item])
    (tmp_path / "extra.py").unlink()
    target = tmp_path / "outside"
    (tmp_path / item["path"]).rename(target)
    (tmp_path / item["path"]).symlink_to(target)
    with pytest.raises(ValueError, match="regular|symlink"):
        campaign.measure_snapshot(tmp_path, [item])


def test_unresolved_runtime_cannot_be_used_for_capture():
    plan = campaign.draft_plans()[0]
    with pytest.raises(ValueError, match="runtime"):
        campaign.require_ready(plan)


def _ready_plan():
    plan = campaign.draft_plans()[0]
    plan["runtime"]["image_digest"] = "sha256:" + "1" * 64
    plan["runtime"]["build_manifest_digest"] = "sha256:" + "2" * 64
    plan["runtime"]["security_review_digest"] = "sha256:" + "3" * 64
    plan["runtime"]["dependency_inventory_digest"] = "sha256:" + "4" * 64
    plan["runtime"]["source_bundle_digest"] = "sha256:" + "5" * 64
    plan["budget"] = {"maximum_wall_seconds": 3600, "maximum_output_tokens": 400000}
    for index, role in enumerate(("baseline", "candidate")):
        plan["model"][role]["materialized"] = {
            "artifact_digest": "sha256:" + str(index + 6) * 64,
            "tensors": {
                "weight": {"dtype": "BF16", "shape": [1], "sha256": str(index + 7) * 64}
            },
        }
    return plan


def _capture(plan, role, *, wrong=False):
    rows = []
    for case in plan["cases"]:
        answer = case["expected"]
        if not isinstance(answer, str):
            answer = json.dumps(answer)
        rows.append(
            {
                "id": case["id"],
                "request": campaign.request_for(case),
                "response": {
                    "model": "k2-campaign",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "wrong" if wrong else answer,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "total_tokens": 60,
                    },
                },
                "latency_ms": 100,
                "error": None,
            }
        )
    captured = {
        "format": campaign.CAPTURE_FORMAT,
        "plan_digest": campaign.digest(plan),
        "role": role,
        "phase": "decision",
        "runtime": campaign.expected_server_settings(plan, role),
        "native_server_info": campaign.expected_server_settings(plan, role),
        "native_model_info": {
            "model_path": f"/models/{role}",
            "tokenizer_path": f"/models/{role}",
            "model_type": "k2_horizon",
            "architectures": ["K2HorizonForCausalLM"],
            "served_model_name": "k2-campaign",
            "weight_version": plan["model"][role]["materialized"]["artifact_digest"],
        },
        "rows": rows,
    }
    captured["final_native_server_info"] = copy.deepcopy(captured["native_server_info"])
    captured["final_native_model_info"] = copy.deepcopy(captured["native_model_info"])
    return captured


def test_real_projection_replays_scores_and_rejects_native_capture_changes():
    plan = _ready_plan()
    left, right = _capture(plan, "baseline"), _capture(plan, "candidate")
    key = Ed25519PrivateKey.generate()
    evidence = campaign.publish(plan, left, right, key)
    result = campaign.verify(
        plan,
        left,
        right,
        evidence,
        key.public_key(),
        expected_plan=campaign.digest(plan),
        expected_baseline_capture=campaign.digest(left),
        expected_candidate_capture=campaign.digest(right),
    )
    assert set(result.values()) == {"pass"}
    changed = copy.deepcopy(right)
    changed["rows"][0]["response"]["choices"][0]["message"]["content"] = "wrong"
    with pytest.raises(ValueError, match="capture"):
        campaign.verify(
            plan,
            left,
            changed,
            evidence,
            key.public_key(),
            expected_plan=campaign.digest(plan),
            expected_baseline_capture=campaign.digest(left),
            expected_candidate_capture=campaign.digest(right),
        )


def test_quality_rejection_and_truncated_generation_are_preserved():
    plan = _ready_plan()
    left, right = _capture(plan, "baseline"), _capture(plan, "candidate", wrong=True)
    evidence = campaign.publish(plan, left, right, Ed25519PrivateKey.generate())
    assert all(
        value["comparison"]["decision"] == "regression" for value in evidence.values()
    )
    right = _capture(plan, "candidate")
    right["rows"][0]["response"]["choices"][0]["finish_reason"] = "length"
    evidence = campaign.publish(plan, left, right, Ed25519PrivateKey.generate())
    assert (
        evidence["classification"]["comparison"]["decision"] == "insufficient_evidence"
    )


def test_changed_request_or_missing_case_is_not_silently_paired():
    plan = _ready_plan()
    captured = _capture(plan, "baseline")
    captured["rows"][0]["request"]["temperature"] = 1
    with pytest.raises(ValueError, match="request"):
        campaign.project_capture(plan, captured)
    captured = _capture(plan, "baseline")
    captured["rows"].pop()
    with pytest.raises(ValueError, match="schedule"):
        campaign.project_capture(plan, captured)


def test_actual_flat_native_server_info_and_loaded_weight_identity_are_checked():
    plan = _ready_plan()
    captured = _capture(plan, "baseline")
    assert campaign.project_capture(plan, captured)
    captured["native_model_info"]["weight_version"] = "wrong-checkpoint"
    with pytest.raises(ValueError, match="model"):
        campaign.project_capture(plan, captured)


def test_capture_requires_post_run_native_observations():
    plan = _ready_plan()
    captured = _capture(plan, "baseline")
    captured.pop("final_native_server_info", None)
    captured.pop("final_native_model_info", None)
    with pytest.raises(ValueError, match="post-run"):
        campaign.project_capture(plan, captured)
