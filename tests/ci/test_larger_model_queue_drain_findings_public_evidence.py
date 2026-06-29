from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "larger_model_queue_drain_findings"

PRIVATE_TEXT_PATTERNS = (
    "/private/tmp",
    "/Users/",
    "/root",
    "root@",
    "private/remote host",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_larger_model_queue_drain_summary_is_public_safe_and_complete() -> None:
    summary = _load_json(EVIDENCE_DIR / "findings_summary.json")

    assert summary["schema"] == "invarlock.larger_model_queue_drain_findings.summary.v1"
    assert summary["status"] == "completed"
    assert summary["validation_environment"] == "CUDA-capable validation host"
    assert summary["raw_logs_published"] is False
    assert summary["weights_vendored"] is False
    assert summary["support_matrix_change_claimed"] is False
    assert summary["source_window"] == "post_batch_18_cutoff"
    assert summary["suite"] == "model-catalog-gpu"
    assert summary["execution_mode"] == "container"

    counts = summary["counts"]
    assert counts == {
        "completed_runs": 44,
        "clean_runs": 40,
        "failed_runs": 4,
        "unique_clean_lanes": 28,
        "unique_failed_lanes": 4,
        "pre_verification_failures": 4,
        "report_materialized_clean": 40,
        "verify_materialized_clean": 40,
    }

    clean_lanes = summary["clean_lanes"]
    assert isinstance(clean_lanes, list)
    assert len(clean_lanes) == counts["unique_clean_lanes"]
    for lane in clean_lanes:
        assert isinstance(lane, dict)
        assert lane["rc"] == 0
        assert lane["evaluate_exit"] == 0
        assert lane["verify_exit"] == 0
        assert lane["report_materialized"] is True
        assert lane["verify_materialized"] is True
        assert lane["status"] == "ok"
        preset = REPO_ROOT / str(lane["preset"])
        assert preset.is_file()

    duplicate_runs = summary["duplicate_clean_runs"]
    assert isinstance(duplicate_runs, list)
    assert {entry["slug"] for entry in duplicate_runs if isinstance(entry, dict)} == {
        "bert_base_uncased",
        "distilbert_base_uncased",
        "google_flan_t5_base",
        "google_gemma_4_12b_it",
        "microsoft_phi_3_mini_4k_instruct",
        "openai_community_gpt2",
        "openai_gpt_oss_20b",
        "roberta_base",
    }

    failed_findings = summary["failed_findings"]
    assert isinstance(failed_findings, list)
    assert len(failed_findings) == counts["unique_failed_lanes"]
    classifications = {
        finding["slug"]: finding["classification"]
        for finding in failed_findings
        if isinstance(finding, dict)
    }
    assert classifications == {
        "mistralai_ministral_3_3b_instruct_2512_bf16": (
            "initial_attempt_failed_later_clean"
        ),
        "huggingfacetb_smollm3_3b": "initial_attempt_failed_later_clean",
        "qwen_qwen3_30b_a3b_instruct_2507": "pre_verification_evaluate_failure",
        "01_ai_yi_34b": "grouped_execution_cuda_failure_later_clean",
    }
    for finding in failed_findings:
        assert isinstance(finding, dict)
        assert finding["attempts"] == 1
        assert finding["rc"] == 1
        assert finding["evaluate_exit"] == 1
        assert finding["verify_exit"] is None
        assert finding["verify_materialized"] is False
        assert (REPO_ROOT / str(finding["preset"])).is_file()

    serialized = json.dumps(summary, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized


def test_larger_model_queue_drain_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert inventory["schema"] == (
        "invarlock.larger_model_queue_drain_findings.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    artifacts = inventory["artifacts"]
    assert isinstance(artifacts, list)

    by_path = {artifact["path"]: artifact for artifact in artifacts}
    assert set(by_path) == {
        "README.md",
        "findings_summary.json",
        "late_clean_addendum.json",
        "evidence.meta.json",
    }
    for rel_path, artifact in by_path.items():
        path = EVIDENCE_DIR / rel_path
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size


def test_larger_model_queue_drain_metadata_declares_summary_only_findings() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "larger_model_queue_drain_findings"
    assert metadata["artifact_paths"] == {
        "findings_summary": "findings_summary.json",
        "late_clean_addendum": "late_clean_addendum.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "invarlock evaluate" not in str(metadata["generated_by"])


def test_larger_model_queue_drain_late_clean_addendum_is_public_safe() -> None:
    addendum = _load_json(EVIDENCE_DIR / "late_clean_addendum.json")

    assert (
        addendum["schema"]
        == "invarlock.larger_model_queue_drain_findings.late_clean_addendum.v1"
    )
    assert addendum["status"] == "completed"
    assert addendum["validation_environment"] == "CUDA-capable validation host"
    assert addendum["raw_logs_published"] is False
    assert addendum["weights_vendored"] is False
    assert addendum["support_matrix_change_claimed"] is False
    assert addendum["model_quality_claimed"] is False
    assert addendum["source_window"] == "post_pr_109_late_clean_addendum"
    assert addendum["execution_mode"] == "container"

    clean_lanes = addendum["late_clean_lanes"]
    assert isinstance(clean_lanes, list)
    assert {lane["slug"] for lane in clean_lanes if isinstance(lane, dict)} == {
        "google_gemma_4_26b_a4b_it",
        "mistralai_mixtral_8x7b_v0_1",
        "qwen_qwen3_30b_a3b_instruct_2507",
    }
    assert addendum["counts"] == {
        "late_clean_lanes": len(clean_lanes),
        "rerun_clean_resolutions": 1,
        "excluded_lanes": 1,
    }
    for lane in clean_lanes:
        assert isinstance(lane, dict)
        assert lane["suite"] == "model-catalog-gpu"
        assert lane["rc"] == 0
        assert lane["evaluate_exit"] == 0
        assert lane["verify_exit"] == 0
        assert lane["report_materialized"] is True
        assert lane["verify_materialized"] is True
        assert lane["status"] == "ok"
        assert (REPO_ROOT / str(lane["preset"])).is_file()

    rerun_classifications = addendum["rerun_classifications"]
    assert rerun_classifications == [
        {
            "slug": "qwen_qwen3_30b_a3b_instruct_2507",
            "previous_classification": "pre_verification_evaluate_failure",
            "later_clean_run_observed": True,
            "public_note": (
                "A later rerun completed cleanly with evaluation and strict "
                "verification exit 0."
            ),
        }
    ]

    excluded_lanes = addendum["excluded_lanes"]
    assert excluded_lanes == [
        {
            "slug": "qwen_qwen2_5_32b",
            "model_id": "Qwen/Qwen2.5-32B",
            "reason": (
                "Held out because the late run used a generic preset rather "
                "than a model-specific repo preset at packaging time."
            ),
        }
    ]

    serialized = json.dumps(addendum, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized
