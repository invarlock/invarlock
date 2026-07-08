from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "public_evidence" / "larger_model_validation_findings"

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


def _categories(outcomes: dict[str, object]) -> dict[str, dict[str, object]]:
    categories = outcomes["categories"]
    assert isinstance(categories, list)
    by_name = {
        category["category"]: category
        for category in categories
        if isinstance(category, dict)
    }
    assert set(by_name) == {
        "bounded_smoke_matrix",
        "initial_validation_matrix",
        "clean_resolutions",
        "followup_lanes",
        "published_basis_verification",
    }
    return by_name


def _assert_clean_lane(lane: dict[str, object]) -> None:
    assert lane["rc"] == 0
    assert lane["evaluate_exit"] == 0
    assert lane["verify_exit"] == 0
    assert lane["report_materialized"] is True
    assert lane["verify_materialized"] is True
    assert lane["status"] == "ok"
    assert (REPO_ROOT / str(lane["preset"])).is_file()


def _assert_public_or_externalized_artifact(rel_path: object) -> None:
    assert isinstance(rel_path, str)
    assert rel_path.startswith("public_evidence/published_basis/")
    assert not Path(rel_path).is_absolute()
    if (REPO_ROOT / rel_path).exists():
        return
    # Published-basis reports are intentionally synced outside compact checkouts.
    assert (
        "/reports/report-001/evaluation.report.json" in rel_path
        or rel_path.endswith("/runtime.manifest.json")
    )


def test_larger_model_validation_lane_outcomes_are_public_safe() -> None:
    outcomes = _load_json(EVIDENCE_DIR / "lane_outcomes.json")

    assert (
        outcomes["schema"]
        == "invarlock.larger_model_validation_findings.lane_outcomes.v1"
    )
    assert outcomes["status"] == "completed"
    assert outcomes["validation_environment"] == "CUDA-capable validation host"
    assert outcomes["raw_logs_published"] is False
    assert outcomes["weights_vendored"] is False
    assert outcomes["support_matrix_change_claimed"] is False
    assert outcomes["model_quality_claimed"] is False
    assert outcomes["execution_mode"] == "container"
    assert outcomes["counts"] == {
        "categories": 5,
        "bounded_smoke_clean_lanes": 17,
        "bounded_smoke_failed_lanes": 1,
        "initial_clean_lanes": 28,
        "initial_failed_lanes": 4,
        "clean_resolution_lanes": 3,
        "followup_clean_lanes": 2,
        "followup_diagnostic_lanes": 4,
        "followup_strict_policy_findings": 2,
        "published_basis_clean_lanes": 3,
        "published_basis_followup_lanes": 2,
    }

    by_name = _categories(outcomes)

    smoke = by_name["bounded_smoke_matrix"]
    assert smoke["source_window"] == "bounded_smoke_matrix"
    assert smoke["suite"] == "model-catalog-gpu"
    smoke_clean_lanes = smoke["clean_lanes"]
    assert isinstance(smoke_clean_lanes, list)
    assert len(smoke_clean_lanes) == 17
    for lane in smoke_clean_lanes:
        assert isinstance(lane, dict)
        _assert_clean_lane(lane)

    smoke_duplicate_runs = smoke["duplicate_clean_runs"]
    assert isinstance(smoke_duplicate_runs, list)
    assert {
        entry["slug"] for entry in smoke_duplicate_runs if isinstance(entry, dict)
    } == {
        "google_flan_t5_base",
        "tinyllama_tinyllama_1_1b_chat_v1_0",
    }
    smoke_failed_findings = smoke["failed_findings"]
    assert smoke_failed_findings == [
        {
            "slug": "microsoft_phi_4_mini_instruct",
            "model_id": "microsoft/Phi-4-mini-instruct",
            "preset": "configs/presets/causal_lm/phi4_mini_512.yaml",
            "attempts": 2,
            "status": "evaluate_failed_before_report",
            "rc": 1,
            "evaluate_exit": 1,
            "verify_exit": None,
            "report_materialized": False,
            "verify_materialized": False,
            "classification": "pre_verification_evaluate_failure",
            "public_note": (
                "Evaluation exited nonzero before report or verifier artifacts were "
                "materialized. This finding is not counted as clean evidence."
            ),
        }
    ]

    initial = by_name["initial_validation_matrix"]
    assert initial["source_window"] == "initial_validation_matrix"
    assert initial["suite"] == "model-catalog-gpu"
    clean_lanes = initial["clean_lanes"]
    assert isinstance(clean_lanes, list)
    assert len(clean_lanes) == 28
    for lane in clean_lanes:
        assert isinstance(lane, dict)
        _assert_clean_lane(lane)

    duplicate_runs = initial["duplicate_clean_runs"]
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

    failed_findings = initial["failed_findings"]
    assert isinstance(failed_findings, list)
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

    resolutions = by_name["clean_resolutions"]
    assert resolutions["source_window"] == "validation_resolution_runs"
    resolution_lanes = resolutions["clean_resolution_lanes"]
    assert isinstance(resolution_lanes, list)
    assert {lane["slug"] for lane in resolution_lanes if isinstance(lane, dict)} == {
        "google_gemma_4_26b_a4b_it",
        "mistralai_mixtral_8x7b_v0_1",
        "qwen_qwen3_30b_a3b_instruct_2507",
    }
    for lane in resolution_lanes:
        assert isinstance(lane, dict)
        _assert_clean_lane(lane)

    followup = by_name["followup_lanes"]
    assert followup["source_window"] == "model_family_followup_runs"
    clean_followup = followup["clean_followup_lanes"]
    diagnostics = followup["diagnostic_lanes"]
    strict_findings = followup["strict_policy_findings"]
    assert isinstance(clean_followup, list)
    assert isinstance(diagnostics, list)
    assert isinstance(strict_findings, list)
    assert {lane["slug"] for lane in clean_followup if isinstance(lane, dict)} == {
        "google_gemma_4_31b_it",
        "openai_gpt_oss_20b",
    }
    assert {lane["slug"] for lane in diagnostics if isinstance(lane, dict)} == {
        "qwen_qwen3_5_27b_dev",
        "qwen_qwen3_5_27b_maxcaps6",
        "qwen_qwen3_6_27b_dev",
        "qwen_qwen3_6_27b_maxcaps6",
    }
    assert {
        finding["slug"] for finding in strict_findings if isinstance(finding, dict)
    } == {
        "qwen_qwen3_5_27b",
        "qwen_qwen3_6_27b",
    }

    published_basis = by_name["published_basis_verification"]
    assert published_basis["source_window"] == "published_basis_verification_runs"
    published_basis_lanes = published_basis["published_basis_clean_lanes"]
    followup_lanes = published_basis["followup_clean_lanes"]
    assert isinstance(published_basis_lanes, list)
    assert isinstance(followup_lanes, list)
    assert {
        lane["slug"] for lane in published_basis_lanes if isinstance(lane, dict)
    } == {
        "google_gemma_4_e2b_it_image_text",
        "qwen_qwen3_5_4b",
        "qwen_qwen3_5_2b",
    }
    assert {lane["slug"] for lane in followup_lanes if isinstance(lane, dict)} == {
        "qwen3_8b_public",
        "qwen3_5_9b_public",
    }
    for lane in [*published_basis_lanes, *followup_lanes]:
        assert isinstance(lane, dict)
        _assert_clean_lane(lane)
        assert lane["summary_ok"] is True
        assert lane["verify_summary_ok"] is True
        assert lane["runtime_provenance_verified"] is True
        assert lane["guard_warnings_present"] is False
        assert lane["warning_count"] == 0
        _assert_public_or_externalized_artifact(lane["existing_public_evidence_report"])
        _assert_public_or_externalized_artifact(
            lane["existing_public_runtime_manifest"]
        )
        metric = lane["metric"]
        assert isinstance(metric, dict)
        assert metric["kind"] in {"accuracy", "ppl_causal"}
        assert isinstance(metric["final"], float)
        assert isinstance(metric["ratio_vs_baseline"], float)

    serialized = json.dumps(outcomes, sort_keys=True)
    for pattern in PRIVATE_TEXT_PATTERNS:
        assert pattern not in serialized


def test_larger_model_validation_hash_inventory_matches_public_files() -> None:
    inventory = _load_json(EVIDENCE_DIR / "hash_inventory.json")

    assert inventory["schema"] == (
        "invarlock.larger_model_validation_findings.hash_inventory.v1"
    )
    assert inventory["status"] == "completed"
    artifacts = inventory["artifacts"]
    assert isinstance(artifacts, list)

    by_path = {artifact["path"]: artifact for artifact in artifacts}
    assert set(by_path) == {
        "README.md",
        "lane_outcomes.json",
        "evidence.meta.json",
    }
    for rel_path, artifact in by_path.items():
        path = EVIDENCE_DIR / rel_path
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size


def test_larger_model_validation_metadata_declares_summary_only_findings() -> None:
    metadata = _load_json(EVIDENCE_DIR / "evidence.meta.json")

    assert metadata["schema"] == "invarlock.public_evidence.meta.v1"
    assert metadata["evidence_class"] == "larger_model_validation_findings"
    assert metadata["artifact_paths"] == {
        "lane_outcomes": "lane_outcomes.json",
        "hash_inventory": "hash_inventory.json",
    }
    assert "invarlock evaluate" not in str(metadata["generated_by"])
