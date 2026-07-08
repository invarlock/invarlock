from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from invarlock.evidence_pack import (
    EvidencePackStatus,
    verify_evidence_pack,
)
from invarlock.public_contracts import load_public_evidence_index, published_basis_lanes
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_valid_primary_ok_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert validate_report(report) is True
    assert report["validation"]["primary_metric_acceptable"] is True
    return report


def _assert_report_verifies(path: Path, assurance_mode: str = "report") -> None:
    result = run_verify_reports(
        [path],
        profile="release",
        assurance_mode=assurance_mode,
    )
    assert result.outcome == VerifyOutcome.OK


def _indexed_artifacts_by_path() -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in load_public_evidence_index()["entries"]:
        entry_artifacts = entry.get("artifacts", {})
        if not isinstance(entry_artifacts, dict):
            continue
        for summary in entry_artifacts.values():
            if not isinstance(summary, dict):
                continue
            path = summary.get("path")
            if isinstance(path, str):
                artifacts[path] = summary
    return artifacts


def _assert_local_or_indexed_artifact(
    rel_path: str,
    *,
    kind: str,
) -> dict[str, Any]:
    path = REPO_ROOT / rel_path
    if kind == "file" and path.is_file():
        return {"kind": "file", "path": rel_path}
    if kind == "directory" and path.is_dir():
        return {"kind": "directory", "path": rel_path}

    summary = _indexed_artifacts_by_path().get(rel_path)
    assert summary is not None, rel_path
    assert summary["kind"] == kind
    external = summary.get("external_asset")
    assert isinstance(external, dict), rel_path
    assert external["archive_path"] == rel_path
    assert external["sha256"].startswith("sha256:")
    assert external["size_bytes"] > 0
    assert external["url"].startswith("https://github.com/invarlock/invarlock/")
    return summary


def test_published_basis_lanes_ship_public_evidence_references() -> None:
    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        assert isinstance(evidence, dict)
        report_fixture = evidence.get("evaluation_report_fixture")
        evidence_pack_recipe = evidence.get("evidence_pack_recipe")
        assert isinstance(report_fixture, str) and report_fixture
        runtime_manifest = evidence.get("runtime_manifest_fixture")
        assert isinstance(runtime_manifest, str) and runtime_manifest
        assert isinstance(evidence_pack_recipe, str) and evidence_pack_recipe
        _assert_local_or_indexed_artifact(report_fixture, kind="file")
        _assert_local_or_indexed_artifact(runtime_manifest, kind="file")
        _assert_local_or_indexed_artifact(evidence_pack_recipe, kind="file")
        evidence_pack_fixture = evidence.get("evidence_pack_fixture")
        if evidence_pack_fixture is not None:
            assert isinstance(evidence_pack_fixture, str) and evidence_pack_fixture
            _assert_local_or_indexed_artifact(evidence_pack_fixture, kind="directory")


def test_packaged_public_evidence_index_matches_repo_public_evidence() -> None:
    index = load_public_evidence_index()
    assert index["carrier_policy"]["installed_wheel"] == "compact_index_only"
    indexed = {entry["slug"]: entry for entry in index["entries"]}

    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        report_fixture = evidence.get("evaluation_report_fixture")
        assert isinstance(report_fixture, str)
        basis_id = Path(report_fixture).parts[2]
        assert basis_id in indexed
        assert lane["lane_id"] in indexed[basis_id]["lanes"]

        artifacts = indexed[basis_id]["artifacts"]
        for key in ("evaluation_report", "runtime_manifest"):
            artifact = artifacts[key]
            source_path = REPO_ROOT / artifact["path"]
            if source_path.is_file():
                expected = (
                    "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
                )
                assert artifact["sha256"] == expected
            else:
                _assert_local_or_indexed_artifact(artifact["path"], kind="file")


def test_offline_golden_runs_public_fixtures() -> None:
    manifest_path = REPO_ROOT / "tests/artifacts/golden_runs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["published_basis"] == [
        "gpt2",
        "bert",
        "mistral_7b",
        "ministral3_8b",
        "ministral3_14b",
        "tinyllama_1_1b",
        "olmo2_7b",
        "olmo2_13b",
        "olmoe_1b_7b",
        "mixtral_8x7b",
        "qwen3_30b_a3b",
        "gpt_oss_20b",
        "open_llama_7b",
        "falcon_7b",
        "qwen2_7b",
        "qwen2_5_7b",
        "qwen2_5_14b",
        "qwen3_8b",
        "qwen3_5_9b",
        "qwen3_5_2b",
        "gemma4_e2b",
        "granite4_1_3b",
        "granite4_1_8b",
        "deepseek_r1_distill_qwen_7b",
        "deepseek_r1_0528_qwen3_8b",
        "phi4_reasoning_plus",
        "deepseek_r1_distill_qwen_14b",
        "ministral3_3b",
        "smollm3_3b",
        "phi4_mini",
        "qwen3_5_27b_scoped",
        "qwen3_6_27b_scoped",
        "gemma4_31b",
        "flan_t5_base",
    ]

    for lane in manifest["lanes"]:
        report_path = REPO_ROOT / lane["report"]
        if not report_path.is_file():
            _assert_local_or_indexed_artifact(lane["report"], kind="file")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert validate_report(report) is True
        assert report["meta"]["model_id"] == lane["model_id"]
        assert report["primary_metric"]["kind"] == lane["primary_metric_kind"]
        assert report["validation"]["primary_metric_acceptable"] is True


def test_published_basis_public_evidence_verifies_release_strict() -> None:
    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        report_fixture = evidence.get("evaluation_report_fixture")
        runtime_manifest = evidence.get("runtime_manifest_fixture")
        assert isinstance(report_fixture, str) and report_fixture
        assert isinstance(runtime_manifest, str) and runtime_manifest
        report_path = REPO_ROOT / report_fixture
        if not report_path.is_file():
            _assert_local_or_indexed_artifact(report_fixture, kind="file")
            _assert_local_or_indexed_artifact(runtime_manifest, kind="file")
            continue
        assert (REPO_ROOT / runtime_manifest).is_file()

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"


def test_public_signed_evidence_pack_verifies_release_strict_pinned() -> None:
    packs = []
    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        pack_fixture = evidence.get("evidence_pack_fixture")
        if isinstance(pack_fixture, str) and pack_fixture:
            packs.append(REPO_ROOT / pack_fixture)

    assert packs
    for pack_dir in packs:
        if not pack_dir.is_dir():
            _assert_local_or_indexed_artifact(
                pack_dir.relative_to(REPO_ROOT).as_posix(),
                kind="directory",
            )
            continue
        manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
        fingerprint = manifest["signing_key_fingerprint"]

        result = verify_evidence_pack(
            pack_dir,
            strict=True,
            profile="release",
            report_assurance="strict",
            expected_fingerprint=fingerprint,
        )

        assert result.status == EvidencePackStatus.OK
        assert result.payload["ok"] is True
        assert result.payload["authenticity"] == "pinned"
        assert result.payload["signer_fingerprint"] == fingerprint


def test_caught_regression_fixtures_fail_expected_guard() -> None:
    cases = {
        "spectral_guard_failure": (
            "validation.spectral_stable == true",
            "spectral did not pass",
        ),
        "rmt_guard_failure": ("validation.rmt_stable == true", "rmt did not pass"),
        "variance_guard_failure": (
            "variance.predictive_gate.passed == true",
            "variance did not pass",
        ),
    }

    for directory, expected_messages in cases.items():
        report_path = (
            REPO_ROOT
            / "public_evidence"
            / "caught_regressions"
            / directory
            / "evaluation.report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["validation"]["primary_metric_acceptable"] is True
        assert report["primary_metric"]["ratio_vs_baseline"] == 1.0

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.POLICY_FAIL
        diagnostics = "\n".join(item.message for item in result.diagnostics)
        for expected in expected_messages:
            assert expected in diagnostics


def test_caught_regressions_show_pm_only_passes_but_guard_chain_rejects() -> None:
    guard_failures = {
        "spectral_guard_failure": ("spectral", "spectral_stable"),
        "rmt_guard_failure": ("rmt", "rmt_stable"),
        "variance_guard_failure": ("variance", None),
    }

    for directory, (guard_key, validation_key) in guard_failures.items():
        report_path = (
            REPO_ROOT
            / "public_evidence"
            / "caught_regressions"
            / directory
            / "evaluation.report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        pm = report["primary_metric"]
        pm_only_accepts = (
            report["validation"]["primary_metric_acceptable"] is True
            and pm["ratio_vs_baseline"] == 1.0
        )
        assert pm_only_accepts is True

        guard = report[guard_key]
        assert guard["status"] == "fail"
        assert guard["passed"] is False
        if validation_key is not None:
            assert report["validation"][validation_key] is False
        else:
            assert report["variance"]["predictive_gate"]["passed"] is False

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )
        assert result.outcome == VerifyOutcome.POLICY_FAIL


def test_real_guard_value_demo_publishes_baseline_relative_spectral_catch() -> None:
    demo_dir = (
        REPO_ROOT
        / "public_evidence"
        / "published_basis"
        / "mistral_7b"
        / "guard_value_demo"
    )
    if not demo_dir.is_dir():
        summary = _assert_local_or_indexed_artifact(
            "public_evidence/published_basis/mistral_7b/guard_value_demo",
            kind="directory",
        )
        assert summary["file_count"] > 0
        assert summary["size_bytes"] > 0
        return
    summary = json.loads(
        (demo_dir / "guard_value_summary.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((demo_dir / "evidence.meta.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (demo_dir / "guard_value_manifest.json").read_text(encoding="utf-8")
    )
    final_verdict = json.loads(
        (demo_dir / "artifact_package" / "reports" / "final_verdict.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["evidence_class"] == "real_guard_value_demo"
    assert "fixture" not in metadata["summary"].lower()
    public_narrative = "\n".join(
        [
            (demo_dir / "README.md").read_text(encoding="utf-8"),
            json.dumps(metadata, sort_keys=True),
            json.dumps(summary, sort_keys=True),
            json.dumps(manifest["source_run"], sort_keys=True),
        ]
    )
    assert "root@" not in public_narrative
    assert re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", public_narrative) is None
    assert "The older FP8" not in public_narrative
    assert "FP8 stress report remains historical context" not in public_narrative
    assert summary["source_run"]["model_id"] == "mistralai/Mistral-7B-v0.1"
    assert summary["source_run"]["host"] == "self-hosted CUDA runner"
    assert manifest["source_run"]["host"] == "self-hosted CUDA runner"
    assert (
        summary["source_run"]["model_revision"]
        == "27d67f1b5f57dc0953326b2601d68371d40ea8da"
    )
    assert final_verdict["verdict"] == "PASS"
    assert final_verdict["counts"]["records_total"] == 5
    assert final_verdict["counts"]["error_injection_total"] == 4
    assert final_verdict["counts"]["primary_guard_required_hits"] == 1
    assert final_verdict["counts"]["error_injection_detected"] == 3
    assert (
        summary["final_verdict"]["contract_scope"]
        == "baseline_relative_spectral_guard_value_probe"
    )
    assert (
        summary["final_verdict"]["current_contract_verdict"]
        == "baseline_relative_guard_value_evidence"
    )
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    expected_manifest_paths = {
        "artifact_package/logs/run_pack.log",
        "artifact_package/reports/guard_value_all_guard_probe_sweep.json",
        "artifact_package/reports/errors/"
        "spectral_moderate_scale_mlp_l31_up_s112/evaluation.report.json",
        "artifact_package/reports/errors/"
        "spectral_moderate_scale_attn_l31_o_s105/evaluation.report.json",
        "artifact_package/reports/errors/"
        "rmt_norm_noise_l31_ffn_up_b030/evaluation.report.json",
        "artifact_package/reports/errors/"
        "ve_mlp_scale_skew_l31_down_s090/evaluation.report.json",
    }
    assert expected_manifest_paths <= manifest_paths
    packaged_attn_reports = {
        PurePosixPath(path).parts[3]
        for path in manifest_paths
        if path.startswith(
            "artifact_package/reports/errors/spectral_moderate_scale_attn_l31_o_"
        )
        and path.endswith("/evaluation.report.json")
    }
    assert packaged_attn_reports == {"spectral_moderate_scale_attn_l31_o_s105"}
    for entry in manifest["files"]:
        path = demo_dir / entry["path"]
        assert path.is_file(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]

    comparison = summary["pm_only_vs_pm_plus_guards"]
    assert comparison["scenario_id"] == "spectral_moderate_scale_mlp_l31_up_s112"
    assert comparison["pm_only_verdict"] == "accept"
    assert comparison["pm_plus_guards_verdict"] == "guard_value_detect"
    assert comparison["primary_metric_acceptable"] is True
    assert comparison["ratio_vs_baseline"] == 1.0076338080085065
    assert comparison["spectral_caps_applied"] == 3
    assert comparison["baseline_relative_guard_hit"] is True
    assert comparison["baseline_relative_evidence"]["new_caps_applied"] == 1
    assert comparison["baseline_relative_evidence"]["delta_caps_applied"] == 1
    assert comparison["baseline_relative_evidence"]["new_capped_modules"] == [
        {"family": "ffn", "module": "model.layers.31.mlp.up_proj"}
    ]
    assert comparison["primary_guard_required"] is True
    assert comparison["strictness"] == "must_detect"

    positive_report_path = demo_dir / comparison["report"]
    positive_report = _load_valid_primary_ok_report(positive_report_path)
    assert positive_report["spectral"]["caps_applied"] == 3
    assert positive_report["spectral"]["caps_applied_by_family"]["ffn"] == 1
    _assert_report_verifies(positive_report_path)

    control = summary["negative_control"]
    assert control["scenario_id"] == "spectral_moderate_scale_attn_l31_o_s105"
    assert control["primary_metric_acceptable"] is True
    assert control["baseline_relative_guard_hit"] is False
    assert control["baseline_relative_evidence"]["new_caps_applied"] == 0
    assert control["spectral_caps_applied"] == 2
    assert control["stock_cap_clean"] is True
    assert control["stock_attention_kappa"] == 3.018
    assert control["target_z"] == 2.7987064430328767
    assert control["target_margin_to_stock_kappa"] > 0.2

    control_report_path = demo_dir / control["report"]
    _load_valid_primary_ok_report(control_report_path)
    _assert_report_verifies(control_report_path)
    legacy_control_key = "margin" + "_policy_control"
    assert legacy_control_key not in summary

    all_guard = json.loads(
        (
            demo_dir
            / "artifact_package"
            / "reports"
            / "guard_value_all_guard_probe_sweep.json"
        ).read_text(encoding="utf-8")
    )
    assert all_guard["published_cases"] == [
        "spectral_moderate_scale_mlp_l31_up_s112",
        "rmt_norm_noise_l31_ffn_up_b030",
        "ve_mlp_scale_skew_l31_down_s090",
    ]
    assert all_guard["source_run"]["host"] == "self-hosted CUDA runner"
    assert all_guard["method"]["clean_confirmation_required"] is True
    assert legacy_control_key not in all_guard["guard_results"]["spectral"]
    assert (
        summary["all_guard_probe_sweep"]["guard_status"]["invariants"]
        == "not_a_statistical_margin_sweep"
    )

    rmt = all_guard["guard_results"]["rmt"]
    assert rmt["status"] == "published_reproduced_positive"
    assert rmt["baseline_relative_guard_hit"] is True
    assert rmt["clean_confirmation"]["primary_metric_acceptable"] is True
    assert rmt["clean_confirmation"]["ratio_vs_baseline"] == 1.0027430699936888
    rmt_probe = rmt["clean_confirmation"]["rmt_probe"]
    assert rmt_probe["stable"] is False
    assert rmt_probe["epsilon_violations"] == [
        {
            "allowed": 15.266412610841739,
            "delta": 0.12714696522015934,
            "edge_base": 15.115260010734394,
            "edge_cur": 17.037119449612906,
            "epsilon": 0.01,
            "family": "ffn",
            "module": "model.layers.31.mlp.up_proj",
        }
    ]
    rmt_report_path = demo_dir / rmt["clean_confirmation"]["report"]
    _load_valid_primary_ok_report(rmt_report_path)
    _assert_report_verifies(rmt_report_path)

    variance = all_guard["guard_results"]["variance"]
    assert variance["status"] == "published_reproduced_positive"
    assert variance["baseline_relative_guard_hit"] is True
    assert variance["baseline_self_probe"]["signal"] is False
    assert variance["clean_confirmation"]["primary_metric_acceptable"] is True
    assert variance["clean_confirmation"]["ratio_vs_baseline"] == 1.0002479838633067
    ve_probe = variance["clean_confirmation"]["ve_probe"]
    assert ve_probe["signal"] is True
    assert ve_probe["ab_gain"] == 0.003181692426797744
    assert ve_probe["abs_improvement"] == 0.554556828832176
    ve_report_path = demo_dir / variance["clean_confirmation"]["report"]
    _load_valid_primary_ok_report(ve_report_path)
    _assert_report_verifies(ve_report_path)

    sweep = json.loads(
        (
            demo_dir / "artifact_package" / "reports" / "guard_value_probe_sweep.json"
        ).read_text(encoding="utf-8")
    )
    assert sweep["method"]["calibration_rerun"] is False
    assert sweep["method"]["baseline_eval_rerun"] is False
    assert sweep["selected_positive"] == comparison["scenario_id"]
    assert sweep["selected_negative_control"] == control["scenario_id"]
    for record in sweep["records"]:
        log = record.get("log")
        if isinstance(log, str):
            assert (demo_dir / log).is_file(), log
        if record.get("log_packaged") is False:
            assert "not retained" in record["log_note"]
    records = {record["scenario_id"]: record for record in sweep["records"]}
    assert {
        record["scale_factor"]
        for record in sweep["records"]
        if record.get("target", {}).get("family") == "attn"
    } == {1.05, 1.18, 1.25}
    assert (
        records["spectral_moderate_scale_mlp_l31_up_s112"]["baseline_relative_guard"][
            "new_caps_applied"
        ]
        == 1
    )
    assert (
        records["spectral_moderate_scale_attn_l31_o_s105"]["baseline_relative_guard"][
            "new_caps_applied"
        ]
        == 0
    )
    assert records["spectral_moderate_scale_attn_l31_o_s105"]["stock_cap_clean"] is True
    assert (
        records["spectral_moderate_scale_attn_l31_o_s105"][
            "target_margin_to_stock_kappa"
        ]
        > 0.2
    )
    assert (
        records["spectral_moderate_scale_attn_l31_o_s118"]["spectral_caps_applied"] == 3
    )

    for directory in ("scale_explosion", "rank_collapse"):
        report_path = (
            demo_dir
            / "artifact_package"
            / "reports"
            / "errors"
            / directory
            / "evaluation.report.json"
        )
        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="report",
        )
        assert result.outcome == VerifyOutcome.POLICY_FAIL
