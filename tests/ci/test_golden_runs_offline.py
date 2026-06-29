from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from invarlock.evidence_pack import (
    EvidencePackStatus,
    _generate_signing_keypair,
    build_evidence_pack,
    verify_evidence_pack,
)
from invarlock.public_contracts import load_public_evidence_index, published_basis_lanes
from invarlock.reporting.report_schema import validate_report
from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json_file(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        assert (REPO_ROOT / report_fixture).is_file(), report_fixture
        assert (REPO_ROOT / runtime_manifest).is_file(), runtime_manifest
        assert (REPO_ROOT / evidence_pack_recipe).is_file(), evidence_pack_recipe
        evidence_pack_fixture = evidence.get("evidence_pack_fixture")
        if evidence_pack_fixture is not None:
            assert isinstance(evidence_pack_fixture, str) and evidence_pack_fixture
            assert (REPO_ROOT / evidence_pack_fixture).is_dir(), evidence_pack_fixture


def test_packaged_public_evidence_index_matches_repo_public_evidence() -> None:
    index = load_public_evidence_index()
    assert index["carrier_policy"]["installed_wheel"] == "compact_index_only"
    indexed = {entry["slug"]: entry for entry in index["entries"]}

    for lane in published_basis_lanes():
        evidence = lane.get("evidence", {})
        report_fixture = evidence.get("evaluation_report_fixture")
        assert isinstance(report_fixture, str)
        basis_id = Path(report_fixture).parts[2]
        source_dir = REPO_ROOT / "public_evidence" / "published_basis" / basis_id
        assert source_dir.is_dir(), source_dir
        assert basis_id in indexed
        assert lane["lane_id"] in indexed[basis_id]["lanes"]

        artifacts = indexed[basis_id]["artifacts"]
        for key in ("evaluation_report", "runtime_manifest"):
            artifact = artifacts[key]
            source_path = REPO_ROOT / artifact["path"]
            assert source_path.is_file(), source_path
            expected = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
            assert artifact["sha256"] == expected


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
    assert "artifact_package/logs/run_pack.log" in manifest_paths
    assert (
        "artifact_package/reports/errors/"
        "spectral_moderate_scale_mlp_l31_up_s112/evaluation.report.json"
        in manifest_paths
    )
    assert (
        "artifact_package/reports/errors/"
        "spectral_moderate_scale_attn_l31_o_s112/evaluation.report.json"
        in manifest_paths
    )
    assert (
        "artifact_package/reports/guard_value_all_guard_probe_sweep.json"
        in manifest_paths
    )
    assert (
        "artifact_package/reports/errors/"
        "rmt_norm_noise_l31_ffn_up_b030/evaluation.report.json" in manifest_paths
    )
    assert (
        "artifact_package/reports/errors/"
        "ve_mlp_scale_skew_l31_down_s090/evaluation.report.json" in manifest_paths
    )
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
    positive_report = json.loads(positive_report_path.read_text(encoding="utf-8"))
    assert validate_report(positive_report) is True
    assert positive_report["validation"]["primary_metric_acceptable"] is True
    assert positive_report["spectral"]["caps_applied"] == 3
    assert positive_report["spectral"]["caps_applied_by_family"]["ffn"] == 1

    result = run_verify_reports(
        [positive_report_path],
        profile="release",
        assurance_mode="report",
    )
    assert result.outcome == VerifyOutcome.OK

    control = summary["negative_control"]
    assert control["scenario_id"] == "spectral_moderate_scale_attn_l31_o_s112"
    assert control["primary_metric_acceptable"] is True
    assert control["baseline_relative_guard_hit"] is False
    assert control["baseline_relative_evidence"]["new_caps_applied"] == 0
    assert control["spectral_caps_applied"] == 2

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
    rmt_report = json.loads(rmt_report_path.read_text(encoding="utf-8"))
    assert validate_report(rmt_report) is True
    assert rmt_report["validation"]["primary_metric_acceptable"] is True
    assert (
        run_verify_reports(
            [rmt_report_path],
            profile="release",
            assurance_mode="report",
        ).outcome
        == VerifyOutcome.OK
    )

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
    ve_report = json.loads(ve_report_path.read_text(encoding="utf-8"))
    assert validate_report(ve_report) is True
    assert ve_report["validation"]["primary_metric_acceptable"] is True
    assert (
        run_verify_reports(
            [ve_report_path],
            profile="release",
            assurance_mode="report",
        ).outcome
        == VerifyOutcome.OK
    )

    sweep = json.loads(
        (
            demo_dir / "artifact_package" / "reports" / "guard_value_probe_sweep.json"
        ).read_text(encoding="utf-8")
    )
    assert sweep["method"]["calibration_rerun"] is False
    assert sweep["method"]["baseline_eval_rerun"] is False
    assert sweep["selected_positive"] == comparison["scenario_id"]
    for record in sweep["records"]:
        log = record.get("log")
        if isinstance(log, str):
            assert (demo_dir / log).is_file(), log
        if record.get("log_packaged") is False:
            assert "not retained" in record["log_note"]
    records = {record["scenario_id"]: record for record in sweep["records"]}
    assert (
        records["spectral_moderate_scale_mlp_l31_up_s112"]["baseline_relative_guard"][
            "new_caps_applied"
        ]
        == 1
    )
    assert (
        records["spectral_moderate_scale_attn_l31_o_s112"]["baseline_relative_guard"][
            "new_caps_applied"
        ]
        == 0
    )
    assert (
        records["spectral_moderate_scale_attn_l31_o_s118"]["spectral_caps_applied"] == 3
    )

    expected_failures = [
        demo_dir
        / "artifact_package"
        / "reports"
        / "errors"
        / "scale_explosion"
        / "evaluation.report.json",
        demo_dir
        / "artifact_package"
        / "reports"
        / "errors"
        / "rank_collapse"
        / "evaluation.report.json",
    ]
    for report_path in expected_failures:
        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="report",
        )
        assert result.outcome == VerifyOutcome.POLICY_FAIL


def test_public_docs_route_to_mistral_guard_value_evidence() -> None:
    docs = {
        "README.md": REPO_ROOT / "README.md",
        "public_evidence/README.md": REPO_ROOT / "public_evidence" / "README.md",
        "scripts/evidence_packs/README.md": (
            REPO_ROOT / "scripts" / "evidence_packs" / "README.md"
        ),
        "docs/user-guide/evidence-packs-internals.md": (
            REPO_ROOT / "docs" / "user-guide" / "evidence-packs-internals.md"
        ),
        "docs/reference/guards.md": REPO_ROOT / "docs" / "reference" / "guards.md",
    }

    for label, path in docs.items():
        text = path.read_text(encoding="utf-8")
        assert "public_evidence/published_basis/mistral_7b/guard_value_demo" in text, (
            label
        )
        assert "baseline-relative" in text, label

    public_evidence_readme = docs["public_evidence/README.md"].read_text(
        encoding="utf-8"
    )
    assert "PM-only accepts" in public_evidence_readme
    assert "PM+guards" in public_evidence_readme
    assert "caught_regressions/` entries remain useful verifier fixtures" in (
        public_evidence_readme
    )

    pack_readme = docs["scripts/evidence_packs/README.md"].read_text(encoding="utf-8")
    assert "Guard-value publishing rule" in pack_readme
    assert "Clean confirmation reruns are required" in pack_readme
    assert "guard_value_all_guard_probe_sweep.json" in pack_readme


def test_policy_failure_fixtures_fail_expected_policy_predicate() -> None:
    cases = {
        "invariants_failure": (
            "validation.invariants_pass == true",
            "invariants did not pass",
        ),
        "primary_metric_failure": (
            "Primary metric policy gate failed",
            "validation.primary_metric_acceptable == true",
        ),
        "runtime_provenance_failure": (
            "runtime.manifest.json marks evaluation.report.json as 'host-bypass'",
            "strict assurance requires verified runtime provenance",
        ),
    }

    for directory, expected_messages in cases.items():
        report_path = (
            REPO_ROOT
            / "public_evidence"
            / "policy_failures"
            / directory
            / "evaluation.report.json"
        )

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.POLICY_FAIL
        diagnostics = "\n".join(item.message for item in result.diagnostics)
        for expected in expected_messages:
            assert expected in diagnostics


def test_byoe_examples_verify_release_strict() -> None:
    examples = {
        "magnitude_prune_byoe": "magnitude_prune",
        "lora_merge_byoe": "lora_merge",
        "fine_tune_byoe": "fine_tune",
    }

    for directory, edit_type in examples.items():
        example_dir = REPO_ROOT / "public_evidence" / "byoe_examples" / directory
        report_path = example_dir / "evaluation.report.json"
        refs_path = example_dir / "checkpoint_refs.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        refs = json.loads(refs_path.read_text(encoding="utf-8"))

        assert validate_report(report) is True
        assert report["artifacts"]["byoe_example"] is True
        assert report["artifacts"]["external_edit_type"] == edit_type
        assert report["artifacts"]["built_in_edit_plugin"] is False
        assert report["plugins"]["edits"] == []
        assert refs["weights_vendored"] is False
        assert refs["subject_checkpoint"]["external_edit_type"] == edit_type
        assert refs["subject_checkpoint"]["built_in_edit_plugin"] is False
        if directory in {"lora_merge_byoe", "fine_tune_byoe"}:
            edit = report["edit"]
            assert edit["edit_provenance"]["edit_family"] == edit_type
            assert edit["edit_provenance"]["edit_count"] == 1
            assert edit["edit_provenance"]["dynamic_runtime_required"] is False
            if directory == "lora_merge_byoe":
                assert edit["edit_provenance"]["edit_method"] == "custom"
                assert edit["edit_impact"]["scenario_types"] == [
                    "target_success",
                    "near_neighbor",
                    "unrelated_locality",
                    "general_ability_sentinel",
                ]
            else:
                assert (
                    edit["edit_provenance"]["edit_method"]
                    == "external_cpu_tiny_fine_tune"
                )
            assert (
                refs["subject_checkpoint"]["edit_provenance"]
                == (edit["edit_provenance"])
            )
            if "edit_impact" in edit:
                assert refs["subject_checkpoint"]["edit_impact"] == edit["edit_impact"]

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"


def test_model_editing_evidence_bundle_v0_lanes_verify_release_strict() -> None:
    bundle_dir = REPO_ROOT / "public_evidence" / "model_editing_evidence_bundle_v0"
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = REPO_ROOT / manifest["verification_summary"]
    training_plan_path = REPO_ROOT / manifest["training_evidence_matrix_plan"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_families = {"quantization", "magnitude_prune", "lora_merge", "fine_tune"}
    lanes = manifest["lanes"]
    assert {lane["edit_family"] for lane in lanes} == expected_families
    assert manifest["evidence_scope"] == "release-evidence wiring only"
    assert summary["schema"] == (
        "invarlock.public_evidence.model_editing_bundle_verification.v1"
    )
    assert summary["bundle_id"] == manifest["bundle_id"]
    assert summary["evidence_scope"] == manifest["evidence_scope"]
    assert summary["verification"] == {
        "assurance": "strict",
        "lane_count": 4,
        "outcome": "all_lanes_verified",
        "profile": "release",
    }
    assert training_plan_path.is_file()
    training_plan = training_plan_path.read_text(encoding="utf-8")
    assert "PEFT LoRA train-and-merge subject" in training_plan
    assert "Full fine-tune subject" in training_plan
    assert "/private/tmp" not in training_plan
    assert "root@" not in training_plan

    summary_lanes = {lane["edit_family"]: lane for lane in summary["lanes"]}
    assert set(summary_lanes) == expected_families
    assert {
        summary_lanes["quantization"]["evidence_mode"],
        summary_lanes["magnitude_prune"]["evidence_mode"],
    } == {"real_tiny_model_run", "real_tiny_model_external_edit_run"}
    assert summary_lanes["lora_merge"]["evidence_mode"] == "public_byoe_subject_fixture"
    assert summary_lanes["fine_tune"]["evidence_mode"] == "public_byoe_subject_fixture"

    for lane in lanes:
        report_path = REPO_ROOT / lane["evaluation_report"]
        refs_path = REPO_ROOT / lane["checkpoint_refs"]
        note_path = REPO_ROOT / lane["evidence_note"]
        summary_lane = summary_lanes[lane["edit_family"]]

        report = json.loads(report_path.read_text(encoding="utf-8"))
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        note = " ".join(note_path.read_text(encoding="utf-8").split())

        assert validate_report(report) is True
        assert (
            refs["subject_checkpoint"]["external_edit_type"]
            == lane["external_edit_type"]
        )
        assert "Evidence takeaways" in note
        assert "Artifact mode:" in note
        assert "Verification surface:" in note
        assert "Companion benchmark evidence:" in note
        assert "/private/tmp" not in note
        assert "root@" not in note
        assert summary_lane["external_edit_type"] == lane["external_edit_type"]
        assert summary_lane["weights_vendored"] is False
        assert summary_lane["strict_verification"] == {
            "assurance": "strict",
            "outcome": "ok",
            "profile": "release",
            "runtime_provenance_status": "verified",
        }
        for key, expected_path in {
            "evaluation_report": lane["evaluation_report"],
            "runtime_manifest": lane["runtime_manifest"],
            "checkpoint_refs": lane["checkpoint_refs"],
            "evidence_note": lane["evidence_note"],
        }.items():
            artifact = summary_lane["artifacts"][key]
            assert artifact["path"] == expected_path
            assert artifact["sha256"] == _sha256_file(REPO_ROOT / expected_path)

        if lane["edit_family"] in {"lora_merge", "fine_tune"}:
            assert (
                report["edit"]["edit_provenance"]["edit_family"] == lane["edit_family"]
            )

        result = run_verify_reports(
            [report_path],
            profile="release",
            assurance_mode="strict",
        )

        assert result.outcome == VerifyOutcome.OK
        verification = result.payload["results"][0]["verification"]
        assert verification["runtime_provenance"]["status"] == "verified"


def test_lora_byoe_metadata_builds_and_verifies_signed_evidence_pack(
    tmp_path: Path,
) -> None:
    example_dir = REPO_ROOT / "public_evidence" / "byoe_examples" / "lora_merge_byoe"
    report_path = example_dir / "evaluation.report.json"
    refs_path = example_dir / "checkpoint_refs.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_provenance = report["edit"]["edit_provenance"]
    expected_impact = report["edit"]["edit_impact"]

    final_verdict = tmp_path / "final_verdict.json"
    signing_key = tmp_path / "evidence-pack-signing-key.pem"
    public_key = tmp_path / "evidence-pack-signing-key.pub.pem"
    pack_dir = tmp_path / "lora_byoe_evidence_pack"
    _write_json_file(
        final_verdict,
        {
            "verdict": "PASS",
            "scope": "lora_merge_byoe_optional_edit_metadata_fixture",
        },
    )
    fingerprint = _generate_signing_keypair(
        signing_key,
        public_key_path=public_key,
    )

    build_result = build_evidence_pack(
        pack_dir,
        final_verdict_path=final_verdict,
        report_paths=[report_path],
        material_specs=[("checkpoint_refs", refs_path)],
        signing_key_path=signing_key,
        profile="release",
        report_assurance="strict",
        release_review=True,
    )

    assert build_result.status == EvidencePackStatus.OK
    assert build_result.payload["ok"] is True
    assert build_result.payload["signature"]["present"] is True
    assert build_result.payload["signature"]["signer_fingerprint"] == fingerprint
    assert build_result.payload["verify"]["summary"]["ok"] is True
    assert build_result.payload["verify"]["results"][0]["ok"] is True

    copied_reports = sorted(pack_dir.glob("reports/**/evaluation.report.json"))
    assert len(copied_reports) == 1
    copied_report = json.loads(copied_reports[0].read_text(encoding="utf-8"))
    assert validate_report(copied_report) is True
    assert copied_report["edit"]["edit_provenance"] == expected_provenance
    assert copied_report["edit"]["edit_impact"] == expected_impact

    copied_refs = json.loads(
        (pack_dir / "metadata" / "checkpoint_refs.json").read_text(encoding="utf-8")
    )
    assert copied_refs["subject_checkpoint"]["edit_provenance"] == expected_provenance
    assert copied_refs["subject_checkpoint"]["edit_impact"] == expected_impact

    verify_result = verify_evidence_pack(
        pack_dir,
        strict=True,
        expected_fingerprint=fingerprint,
        profile="release",
        report_assurance="strict",
    )

    assert verify_result.status == EvidencePackStatus.OK
    assert verify_result.payload["ok"] is True
    assert verify_result.payload["authenticity"] == "pinned"
    assert verify_result.payload["verify"]["summary"]["ok"] is True
    assert verify_result.payload["verify"]["results"][0]["ok"] is True
    verification = verify_result.payload["verify"]["results"][0]["verification"]
    assert verification["runtime_provenance"]["status"] == "verified"
