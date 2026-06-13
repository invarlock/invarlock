from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import invarlock.public_contracts as contracts
from invarlock import evidence_pack_integrity, runtime_security_helpers
from invarlock.cli import constants as cli_constants
from invarlock.reporting import report_schema, verify_output
from tests._repo_root import REPO_ROOT


def test_public_subcontract_versions_are_single_sourced() -> None:
    assert report_schema.REPORT_SCHEMA_VERSION == contracts.REPORT_SCHEMA_VERSION
    assert (
        evidence_pack_integrity.EVIDENCE_PACK_FORMAT
        == contracts.EVIDENCE_PACK_FORMAT_VERSION
    )
    assert verify_output.FORMAT_VERIFY == contracts.VERIFY_OUTPUT_FORMAT_VERSION
    assert (
        runtime_security_helpers.RUNTIME_VERIFIER_CONTRACT_VERSION
        == contracts.RUNTIME_MANIFEST_CONTRACT_VERSION
    )

    assert cli_constants.DOCTOR_FORMAT_VERSION == contracts.DOCTOR_OUTPUT_FORMAT_VERSION
    assert (
        cli_constants.PLUGINS_FORMAT_VERSION == contracts.PLUGINS_OUTPUT_FORMAT_VERSION
    )
    assert cli_constants.VERIFY_FORMAT_VERSION == contracts.VERIFY_OUTPUT_FORMAT_VERSION
    assert (
        cli_constants.RUNTIME_VERIFY_FORMAT_VERSION
        == contracts.RUNTIME_VERIFY_OUTPUT_FORMAT_VERSION
    )
    assert (
        cli_constants.POLICY_PACK_VERIFY_FORMAT_VERSION
        == contracts.POLICY_PACK_VERIFY_OUTPUT_FORMAT_VERSION
    )
    assert (
        cli_constants.EVIDENCE_PACK_VERIFY_FORMAT_VERSION
        == contracts.EVIDENCE_PACK_VERIFY_OUTPUT_FORMAT_VERSION
    )

    subcontract_catalog = contracts.public_subcontract_catalog()
    assert (
        subcontract_catalog["report_schema"]["version"]
        == contracts.REPORT_SCHEMA_VERSION
    )
    assert (
        subcontract_catalog["evidence_pack_format"]["version"]
        == contracts.EVIDENCE_PACK_FORMAT_VERSION
    )
    assert (
        subcontract_catalog["verifier_output"]["version"]
        == contracts.VERIFY_OUTPUT_FORMAT_VERSION
    )
    assert (
        subcontract_catalog["runtime_manifest"]["version"]
        == contracts.RUNTIME_MANIFEST_CONTRACT_VERSION
    )
    assert (
        subcontract_catalog["cli_stability_policy"]["version"]
        == contracts.CLI_STABILITY_POLICY_VERSION
    )
    assert (
        subcontract_catalog["adapter_support_tiers"]["version"]
        == contracts.ADAPTER_SUPPORT_TIER_POLICY_VERSION
    )
    assert (
        subcontract_catalog["cli_stability_policy"]["stable_json_surfaces"]
        == contracts.stable_cli_json_surfaces()
    )


def test_public_contract_loaders_and_catalog_round_trip() -> None:
    support_matrix = contracts.load_support_matrix()
    assert support_matrix["format_version"] == "support-matrix-v1"
    assert contracts.support_tiers() == tuple(support_matrix["support_tiers"])
    assert {lane["lane_id"] for lane in contracts.published_basis_lanes()} == {
        "gpt2-causal-hf",
        "bert-mlm-hf",
        "mistral-7b-causal-hf",
        "ministral-3-3b-text-causal-hf",
        "ministral-3-8b-text-causal-hf",
        "ministral-3-14b-text-causal-hf",
        "tinyllama-1-1b-causal-hf",
        "olmo-2-7b-causal-hf",
        "olmo-2-13b-causal-hf",
        "olmoe-1b-7b-0924-causal-hf",
        "mixtral-8x7b-moe-causal-hf",
        "qwen3-30b-a3b-moe-causal-hf",
        "open-llama-7b-causal-hf",
        "falcon-7b-causal-hf",
        "qwen2-7b-causal-hf",
        "qwen2-5-7b-causal-hf",
        "qwen2-5-14b-causal-hf",
        "qwen3-causal-hf",
        "qwen3-5-causal-hf",
        "granite-4-1-3b-causal-hf",
        "granite-4-1-8b-causal-hf",
        "gemma4-e2b-text-causal-hf",
        "gemma4-12b-any-to-any-hf",
        "gemma4-26b-a4b-moe-image-text-hf",
        "deepseek-r1-distill-qwen-causal-hf",
        "deepseek-r1-0528-qwen3-8b-causal-hf",
        "deepseek-r1-distill-qwen-14b-causal-hf",
        "phi-4-text-causal-hf",
        "smollm3-3b-causal-hf",
        "phi-4-mini-causal-hf",
        "falcon-h1r-7b-causal-hf",
        "flan-t5-base-seq2seq-hf",
    }

    family_catalog = contracts.load_model_family_catalog()
    assert family_catalog["format_version"] == "model-family-catalog-v1"
    assert family_catalog["as_of"] == "2026-06-13"
    assert family_catalog["declared_support"][0]["display_name"] == "GPT-2 causal LM"
    declared = {item["display_name"] for item in family_catalog["declared_support"]}
    assert declared == {
        "GPT-2 causal LM",
        "BERT / RoBERTa MLM",
        "Mistral 7B causal LM",
        "Ministral 3 8B causal LM (text-only eval)",
        "Ministral 3 14B causal LM (text-only eval)",
        "Qwen2 7B causal LM",
        "Qwen2.5 7B causal LM",
        "Qwen2.5 14B causal LM",
        "Qwen3 causal LM",
        "DeepSeek-R1-Distill-Qwen causal LM",
        "Phi-4 causal LM (text-only eval)",
        "Gemma 4 E2B causal LM (text-only eval)",
        "TinyLlama 1.1B causal LM",
        "OLMo 2 7B causal LM",
        "OLMo 2 13B causal LM",
        "OLMoE 1B-active/7B-total causal LM",
        "OpenLLaMA 7B causal LM",
        "Falcon 7B causal LM",
        "Qwen3.5 causal LM",
        "Gemma 4 12B any-to-any LM",
        "Qwen3.5 4B image-text LM",
        "Qwen3.5 2B image-text LM",
        "Gemma 3n E4B image-text LM",
        "Gemma 3 4B IT image-text LM",
        "Ministral 3 3B causal LM (text-only eval)",
        "Granite 4.1 3B causal LM",
        "Granite 4.1 8B causal LM",
        "SmolLM3 3B causal LM",
        "Phi-4 mini causal LM",
        "DeepSeek-R1-Distill-Qwen 14B causal LM",
        "DeepSeek-R1-0528-Qwen3 8B causal LM",
        "Falcon-H1R 7B causal LM",
        "FLAN-T5 base seq2seq LM",
    }
    assert all(item["support_groups"] for item in family_catalog["declared_support"])
    implemented = {
        item["display_name"]: item for item in family_catalog["implemented_coverage"]
    }
    qwen3_moe = implemented["Qwen3 30B-A3B MoE causal LM"]
    assert qwen3_moe["state"] == "published_basis"
    assert any(
        evidence == "public_evidence/published_basis/qwen3_30b_a3b/evidence_pack"
        for evidence in qwen3_moe["repo_evidence"]
    )
    assert "not a benchmark-quality" in qwen3_moe["notes"]
    assert "four-GPU diagnostic lane failed" in qwen3_moe["notes"]
    gemma4_26b = implemented["Gemma 4 26B-A4B MoE image-text LM"]
    assert gemma4_26b["state"] == "published_basis"
    assert any(
        evidence == "public_evidence/published_basis/gemma4_26b_a4b/evidence_pack"
        for evidence in gemma4_26b["repo_evidence"]
    )

    usage_only = {item["display_name"] for item in family_catalog["usage_only"]}
    assert "QwQ 32B reasoning" not in usage_only
    assert "Qwen2.5 7B" not in usage_only
    assert "Qwen2.5 32B" in usage_only
    promotion = family_catalog["promotion_candidates_text_le_14b"]
    assert promotion["format_version"] == "promotion-candidates-text-le-14b-v1"
    candidates = {item["display_name"]: item for item in promotion["candidates"]}
    assert candidates["Qwen2.5 7B causal LM"]["decision"] == "promoted_published_basis"
    assert (
        candidates["Qwen2.5 7B causal LM"]["current_catalog_state"] == "published_basis"
    )
    assert (
        candidates["Qwen2.5 7B causal LM"]["criteria_status"][
            "approved_calibration_or_evaluation_evidence"
        ]
        == "pass"
    )
    assert candidates["Qwen2.5 14B causal LM"]["decision"] == "promoted_published_basis"
    assert (
        candidates["Qwen2.5 14B causal LM"]["current_catalog_state"]
        == "published_basis"
    )
    assert candidates["Qwen3 8B causal LM"]["decision"] == "promoted_published_basis"
    assert (
        candidates["Qwen3 8B causal LM"]["current_catalog_state"] == "published_basis"
    )
    assert (
        candidates["DeepSeek-R1-Distill-Qwen causal LM"]["decision"]
        == "promoted_published_basis"
    )
    assert (
        candidates["DeepSeek-R1-Distill-Qwen causal LM"]["current_catalog_state"]
        == "published_basis"
    )
    assert (
        candidates["Phi-4 reasoning-plus causal LM"]["decision"]
        == "promoted_published_basis"
    )
    assert (
        candidates["Phi-4 reasoning-plus causal LM"]["current_catalog_state"]
        == "published_basis"
    )
    assert (
        candidates["OpenLLaMA 7B causal LM"]["decision"] == "promoted_published_basis"
    )
    assert (
        candidates["OpenLLaMA 7B causal LM"]["current_catalog_state"]
        == "published_basis"
    )
    assert candidates["Falcon 7B causal LM"]["decision"] == "promoted_published_basis"
    assert (
        candidates["Falcon 7B causal LM"]["current_catalog_state"] == "published_basis"
    )
    for display_name in (
        "OPT 1.3B causal LM",
        "GLM 4 9B Chat",
    ):
        assert candidates[display_name]["criteria_status"]["included_preset"] == "pass"
        assert (
            candidates[display_name]["criteria_status"]["included_calibration_config"]
            == "pass"
        )
        assert (
            candidates[display_name]["criteria_status"]["cli_smoke_evidence"] == "pass"
        )
    assert candidates["Gemma 3 4B IT"]["decision"] == (
        "redirected_to_image_text_candidate"
    )
    assert (
        candidates["Gemma 3 4B IT"]["current_catalog_state"]
        == "community_experimental"
    )
    assert candidates["Gemma 3 4B IT"]["criteria_status"]["included_preset"] == "pass"
    assert (
        candidates["Gemma 3 4B IT"]["criteria_status"][
            "included_calibration_config"
        ]
        == "pass"
    )
    assert (
        candidates["Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA)"][
            "decision"
        ]
        == "blocked_missing_artifacts"
    )
    assert (
        candidates["Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA)"][
            "criteria_status"
        ]["included_preset"]
        == "pass"
    )
    assert (
        candidates["Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA)"][
            "criteria_status"
        ]["included_calibration_config"]
        == "pass"
    )
    assert (
        candidates["Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA)"][
            "criteria_status"
        ]["cli_smoke_evidence"]
        == "pass"
    )
    assert (
        candidates["OPT 1.3B causal LM"]["criteria_status"]["targeted_tests"] == "pass"
    )
    recommended = {
        item["display_name"] for item in family_catalog["recommended_additions"]
    }
    assert recommended == {"Audio-text evaluation pipeline"}

    gpt2_lane = contracts.support_lane_by_id("gpt2-causal-hf")
    assert gpt2_lane is not None
    assert gpt2_lane["support_tier"] == "published_basis"

    catalog = contracts.contract_catalog()
    assert catalog["support_matrix"]["format_version"] == "support-matrix-v1"
    assert (
        catalog["model_family_catalog"]["format_version"] == "model-family-catalog-v1"
    )
    assert catalog["plugin_compatibility"]["core_abi"] == "0.1"
    assert catalog["plugin_compatibility"]["match_policy"] == "exact_match"
    assert catalog["validation_keys"]["path"] == "contracts/validation_keys.json"
    assert catalog["validation_keys"]["kind"] == "array"
    assert catalog["validation_keys"]["item_count"] == len(
        contracts.load_json_contract("validation_keys.json")
    )
    assert catalog["console_labels"]["path"] == "contracts/console_labels.json"
    assert catalog["console_labels"]["kind"] == "array"
    assert catalog["console_labels"]["item_count"] == len(
        contracts.load_json_contract("console_labels.json")
    )
    assert catalog["metric_kinds"]["path"] == "contracts/metric_kinds.json"
    assert catalog["metric_kinds"]["kind"] == "array"
    assert catalog["metric_kinds"]["item_count"] == len(
        contracts.load_json_contract("metric_kinds.json")
    )
    assert (
        catalog["runtime_manifest"]["path"] == "contracts/runtime_manifest.schema.json"
    )
    assert catalog["verify_output"]["path"] == "contracts/verify_output.schema.json"
    assert catalog["policy_pack"]["path"] == "contracts/policy_pack.schema.json"

    schema = contracts.load_policy_pack_schema()
    assert schema["title"] == "InvarLock Policy Pack"
    assert (
        contracts.load_evidence_pack_manifest_schema()["title"]
        == "InvarLock Evidence Pack Manifest"
    )
    assert (
        contracts.load_runtime_manifest_schema()["title"]
        == "InvarLock Runtime Manifest"
    )
    verify_schema = contracts.load_verify_output_schema()
    assert verify_schema["title"] == "InvarLock Verify JSON Output"
    assert (
        verify_schema["properties"]["format_version"]["const"]
        == contracts.VERIFY_OUTPUT_FORMAT_VERSION
    )


def test_support_tier_descriptions_and_policy_pack_schema_are_in_sync() -> None:
    support_matrix = contracts.load_support_matrix()
    tiers = support_matrix["support_tiers"]
    descriptions = support_matrix["support_tier_descriptions"]

    assert set(descriptions) == set(tiers)
    assert all(descriptions[tier] for tier in tiers)

    policy_schema = contracts.load_policy_pack_schema()
    enum = policy_schema["properties"]["compatibility"]["properties"]["support_tiers"][
        "items"
    ]["enum"]
    assert enum == tiers

    for lane in support_matrix["lanes"]:
        assert lane["support_tier"] in tiers


def test_support_tiers_returns_empty_for_malformed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "_load_contract_or_raise",
        lambda filename: {"support_matrix.json": {"support_tiers": "bad"}}[filename],
    )

    assert contracts.support_tiers() == ()


def test_public_contract_paths_are_repo_relative() -> None:
    path = contracts.contract_path("support_matrix.json")
    assert path == contracts.CONTRACTS_ROOT / "support_matrix.json"
    assert path.is_file()
    assert (
        contracts.contract_relpath("support_matrix.json")
        == "contracts/support_matrix.json"
    )
    assert Path(
        contracts.contract_reference("support_matrix.json")["path"]
    ).as_posix() == ("contracts/support_matrix.json")


def test_support_matrix_published_basis_evidence_uses_public_evidence_paths() -> None:
    support_matrix = contracts.load_support_matrix()

    published_basis = [
        lane
        for lane in support_matrix["lanes"]
        if lane.get("support_tier") == "published_basis"
    ]
    assert published_basis

    for lane in published_basis:
        evidence = lane.get("evidence", {})
        assert evidence["evaluation_report_fixture"].startswith(
            "public_evidence/published_basis/"
        )
        assert evidence["runtime_manifest_fixture"].startswith(
            "public_evidence/published_basis/"
        )
        assert evidence["evidence_pack_recipe"].startswith(
            "public_evidence/published_basis/"
        )
        if "evidence_pack_fixture" in evidence:
            assert evidence["evidence_pack_fixture"].startswith(
                "public_evidence/published_basis/"
            )
        if "artifact_package" in evidence:
            assert evidence["artifact_package"].startswith(
                "public_evidence/published_basis/"
            )
            assert (REPO_ROOT / evidence["artifact_package"]).is_dir()
        if "guard_value_demo" in evidence:
            assert evidence["guard_value_demo"].startswith(
                "public_evidence/published_basis/"
            )
            assert (REPO_ROOT / evidence["guard_value_demo"]).is_dir()
        assert "tests/fixtures/" not in evidence["evaluation_report_fixture"]
        assert "tests/fixtures/" not in evidence["runtime_manifest_fixture"]
        assert "tests/fixtures/" not in evidence["evidence_pack_recipe"]


def test_readme_surfaces_public_contract_catalog_entries() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "doctor --json" in readme
    assert "advanced plugins ... --json" in readme
    assert "`validation_keys`, `console_labels`, and `metric_kinds`" in readme


def test_contract_reference_records_scalar_payload_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    (contracts_dir / "scalar.json").write_text("42\n", encoding="utf-8")

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", contracts_dir)
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", contracts_dir)

    assert contracts.contract_reference("scalar.json") == {
        "path": "contracts/scalar.json",
        "kind": "int",
    }


def test_public_contract_loader_falls_back_to_packaged_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")

    payload = contracts.load_support_matrix()
    assert payload["format_version"] == "support-matrix-v1"
    assert payload["lanes"]


def test_public_contract_loader_falls_back_to_workspace_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.chdir(workspace)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"


def test_public_contract_loader_tries_env_then_workspace_and_deduplicates(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setenv("INVARLOCK_CONTRACTS_ROOT", str(tmp_path / "env-contracts"))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.chdir(workspace)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"

    roots = contracts._fallback_contract_roots()
    assert roots == [tmp_path / "env-contracts", contracts_dir]


def test_public_contract_loader_tries_pyinstaller_bundle_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    bundle_root = tmp_path / "_MEI12345"
    bundle_contracts = bundle_root / "contracts"
    bundle_contracts.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (bundle_contracts / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.delenv("INVARLOCK_CONTRACTS_ROOT", raising=False)
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"
    assert contracts._fallback_contract_roots()[:2] == [
        bundle_contracts,
        bundle_root / "invarlock" / "_data" / "contracts",
    ]


def test_public_contract_loader_discovers_ancestor_contracts_for_build_out(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    build_out = workspace / "build-out" / "python" / "invarlock"
    build_out.mkdir(parents=True)
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "__file__", str(build_out / "public_contracts.py"))
    monkeypatch.setattr(sys, "argv", [])
    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.chdir(build_out)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"
    assert contracts._ancestor_contract_roots(filename="policy_pack.schema.json") == [
        contracts_dir
    ]


def test_public_contract_loader_discovers_contracts_from_executable_ancestor(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    build_out = workspace / "build-out"
    build_out.mkdir(parents=True)
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(
        contracts,
        "__file__",
        str(tmp_path / "bundle" / "public_contracts.py"),
    )
    monkeypatch.setattr(sys, "argv", [str(build_out / "bundled_policy_tool")])
    monkeypatch.setattr(sys, "executable", str(build_out / "bundled_policy_tool.pkg"))
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"
    assert contracts._ancestor_contract_roots(filename="policy_pack.schema.json") == [
        contracts_dir
    ]


def test_public_contract_loader_handles_missing_process_paths(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    build_out = workspace / "build-out" / "python" / "invarlock"
    build_out.mkdir(parents=True)
    contracts_dir = workspace / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "__file__", str(build_out / "public_contracts.py"))
    monkeypatch.setattr(sys, "argv", [])
    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.chdir(build_out)

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"
    assert contracts._ancestor_contract_roots(filename="policy_pack.schema.json") == [
        contracts_dir
    ]


def test_public_contract_loader_skips_missing_ancestor_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    contracts_dir = tmp_path / "workspace" / "contracts"
    contracts_dir.mkdir(parents=True)
    source = contracts.CONTRACTS_ROOT / "policy_pack.schema.json"
    (contracts_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "_fallback_contract_roots", lambda: [])
    monkeypatch.setattr(
        contracts,
        "_ancestor_contract_roots",
        lambda *, filename: [tmp_path / "missing-ancestor", contracts_dir],
    )

    payload = contracts.load_policy_pack_schema()
    assert payload["title"] == "InvarLock Policy Pack"


def test_public_contract_loader_raises_when_all_roots_are_missing(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(contracts, "CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(contracts, "PACKAGE_CONTRACTS_ROOT", tmp_path / "missing")
    monkeypatch.setattr(
        contracts, "__file__", str(tmp_path / "sandbox" / "public_contracts.py")
    )
    monkeypatch.setattr(sys, "argv", [])
    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setenv("INVARLOCK_CONTRACTS_ROOT", str(tmp_path / "env-contracts"))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.chdir(workspace)

    with pytest.raises(FileNotFoundError, match="policy_pack.schema.json"):
        contracts.load_json_contract("policy_pack.schema.json")


def test_packaged_contract_copies_match_repo_contracts() -> None:
    repo_contracts = sorted(contracts.CONTRACTS_ROOT.glob("*.json"))
    assert repo_contracts

    for repo_path in repo_contracts:
        packaged = contracts.PACKAGE_CONTRACTS_ROOT.joinpath(repo_path.name)
        assert packaged.is_file(), repo_path.name
        assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
            repo_path.read_text(encoding="utf-8")
        )


def test_public_contract_helpers_raise_when_contracts_are_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(OSError("missing")),
    )

    with pytest.raises(contracts.ContractLoadError, match="support_matrix.json"):
        contracts.load_support_matrix()
    with pytest.raises(contracts.ContractLoadError, match="adapter_capabilities.json"):
        contracts.load_adapter_capabilities()
    with pytest.raises(contracts.ContractLoadError, match="model_family_catalog.json"):
        contracts.load_model_family_catalog()
    with pytest.raises(contracts.ContractLoadError, match="plugin_compatibility.json"):
        contracts.load_plugin_compatibility()
    with pytest.raises(contracts.ContractLoadError, match="policy_pack.schema.json"):
        contracts.load_policy_pack_schema()
    with pytest.raises(
        contracts.ContractLoadError, match="evidence_pack_manifest.schema.json"
    ):
        contracts.load_evidence_pack_manifest_schema()
    with pytest.raises(
        contracts.ContractLoadError, match="runtime_manifest.schema.json"
    ):
        contracts.load_runtime_manifest_schema()
    with pytest.raises(contracts.ContractLoadError, match="verify_output.schema.json"):
        contracts.load_verify_output_schema()
    assert contracts.contract_reference("support_matrix.json") == {
        "path": "contracts/support_matrix.json",
        "load_error": "missing",
    }


def test_public_contract_helpers_wrap_unicode_decode_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        contracts,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        ),
    )

    with pytest.raises(contracts.ContractLoadError, match="support_matrix.json"):
        contracts.load_support_matrix()


def test_public_contract_helpers_reject_non_mapping_payloads(monkeypatch) -> None:
    payloads = {
        "support_matrix.json": ["unexpected"],
        "model_family_catalog.json": "unexpected",
        "adapter_capabilities.json": "unexpected",
        "plugin_compatibility.json": ["unexpected"],
        "runtime_manifest.schema.json": ["unexpected"],
        "verify_output.schema.json": ["unexpected"],
        "policy_pack.schema.json": ["unexpected"],
        "evidence_pack_manifest.schema.json": ["unexpected"],
    }
    monkeypatch.setattr(
        contracts,
        "_load_contract_or_raise",
        lambda filename: payloads[filename],
    )

    with pytest.raises(contracts.ContractLoadError, match="support_matrix.json"):
        contracts.load_support_matrix()
    with pytest.raises(contracts.ContractLoadError, match="adapter_capabilities.json"):
        contracts.load_adapter_capabilities()
    with pytest.raises(contracts.ContractLoadError, match="model_family_catalog.json"):
        contracts.load_model_family_catalog()
    with pytest.raises(contracts.ContractLoadError, match="plugin_compatibility.json"):
        contracts.load_plugin_compatibility()
    with pytest.raises(contracts.ContractLoadError, match="policy_pack.schema.json"):
        contracts.load_policy_pack_schema()
    with pytest.raises(
        contracts.ContractLoadError, match="evidence_pack_manifest.schema.json"
    ):
        contracts.load_evidence_pack_manifest_schema()
    with pytest.raises(
        contracts.ContractLoadError, match="runtime_manifest.schema.json"
    ):
        contracts.load_runtime_manifest_schema()
    with pytest.raises(contracts.ContractLoadError, match="verify_output.schema.json"):
        contracts.load_verify_output_schema()


def test_public_contract_lane_and_adapter_helpers_cover_non_matching_entries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contracts,
        "_load_contract_or_raise",
        lambda filename: {
            "support_matrix.json": {
                "format_version": "support-matrix-v1",
                "lanes": [
                    {"lane_id": "first", "support_tier": "community_experimental"},
                    {"lane_id": "second", "support_tier": "published_basis"},
                    "bad",
                ],
            },
            "adapter_capabilities.json": {
                "format_version": "adapter-capabilities-v1",
                "adapters": [
                    {"adapter": "", "guard_coverage": "none"},
                    {"adapter": "good", "guard_coverage": "full"},
                    "bad",
                ],
            },
            "model_family_catalog.json": {
                "format_version": "model-family-catalog-v1",
                "declared_support": [
                    {"family_id": "gpt2-causal-lm", "display_name": "GPT-2 causal LM"},
                    "bad",
                ],
                "implemented_coverage": [],
                "usage_only": [],
                "promotion_candidates_text_le_14b": {"candidates": []},
                "recommended_additions": [],
            },
            "plugin_compatibility.json": {
                "format_version": "plugin-compatibility-v1",
                "format": "compatibility-doc",
                "core_abi": "0.1",
                "match_policy": "exact_match",
            },
        }[filename],
    )

    assert contracts.support_lane_by_id("missing") is None
    assert contracts.support_lane_by_id("second") == {
        "lane_id": "second",
        "support_tier": "published_basis",
    }
    assert contracts.adapter_capability_map() == {
        "good": {"adapter": "good", "guard_coverage": "full"}
    }
    assert contracts.load_model_family_catalog()["declared_support"][0] == {
        "family_id": "gpt2-causal-lm",
        "display_name": "GPT-2 causal LM",
    }
    assert contracts.contract_reference("plugin_compatibility.json") == {
        "path": "contracts/plugin_compatibility.json",
        "format_version": "plugin-compatibility-v1",
        "format": "compatibility-doc",
        "core_abi": "0.1",
        "match_policy": "exact_match",
    }
    assert contracts.contract_reference("validation_keys.json") == {
        "path": "contracts/validation_keys.json",
        "kind": "array",
        "item_count": len(contracts.load_json_contract("validation_keys.json")),
    }
