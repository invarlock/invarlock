from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import invarlock.evidence_pack as evidence_pack_mod
import invarlock.evidence_pack_deployable_validation as deployable_mod
from invarlock.evidence_pack import (
    EvidencePackStatus,
    validate_manifest,
    verify_evidence_pack,
    verify_manifest_provenance,
)
from invarlock.evidence_pack_contracts.deployable_coverage import (
    canonical_names_sha256,
)
from tests.reporting._support_evidence_pack_paths import (
    _build_pack,
    _write_json,
)


def _allow_unsigned_pack(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_pack_mod,
        "_verify_signature",
        lambda pack_dir, strict: ([], [], None),
        raising=True,
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )


def test_evidence_pack_manifest_and_provenance_round_trip(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    assert validate_manifest(pack_dir / "manifest.json") == []
    assert verify_manifest_provenance(pack_dir) == []

    result = verify_evidence_pack(pack_dir, skip_verify=True)
    payload = result.payload
    exit_code = result.status
    assert exit_code == EvidencePackStatus.SIGNATURE
    assert payload["ok"] is False
    assert payload["errors"] == [
        "manifest.signature.json missing; signed manifest required by default."
    ]


def test_evidence_pack_verify_rejects_json_out_inside_pack(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = verify_evidence_pack(
        pack_dir, json_out_path=pack_dir / "verify.json", skip_verify=True
    )
    payload = result.payload
    exit_code = result.status

    assert exit_code == EvidencePackStatus.USAGE
    assert payload["ok"] is False
    assert "--json-out must point outside the pack directory." in payload["errors"]


def test_evidence_pack_verify_strict_rejects_extra_files_without_bypass(
    tmp_path: Path,
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    (pack_dir / "extra.txt").write_text("extra", encoding="utf-8")
    original_verify_signature = evidence_pack_mod._verify_signature
    evidence_pack_mod._verify_signature = lambda pack_dir, strict: ([], [], None)

    try:
        result = verify_evidence_pack(pack_dir, skip_verify=False, strict=True)
    finally:
        evidence_pack_mod._verify_signature = original_verify_signature

    payload = result.payload
    exit_code = result.status
    assert exit_code == EvidencePackStatus.INTEGRITY
    assert payload["ok"] is False
    assert any("extra files not covered" in error for error in payload["errors"])


def test_evidence_pack_verify_requires_clean_reports(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/errors/noop/evaluation.report.json",
        scenario_strictness="must_fail",
    )
    monkeypatch.setattr(
        evidence_pack_mod,
        "unverified_provenance_allowed",
        lambda: True,
        raising=True,
    )

    result = verify_evidence_pack(pack_dir)
    payload = result.payload
    exit_code = result.status

    assert exit_code == EvidencePackStatus.REPORTS
    assert payload["ok"] is False
    assert any("No reports expected to pass" in error for error in payload["errors"])


def test_evidence_pack_verify_requires_validation_edit_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/quant_4bit_clean/run_1/evaluation.report.json",
        scenario_metadata={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {
                "kind": "edit",
                "edit_spec": "quant_rtn:clean",
                "version": "clean",
            },
        },
    )
    _allow_unsigned_pack(monkeypatch)

    result = verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == EvidencePackStatus.INTEGRITY
    assert any(
        "quant_4bit_clean: edit_metadata.json missing next to report" in error
        for error in result.payload["errors"]
    )


def test_evidence_pack_verify_requires_deployable_sidecars(
    monkeypatch, tmp_path: Path
) -> None:
    report_rel = "reports/model/deploy_bnb_8bit_clean/run_1/evaluation.report.json"
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path=report_rel,
        scenario_metadata={
            "artifact_class": "deployable_optimized_subject",
            "optimized_deployment_backend": True,
            "generation": {
                "kind": "deployable_edit",
                "backend": "bitsandbytes",
                "edit_spec": "bnb_8bit:8:all",
                "version": "deployable",
            },
        },
        report_sidecars={
            "edit_metadata.json": {
                "schema": "invarlock/evidence-pack-edit-metadata-v1",
                "artifact_class": "deployable_optimized_subject",
                "edit_type": "bnb_8bit",
                "optimized_deployment_backend": True,
                "packed_quantized_storage": True,
                "coverage": {},
            }
        },
    )
    _allow_unsigned_pack(monkeypatch)

    result = verify_evidence_pack(pack_dir, skip_verify=True)

    assert result.status == EvidencePackStatus.INTEGRITY
    assert any(
        "deploy_bnb_8bit_clean: deployable sidecar missing" in error
        for error in result.payload["errors"]
    )


def test_deployable_pack_binding_rejects_subject_backend_and_ledger_tampering(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    module_names = ["layer.0", "layer.1"]
    weight_names = [f"{name}.weight" for name in module_names]
    logical_coverage = {
        "basis": "dense_baseline_unique_parameters",
        "weight_tensor_names": weight_names,
        "weight_tensor_names_sha256": canonical_names_sha256(weight_names),
        "weight_tensor_count": 2,
        "parameter_elements": 8,
        "total_unique_parameter_elements": 10,
    }
    packed_facts = {
        "quantized_module_count": 2,
        "quantized_module_names": module_names,
        "quantized_module_names_sha256": canonical_names_sha256(module_names),
        "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
        "packed_weight_storage_elements": 4,
        "logical_coverage": logical_coverage,
    }
    sidecars: dict[str, dict[str, object]] = {}
    for name in (
        "backend_inventory.json",
        "memory_report.json",
        "load_smoke.json",
        "inference_smoke.json",
    ):
        payload: dict[str, object] = {
            "artifact_identity": identity,
            "baseline_identity": baseline_identity,
            "bits": 8,
        }
        if name == "backend_inventory.json":
            payload["backend"] = "bitsandbytes"
            payload["quantization_config"] = {
                "load_in_8bit": True,
                "load_in_4bit": False,
            }
            payload.update(packed_facts)
        elif name == "load_smoke.json":
            payload.update(packed_facts)
        elif name == "inference_smoke.json":
            payload["logits_sha256"] = "sha256:" + "c" * 64
            payload["logits_shape"] = [1, 2, 3]
            payload["all_logits_finite"] = True
        elif name == "memory_report.json":
            payload["baseline_reported_bytes"] = 200
            payload["quantized_reported_bytes"] = 100
            payload["reduction_bytes"] = 100
            payload["reduction_ratio"] = 0.5
            payload["runtime_memory_reduction_observed"] = True
        _write_json(report_dir / name, payload)
        sidecars[name] = payload
    ledger = {
        name: "sha256:" + hashlib.sha256((report_dir / name).read_bytes()).hexdigest()
        for name in (
            "backend_inventory.json",
            "memory_report.json",
            "load_smoke.json",
            "inference_smoke.json",
        )
    }
    validation: dict[str, object] = {
        "artifact_identity": identity,
        "baseline_identity": baseline_identity,
        "backend": "bitsandbytes",
        "bits": 8,
        "validation_scope": "structural_only",
        "runtime_proof_authoritative": False,
        "sidecar_digests": ledger,
    }
    runtime_validation = dict(validation)
    runtime_validation["validation_scope"] = "runtime_reproof"
    runtime_validation["runtime_proof_authoritative"] = True
    runtime_validation["runtime_proof"] = {
        "artifact_identity": identity,
        "baseline_identity": baseline_identity,
        **packed_facts,
        "logits_sha256": "sha256:" + "c" * 64,
        "logits_shape": [1, 2, 3],
        "all_logits_finite": True,
        "baseline_reported_bytes": 200,
        "quantized_reported_bytes": 100,
        "reduction_bytes": 100,
        "reduction_ratio": 0.5,
        "runtime_memory_reduction_observed": True,
    }
    _write_json(report_dir / "deployable_artifact_validation.json", validation)
    publication: dict[str, object] = {
        "artifact_identity": identity,
        "baseline_identity": baseline_identity,
        "bits": 8,
        "validation_scope": "structural_only",
        "runtime_proof_authoritative": False,
        "sidecar_digests": ledger,
        "proof_validation_sha256": "sha256:"
        + hashlib.sha256(
            (report_dir / "deployable_artifact_validation.json").read_bytes()
        ).hexdigest(),
    }
    sidecars.update(
        {
            "deployable_artifact_validation.json": validation,
            "runtime_deployability_validation.json": runtime_validation,
            "publication_commit.json": publication,
        }
    )
    spec = {
        "generation": {
            "kind": "deployable_edit",
            "backend": "bitsandbytes",
            "edit_spec": "bnb_8bit:8:all",
        }
    }
    metadata = {
        "backend": "bitsandbytes",
        "edit_type": "bnb_8bit",
        "logical_coverage": logical_coverage,
        "coverage": {
            "edited_tensors": 2,
            "edited_params": 8,
            "total_params": 10,
            "coverage_ratio": 0.8,
        },
    }
    report = {
        "meta": {"model_identity": identity},
        "baseline_ref": {"model_identity": baseline_identity},
    }

    assert (
        deployable_mod._deployable_binding_errors(
            scenario_id="quant_8bit_deployable",
            spec=spec,
            report=report,
            metadata=metadata,
            report_dir=report_dir,
            sidecars=sidecars,
        )
        == []
    )

    adversarial_cases = (
        (
            lambda s, _spec, _report, _metadata: s["publication_commit.json"].update(
                artifact_identity={
                    "kind": "local_checkpoint_tree",
                    "sha256": "sha256:" + "0" * 64,
                }
            ),
            "publication artifact identity mismatch",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "deployable_artifact_validation.json"
            ].update(
                artifact_identity={
                    "kind": "local_checkpoint_tree",
                    "sha256": "sha256:" + "0" * 64,
                }
            ),
            "generated and runtime deployable artifact identities disagree",
        ),
        (
            lambda _s, spec, _report, _metadata: spec.update(generation={}),
            "scenario backend missing",
        ),
        (
            lambda _s, _spec, _report, metadata: metadata.update(edit_type="bnb_4bit"),
            "edit type does not match scenario",
        ),
        (
            lambda _s, spec, _report, metadata: (
                spec["generation"].update(edit_spec="unsupported"),
                metadata.update(edit_type="unsupported"),
            ),
            "edit type has no supported bitwidth",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "runtime_deployability_validation.json"
            ]["runtime_proof"].update(quantized_module_types=["different"]),
            "runtime module types disagree",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "runtime_deployability_validation.json"
            ]["runtime_proof"].update(quantized_module_count=1),
            "quantized module count does not match module names",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "runtime_deployability_validation.json"
            ]["runtime_proof"].update(packed_weight_storage_elements=99),
            "packed_weight_storage_elements disagrees",
        ),
        (
            lambda s, _spec, _report, _metadata: s["backend_inventory.json"].update(
                quantized_module_names_sha256="sha256:" + "0" * 64
            ),
            "quantized_module_names_sha256 disagrees",
        ),
        (
            lambda _s, _spec, _report, metadata: metadata["coverage"].update(
                total_params=11, coverage_ratio=8 / 11
            ),
            "metadata coverage is not canonical",
        ),
        (
            lambda _s, _spec, _report, metadata: metadata["coverage"].update(
                coverage_ratio=0.5
            ),
            "metadata coverage is not canonical",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "runtime_deployability_validation.json"
            ]["runtime_proof"].update(logits_sha256="sha256:" + "0" * 64),
            "runtime inference disagrees on logits_sha256",
        ),
        (
            lambda s, _spec, _report, _metadata: s[
                "runtime_deployability_validation.json"
            ]["runtime_proof"].update(reduction_ratio=0.25),
            "runtime memory disagrees on reduction_ratio",
        ),
    )
    for mutate, fragment in adversarial_cases:
        case_sidecars = deepcopy(sidecars)
        case_spec = deepcopy(spec)
        case_report = deepcopy(report)
        case_metadata = deepcopy(metadata)
        mutate(case_sidecars, case_spec, case_report, case_metadata)
        case_errors = deployable_mod._deployable_binding_errors(
            scenario_id="quant_8bit_deployable",
            spec=case_spec,
            report=case_report,
            metadata=case_metadata,
            report_dir=report_dir,
            sidecars=case_sidecars,
        )
        assert any(fragment in error for error in case_errors), (fragment, case_errors)

    report["meta"]["model_identity"] = {
        **identity,
        "sha256": "sha256:" + "c" * 64,
    }
    sidecars["backend_inventory.json"]["backend"] = "fabricated"
    sidecars["inference_smoke.json"]["bits"] = 4
    sidecars["runtime_deployability_validation.json"]["sidecar_digests"] = {}
    sidecars["runtime_deployability_validation.json"]["runtime_proof"][
        "quantized_module_count"
    ] = 99
    sidecars["memory_report.json"]["baseline_identity"] = {
        **baseline_identity,
        "sha256": "sha256:" + "d" * 64,
    }
    errors = deployable_mod._deployable_binding_errors(
        scenario_id="quant_8bit_deployable",
        spec=spec,
        report=report,
        metadata=metadata,
        report_dir=report_dir,
        sidecars=sidecars,
    )
    assert any("evaluation subject identity" in error for error in errors)
    assert any("inventory backend mismatch" in error for error in errors)
    assert any("bitwidth mismatch" in error for error in errors)
    assert any("runtime validation sidecar digest ledger" in error for error in errors)
    assert any("runtime module count" in error for error in errors)
    assert any("baseline identity mismatch" in error for error in errors)
