from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.evidence_packs.python.editing.training_contract import (
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_profile_snapshot import (
    produce_training_profile_snapshot,
)
from tests.cli._support_verify_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
    _write_runtime_manifest,
)
from tests.evidence_packs._support_training_evidence_proof import _proof_for
from tests.evidence_packs._support_training_receipt import (
    valid_training_receipt as _valid_receipt,
)


def valid_hqq_runtime_quantization_proof() -> dict[str, object]:
    """Return the strict runtime proof fixture shared by source-matrix tests."""
    return {
        "schema": "invarlock/runtime-quantization-proof-v1",
        "proof_kind": "live_loaded_model_runtime_type_inventory",
        "adapter": "hf_hqq",
        "backend": "hqq",
        "backend_version": "3.0.0",
        "ok": True,
        "status": "verified_live_runtime_types",
        "reason": "recognized_live_quantized_runtime_types",
        "live_model_observed": True,
        "module_inventory_observed": True,
        "recognized_quantized_runtime_type_count": 1,
        "recognized_quantized_runtime_types": ["hqq.core.quantize.HQQLinear"],
        "recognized_quantized_runtime_observation_kinds": ["module"],
        "live_model_quantization_method": None,
        "backend_runtime_importable": None,
        "backend_runtime_import_error_type": None,
        "backend_runtime_version": None,
        "backend_runtime_compatibility_bridge_required": None,
        "backend_runtime_compatibility_bridge_applied": None,
        "backend_runtime_compatibility_bridge_error_type": None,
        "packed_storage_artifact_proof_required": False,
        "artifact_binding": "not_attempted",
    }


def valid_hqq_backend_inventory() -> dict[str, object]:
    """Return the strict backend inventory paired with the HQQ proof fixture."""
    return {
        "schema": "invarlock/backend-inventory-v1",
        "adapter": "hf_hqq",
        "backend": "hqq",
        "backend_version": "3.0.0",
        "transformers_version": "5.12.0",
        "quantization_config": {},
        "quantized_module_count": 1,
        "quantized_module_types": ["hqq.core.quantize.HQQLinear"],
        "quantized_observation_kinds": ["module"],
        "device_map": "cuda",
        "memory_footprint": {"reported_bytes": 1, "method": "test"},
        "load_smoke": True,
        "inference_smoke": True,
    }


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    REPO_ROOT
    / "examples"
    / "integrations"
    / "_shared"
    / "validate_source_matrix_artifacts.py"
)


def _load_validator():
    module_name = "source_matrix_artifact_validator"
    spec = importlib.util.spec_from_file_location(module_name, VALIDATOR)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_matrix_artifact_set(report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True)
    report_payload = _strict_provenance_gate_cert()
    report_path = report_dir / "evaluation.report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    _write_runtime_manifest(report_path)
    baseline_path = report_dir.parent / "acceptance-baseline.json"
    baseline_path.write_text(
        json.dumps(_matching_strict_ppl_baseline(report_payload)),
        encoding="utf-8",
    )
    policy_path = report_dir.parent / "acceptance-policy-pack.json"
    policy_path.write_text(
        json.dumps(_matching_strict_policy_pack(report_payload)),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "verify",
            str(report_path),
            "--json",
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--runtime-provenance",
            "container",
            "--baseline",
            str(baseline_path),
            "--policy-pack",
            str(policy_path),
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (report_dir / "verify.json").write_text(completed.stdout, encoding="utf-8")
    (report_dir / "evaluation.html").write_text("<html></html>\n", encoding="utf-8")
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(valid_hqq_backend_inventory()) + "\n",
        encoding="utf-8",
    )
    (report_dir / "runtime_quantization_proof.json").write_text(
        json.dumps(valid_hqq_runtime_quantization_proof(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_dir / "lane_artifact.json").write_text(
        json.dumps(
            {
                "lane_artifact_label": "cuda-container-strict",
                "lane": "cuda",
                "execution_mode": "container",
                "assurance": "strict",
                "runtime_provenance": "container",
                "device": "cuda",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_command.txt").write_text(
        "wrapper: run_tiny_hf_hqq.sh --lane cuda "
        "--require-backend-inventory --require-runtime-quantization-proof\n"
        "evaluate: invarlock evaluate baseline subject\n",
        encoding="utf-8",
    )
    (report_dir / "run_summary.txt").write_text(
        "status: success\n"
        "lane_artifact_label: cuda-container-strict\n"
        "verify_status: ok\n"
        "verify_runtime_provenance_declared: container\n"
        "verify_runtime_provenance_verified: true\n",
        encoding="utf-8",
    )
    (report_dir / "checkpoint_refs.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "adapter_runtime_summary.json").write_text("{}\n", encoding="utf-8")
    (report_dir / "fixture_summary.json").write_text("{}\n", encoding="utf-8")
    return baseline_path, policy_path


def _acceptance_inputs(validator, baseline_path: Path, policy_path: Path):
    return validator.AcceptanceInputs(
        baseline_report=baseline_path,
        policy_pack=policy_path,
        expected_runtime_image_digest=_VALID_TEST_IMAGE_DIGEST,
        python_bin=sys.executable,
    )


def _write_test_source_matrix(repo_root: Path) -> Path:
    matrix_path = repo_root / "examples" / "integrations" / "source_matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "invarlock.integration_source_matrix.v1",
                "description": "Strict test source matrix.",
                "entries": [
                    {
                        "target": "hqq",
                        "readme": "examples/integrations/hqq/README.md",
                        "runner": "examples/integrations/hqq/run_tiny_hf_hqq.sh",
                        "status_label": "runnable",
                        "verification_profile": "ci",
                        "strict_claim_phrase": "`cuda-container-strict` result requires",
                        "subject_form": "runtime-adapter",
                        "report_path": "reports/tiny-hf-hqq/<artifact-lane>",
                        "subject_adapter": "hf_hqq",
                        "lane": "cuda-container-strict",
                        "command_shape": "--lane cuda",
                        "runtime_image": {
                            "family": "cuda-hqq",
                            "source_command": (
                                "examples/integrations/_runtime_images/"
                                "build_example_runtime_image.sh cuda-hqq"
                            ),
                            "declared_digest_source": "runtime.manifest.json",
                            "expected_digest_source": (
                                "wrapper_input_from_independent_policy"
                            ),
                        },
                        "expected": {
                            "lane_artifact_label": "cuda-container-strict",
                            "verify_status": "ok",
                            "runtime_provenance_declared": "container",
                            "runtime_provenance_verified": True,
                            "runtime_provenance_status": (
                                "expected_image_digest_matched"
                            ),
                            "runtime_expected_digest_matched": True,
                        },
                        "required_artifacts": [
                            "evaluation.report.json",
                            "verify.json",
                            "runtime.manifest.json",
                            "evaluation.html",
                            "backend_inventory.json",
                            "runtime_quantization_proof.json",
                            "lane_artifact.json",
                            "run_command.txt",
                            "run_summary.txt",
                            "checkpoint_refs.json",
                            "adapter_runtime_summary.json",
                            "fixture_summary.json",
                        ],
                        "provenance_artifacts": [
                            "checkpoint_refs.json",
                            "adapter_runtime_summary.json",
                            "fixture_summary.json",
                        ],
                        "runner_enforcement": {
                            "backend_inventory": "--require-backend-inventory",
                            "runtime_quantization_proof": (
                                "--require-runtime-quantization-proof"
                            ),
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix_path


def _report_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "examples"
        / "integrations"
        / "hqq"
        / "reports"
        / "tiny-hf-hqq"
        / "cuda-container-strict"
    )


def _write_training_binding_set(
    report_dir: Path,
    *,
    profile_id: str = "tiny_gpt2_lora_v1",
) -> tuple[dict, dict]:
    report_dir.mkdir()
    profile = load_training_profile(profile_id)
    receipt = _valid_receipt(profile)
    receipt_path = report_dir / "training_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    report_path = report_dir / "evaluation.report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    verify_path = report_dir / "verify.json"
    verify_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "schema": "invarlock.integration_training_binding.v1",
        "verified": True,
        "receipt_sha256": receipt["receipt_sha256"],
        "subject_tree_sha256": receipt["hashes"]["subject_tree_sha256"],
        "training_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "evaluation_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "verify_artifact_sha256": hashlib.sha256(verify_path.read_bytes()).hexdigest(),
    }
    (report_dir / "training_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    return receipt, binding


def _write_training_evidence_set(
    report_dir: Path,
    *,
    profile_id: str = "tiny_gpt2_lora_v1",
    scope: str = "attn",
) -> tuple[dict, dict]:
    report_dir.mkdir()
    profile = load_training_profile(profile_id)
    receipt = _valid_receipt(profile)
    proof, baseline_identity, artifact_identity = _proof_for(receipt)
    receipt_path = report_dir / "training_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (report_dir / "training_evidence_proof.json").write_text(
        json.dumps(proof), encoding="utf-8"
    )
    (report_dir / "evaluation.report.json").write_text(
        json.dumps(
            {
                "meta": {"model_identity": artifact_identity},
                "baseline_ref": {"model_identity": baseline_identity},
            }
        ),
        encoding="utf-8",
    )
    verify_path = report_dir / "verify.json"
    verify_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "schema": "invarlock.integration_training_binding.v1",
        "verified": True,
        "receipt_sha256": receipt["receipt_sha256"],
        "subject_tree_sha256": receipt["hashes"]["subject_tree_sha256"],
        "training_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "evaluation_report_sha256": hashlib.sha256(
            (report_dir / "evaluation.report.json").read_bytes()
        ).hexdigest(),
        "verify_artifact_sha256": hashlib.sha256(verify_path.read_bytes()).hexdigest(),
    }
    (report_dir / "training_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    produce_training_profile_snapshot(
        profile_id=profile_id,
        scope=scope,
        output_path=report_dir / "training_profile_snapshot.json",
        repo_root=REPO_ROOT,
    )
    return receipt, proof
