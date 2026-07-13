from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.scripts._support_source_matrix_artifact_validator import (
    _acceptance_inputs,
    _load_validator,
    _report_dir,
    _write_matrix_artifact_set,
    _write_test_source_matrix,
)


def test_source_matrix_requires_strict_runtime_quantization_proof(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(report_dir)
    (report_dir / "runtime_quantization_proof.json").unlink()

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )
    assert any(
        issue.path.endswith("runtime_quantization_proof.json")
        and issue.message == "required artifact is missing"
        for issue in issues
    )

    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    entry = matrix_payload["entries"][0]
    entry["required_artifacts"].remove("runtime_quantization_proof.json")
    matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")
    issues = validator.validate_entry(
        tmp_path,
        entry,
        acceptance_inputs=None,
    )
    assert any(
        "strict module-backed quantized source matrix requires "
        "runtime_quantization_proof.json" in issue.message
        for issue in issues
    )


def test_source_matrix_requires_quantized_runner_enforcement(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]
    entry["runner_enforcement"] = {"backend_inventory": "--wrong-flag"}

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        "must bind backend_inventory to --require-backend-inventory" in issue.message
        for issue in issues
    )
    assert any(
        "must bind runtime_quantization_proof to "
        "--require-runtime-quantization-proof" in issue.message
        for issue in issues
    )

    (report_dir / "run_command.txt").write_text(
        "wrapper: run_tiny_hf_hqq.sh --require-backend-inventory\n"
        "evaluate: invarlock evaluate baseline subject\n",
        encoding="utf-8",
    )
    entry["runner_enforcement"] = {
        "backend_inventory": "--require-backend-inventory",
        "runtime_quantization_proof": "--require-runtime-quantization-proof",
    }
    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)
    assert any(
        "run command is missing --require-runtime-quantization-proof" in issue.message
        for issue in issues
    )


def test_source_matrix_rejects_duplicate_runtime_quantization_proof_keys(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    proof_path = report_dir / "runtime_quantization_proof.json"
    proof_text = proof_path.read_text(encoding="utf-8")
    proof_path.write_text(
        proof_text.replace('"ok": true', '"ok": false, "ok": true'),
        encoding="utf-8",
    )
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        "runtime quantization proof is not strict JSON" in issue.message
        and "duplicate JSON key 'ok'" in issue.message
        for issue in issues
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "backend_inventory.json",
        "runtime.manifest.json",
        "verify.json",
        "lane_artifact.json",
        "checkpoint_refs.json",
    ],
)
def test_source_matrix_rejects_ambiguous_required_json_by_family(
    tmp_path: Path, artifact_name: str
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    (report_dir / artifact_name).write_text(
        '{"authority": false, "authority": true}\n', encoding="utf-8"
    )
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        artifact_name in issue.path
        and "duplicate JSON key 'authority'" in issue.message
        for issue in issues
    )


def test_source_matrix_rejects_nonfinite_and_nonobject_required_json(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    (report_dir / "checkpoint_refs.json").write_text(
        '{"score": NaN}\n', encoding="utf-8"
    )
    (report_dir / "fixture_summary.json").write_text("[]\n", encoding="utf-8")
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)
    messages = [issue.message for issue in issues]

    assert any("non-finite JSON value" in message for message in messages)
    assert any("must contain an object" in message for message in messages)


@pytest.mark.parametrize(
    "artifact_name", ["verify.json", "run_summary.txt", "run_command.txt"]
)
def test_source_matrix_rejects_symlinked_authority_inputs(
    tmp_path: Path, artifact_name: str
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    artifact_path = report_dir / artifact_name
    target_path = report_dir / f"real-{artifact_name}"
    artifact_path.rename(target_path)
    artifact_path.symlink_to(target_path.name)
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        artifact_name in issue.path and "readable regular file" in issue.message
        for issue in issues
    )


def test_source_matrix_summary_and_command_are_not_last_key_or_substring_wins(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]
    (report_dir / "run_summary.txt").write_text(
        "status: failed\nstatus: success\n", encoding="utf-8"
    )
    (report_dir / "run_command.txt").write_text(
        "wrapper: run_tiny_hf_hqq.sh\n"
        "evaluate: invarlock evaluate --note "
        "--require-backend-inventory "
        "--require-runtime-quantization-proof\n",
        encoding="utf-8",
    )

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)
    messages = [issue.message for issue in issues]

    assert any(
        "run summary duplicates field 'status'" in message for message in messages
    )
    assert any(
        "run command is missing --require-backend-inventory" in message
        for message in messages
    )
    assert any(
        "run command is missing --require-runtime-quantization-proof" in message
        for message in messages
    )


def test_source_matrix_rejects_duplicate_command_fields(tmp_path: Path) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    (report_dir / "run_command.txt").write_text(
        "wrapper: first --require-backend-inventory "
        "--require-runtime-quantization-proof\n"
        "wrapper: second --require-backend-inventory "
        "--require-runtime-quantization-proof\n"
        "evaluate: invarlock evaluate baseline subject\n",
        encoding="utf-8",
    )
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        "run command duplicates field 'wrapper'" in issue.message for issue in issues
    )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_source_matrix_requires_exact_lane_artifact_v1_fields(
    tmp_path: Path, mutation: str
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    _write_matrix_artifact_set(report_dir)
    lane_path = report_dir / "lane_artifact.json"
    payload = json.loads(lane_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        payload["claimed_green"] = True
    else:
        payload.pop("device")
    lane_path.write_text(json.dumps(payload), encoding="utf-8")
    entry = json.loads(matrix_path.read_text(encoding="utf-8"))["entries"][0]

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        "lane artifact fields must match v1 exactly" in issue.message
        for issue in issues
    )


def test_source_matrix_rejects_hf_ct_strict_lane_without_packed_storage_proof(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    entry = {
        "target": "compressed_tensors",
        "readme": "examples/integrations/compressed_tensors/README.md",
        "verification_profile": "ci",
        "subject_adapter": "hf_ct",
        "lane": "cuda-container-strict",
        "report_path": "reports/tiny-hf-ct/<artifact-lane>",
        "required_artifacts": [],
        "expected": {},
    }

    issues = validator.validate_entry(tmp_path, entry, acceptance_inputs=None)

    assert any(
        "hf_ct) is not eligible for strict source matrix validation" in issue.message
        for issue in issues
    )


def test_source_matrix_artifact_validator_reports_malformed_json_artifacts(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    matrix_path = _write_test_source_matrix(tmp_path)
    report_dir = _report_dir(tmp_path)
    baseline_path, policy_path = _write_matrix_artifact_set(report_dir)
    (report_dir / "lane_artifact.json").write_text("{", encoding="utf-8")
    (report_dir / "verify.json").write_text("{", encoding="utf-8")

    _, issues = validator.validate_matrix(
        repo_root=tmp_path,
        matrix_path=matrix_path,
        targets={"hqq"},
        acceptance_inputs=_acceptance_inputs(validator, baseline_path, policy_path),
    )
    messages = [issue.message for issue in issues]

    assert any("lane artifact is not strict JSON" in message for message in messages)
    assert any("verify artifact is not strict JSON" in message for message in messages)
