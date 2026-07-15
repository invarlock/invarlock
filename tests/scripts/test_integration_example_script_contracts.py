from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET_V2
from invarlock.guards.authority import DEFAULT_GUARD_AUTHORITY
from scripts.smoke.gpt2_journey_helpers import write_strict_bundle_fixture
from tests.scripts._support_peft_torchao_integration_examples import (
    REPO_ROOT,
    TORCHAO_DIR,
)


def test_current_strict_smoke_fixture_binds_v2_guard_authority(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "evaluation.report.json"

    assert write_strict_bundle_fixture(Namespace(report=report_path)) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["assurance"]["claim_set"] == ASSURANCE_CLAIM_SET_V2
    assert report["resolved_policy"]["guard_authority"] == DEFAULT_GUARD_AUTHORITY
    assert report["assurance"]["guard_authority"] == DEFAULT_GUARD_AUTHORITY


def test_shared_compare_wrapper_checks_report_materialization() -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    subprocess.run(["bash", "-n", str(wrapper)], check=True)

    text = wrapper.read_text(encoding="utf-8")
    assert "Evaluate completed but did not write the expected report" in text
    assert "Evaluate completed but did not write the required backend inventory" in text
    assert (
        "Evaluate completed but did not write the required runtime quantization proof"
        in text
    )
    assert '[[ ! -s "$report_json" ]]' in text
    assert (
        '[[ "$require_backend_inventory" -eq 1 && ! -s "$backend_inventory_json" ]]'
        in text
    )
    assert (
        '[[ "$require_runtime_quantization_proof" -eq 1 && ! -s "$runtime_quantization_proof_json" ]]'
        in text
    )
    assert 'rm -f "$report_json" "$verify_json"' in text
    assert 'rm -f "$report_json" "$verify_json" "$backend_inventory_json"' in text
    assert 'rm -f "$runtime_quantization_proof_json"' in text
    assert 'CLI=("$PYTHON_BIN" -m invarlock)' in text
    assert '"${CLI[@]}" evaluate' in text
    assert '"${CLI[@]}" verify' in text
    assert "--lane MODE" in text
    assert 'execution_mode="container"' in text
    assert 'runtime_provenance="container"' in text
    assert 'device="cuda"' in text
    assert "lane_artifact.json" in text
    assert "lane_artifact_label" in text
    assert "run_summary.txt" in text
    assert 'internal_runs_dir="$report_out/.invarlock-evaluation-runs"' in text
    assert '--out "$internal_runs_dir"' in text
    assert "--require-backend-inventory" in text
    assert "--require-runtime-quantization-proof" in text
    assert "runtime_quantization_proof.json" in text
    assert "validate-sidecars" in text
    assert "invarlock.core.runtime_quantization_proof" in text
    assert "matching backend inventory" in text
    assert "--expected-runtime-image-digest" in text
    assert "INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST" in text
    assert "--baseline-report" in text
    assert "--baseline-revision" in text
    assert "--subject-revision" in text
    assert "INVARLOCK_ACCEPTANCE_BASELINE_REPORT" in text
    assert "--policy-pack" in text
    assert "INVARLOCK_ACCEPTANCE_POLICY_PACK" in text
    assert "InvarLock integration run complete" in text
    assert "InvarLock integration run failed" in text
    assert 'write_run_summary "success"' in text
    assert "emit_verify_summary_fields" in text
    assert "verify_status" in text
    assert "verify_runtime_provenance_status" in text
    assert "runtime provenance:" in text
    assert "integration_log_step" in text
    assert "integration_log_kv" in text
    assert "integration_default_host_device" in text
    assert "integration_require_strict_acceptance_inputs" in text
    assert "integration_lane_artifact_label" in text
    assert "integration_lane_report_out" in text
    assert "report_out_was_default=1" in text
    assert "report_out_was_default=0" in text
    assert "<artifact-lane>" in text


def test_shared_compare_wrapper_enforces_quantized_runtime_sidecars(
    tmp_path: Path,
) -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REAL_PYTHON = {sys.executable!r}

if sys.argv[1:3] != ["-m", "invarlock"]:
    raise SystemExit(subprocess.run([REAL_PYTHON, *sys.argv[1:]]).returncode)

args = sys.argv[3:]
command = args[0]
if command == "evaluate":
    assurance = args[args.index("--assurance") + 1]
    if assurance == "strict":
        if "--baseline-report" not in args:
            raise SystemExit("fake strict evaluate requires --baseline-report")
        baseline_report = Path(args[args.index("--baseline-report") + 1]).resolve()
        expected_baseline = Path(os.environ["FAKE_EXPECTED_BASELINE_REPORT"]).resolve()
        if baseline_report != expected_baseline:
            raise SystemExit("fake evaluate received the wrong baseline report")
    report_out = Path(args[args.index("--report-out") + 1])
    report_out.mkdir(parents=True, exist_ok=True)
    (report_out / "evaluation.report.json").write_text(
        json.dumps({{"schema": "fake", "results": []}}) + "\\n",
        encoding="utf-8",
    )
    if os.environ.get("FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY") == "1":
        inventory = json.loads(
            os.environ.get("FAKE_BACKEND_INVENTORY_JSON", '{{"adapter": "fake"}}')
        )
        (report_out / "backend_inventory.json").write_text(
            json.dumps(inventory) + "\\n",
            encoding="utf-8",
        )
    if os.environ.get("FAKE_INVARLOCK_WRITE_RUNTIME_QUANTIZATION_PROOF") == "1":
        proof = json.loads(
            os.environ.get("FAKE_RUNTIME_QUANTIZATION_PROOF_JSON", '{{"ok": true}}')
        )
        (report_out / "runtime_quantization_proof.json").write_text(
            json.dumps(proof) + "\\n",
            encoding="utf-8",
        )
    raise SystemExit(0)
if command == "verify":
    assurance = args[args.index("--assurance") + 1]
    if assurance == "strict":
        required_flags = {{
            "--baseline",
            "--policy-pack",
            "--expected-runtime-image-digest",
        }}
        missing_flags = sorted(required_flags - set(args))
        if missing_flags:
            raise SystemExit(f"fake strict verify missing inputs: {{missing_flags!r}}")
        baseline_report = Path(args[args.index("--baseline") + 1]).resolve()
        policy_pack = Path(args[args.index("--policy-pack") + 1]).resolve()
        expected_baseline = Path(os.environ["FAKE_EXPECTED_BASELINE_REPORT"]).resolve()
        expected_policy = Path(os.environ["FAKE_EXPECTED_POLICY_PACK"]).resolve()
        if baseline_report != expected_baseline:
            raise SystemExit("fake verify received the wrong baseline report")
        if policy_pack != expected_policy:
            raise SystemExit("fake verify received the wrong policy pack")
        report_paths = [
            Path(arg).resolve()
            for arg in args
            if arg.endswith("evaluation.report.json")
        ]
        if len(report_paths) != 1 or report_paths[0] == baseline_report:
            raise SystemExit("fake verify requires distinct subject and baseline reports")
    payload = {{
        "summary": {{"ok": True, "reason": "ok"}},
        "results": [
            {{
                "verification": {{
                    "runtime_provenance": {{
                        "declared_mode": "container",
                        "status": "expected_image_digest_matched",
                        "verified": True,
                        "expected_digest_matched": True,
                    }}
                }}
            }}
        ],
    }}
    print(json.dumps(payload))
    raise SystemExit(0)
if command == "report" and args[1] == "html":
    output = Path(args[args.index("-o") + 1])
    output.write_text("<html></html>\\n", encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(f"unexpected fake invarlock command: {{args!r}}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    base_cmd = [
        str(wrapper),
        "--baseline",
        "dense",
        "--subject",
        "quant",
        "--subject-adapter",
        "hf_hqq",
        "--baseline-revision",
        "0123456789abcdef",
        "--subject-revision",
        "fedcba9876543210",
        "--report-out",
        str(tmp_path / "reports"),
        "--lane",
        "cuda",
        "--require-backend-inventory",
        "--require-runtime-quantization-proof",
        "--no-html",
    ]
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(fake_python)
    env["INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST"] = "sha256:" + ("a" * 64)
    baseline_report = tmp_path / "acceptance-baseline.report.json"
    baseline_report.write_text('{"schema": "raw-baseline"}\n', encoding="utf-8")
    policy_pack = tmp_path / "acceptance-policy-pack.json"
    policy_pack.write_text('{"schema": "policy-pack-v2"}\n', encoding="utf-8")
    env["INVARLOCK_ACCEPTANCE_BASELINE_REPORT"] = str(baseline_report)
    env["INVARLOCK_ACCEPTANCE_POLICY_PACK"] = str(policy_pack)
    env["FAKE_EXPECTED_BASELINE_REPORT"] = str(baseline_report)
    env["FAKE_EXPECTED_POLICY_PACK"] = str(policy_pack)
    valid_inventory = {
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
    valid_proof = {
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

    malformed_digest_env = dict(env)
    malformed_digest_env["INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST"] = "sha256:ABC"
    malformed_digest = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=malformed_digest_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert malformed_digest.returncode == 2
    assert "must be a canonical sha256 digest" in malformed_digest.stderr

    non_object_baseline = tmp_path / "non-object-baseline.json"
    non_object_baseline.write_text("[]\n", encoding="utf-8")
    non_object_baseline_env = dict(env)
    non_object_baseline_env["INVARLOCK_ACCEPTANCE_BASELINE_REPORT"] = str(
        non_object_baseline
    )
    rejected_baseline = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=non_object_baseline_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_baseline.returncode == 2
    assert "baseline report must contain one valid JSON object" in (
        rejected_baseline.stderr
    )

    malformed_policy = tmp_path / "malformed-policy.json"
    malformed_policy.write_text("{\n", encoding="utf-8")
    malformed_policy_env = dict(env)
    malformed_policy_env["INVARLOCK_ACCEPTANCE_POLICY_PACK"] = str(malformed_policy)
    rejected_policy = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=malformed_policy_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_policy.returncode == 2
    assert "policy pack must contain one valid JSON object" in rejected_policy.stderr

    unpinned_env = dict(env)
    unpinned_env.pop("INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST")
    unpinned = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=unpinned_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unpinned.returncode == 2
    assert "Strict assurance requires an independently supplied runtime image pin" in (
        unpinned.stderr
    )

    no_baseline_env = dict(env)
    no_baseline_env.pop("INVARLOCK_ACCEPTANCE_BASELINE_REPORT")
    no_baseline = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=no_baseline_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_baseline.returncode == 2
    assert (
        "requires an independently supplied raw baseline report" in no_baseline.stderr
    )

    no_policy_env = dict(env)
    no_policy_env.pop("INVARLOCK_ACCEPTANCE_POLICY_PACK")
    no_policy = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=no_policy_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_policy.returncode == 2
    assert "requires an independently supplied policy pack" in no_policy.stderr

    missing = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing.returncode == 1
    assert "required backend inventory" in missing.stderr
    assert (tmp_path / "reports" / "evaluation.report.json").is_file()
    assert not (tmp_path / "reports" / "backend_inventory.json").exists()

    proof_missing_env = dict(env)
    proof_missing_env["FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY"] = "1"
    proof_missing = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=proof_missing_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proof_missing.returncode == 1
    assert "required runtime quantization proof" in proof_missing.stderr
    assert (tmp_path / "reports" / "backend_inventory.json").is_file()
    assert not (tmp_path / "reports" / "runtime_quantization_proof.json").exists()

    fake_green_env = dict(env)
    fake_green_env["FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY"] = "1"
    fake_green_env["FAKE_INVARLOCK_WRITE_RUNTIME_QUANTIZATION_PROOF"] = "1"
    fake_green = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=fake_green_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert fake_green.returncode == 1
    assert "runtime quantization proof validation failed" in fake_green.stderr

    mismatched_inventory_env = dict(fake_green_env)
    mismatched_inventory = dict(valid_inventory)
    mismatched_inventory["adapter"] = "hf_bnb"
    mismatched_inventory["backend"] = "bitsandbytes"
    mismatched_inventory_env["FAKE_BACKEND_INVENTORY_JSON"] = json.dumps(
        mismatched_inventory
    )
    mismatched_inventory_env["FAKE_RUNTIME_QUANTIZATION_PROOF_JSON"] = json.dumps(
        valid_proof
    )
    mismatched_inventory_result = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        env=mismatched_inventory_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatched_inventory_result.returncode == 1
    assert (
        "backend inventory adapter does not match" in mismatched_inventory_result.stderr
    )

    explicit_env = dict(env)
    explicit_env.pop("INVARLOCK_ACCEPTANCE_BASELINE_REPORT")
    explicit_env.pop("INVARLOCK_ACCEPTANCE_POLICY_PACK")
    explicit_env["FAKE_INVARLOCK_WRITE_BACKEND_INVENTORY"] = "1"
    explicit_env["FAKE_INVARLOCK_WRITE_RUNTIME_QUANTIZATION_PROOF"] = "1"
    explicit_env["FAKE_BACKEND_INVENTORY_JSON"] = json.dumps(valid_inventory)
    explicit_env["FAKE_RUNTIME_QUANTIZATION_PROOF_JSON"] = json.dumps(valid_proof)
    ok = subprocess.run(
        [
            *base_cmd,
            "--baseline-report",
            str(baseline_report),
            "--policy-pack",
            str(policy_pack),
        ],
        cwd=REPO_ROOT,
        env=explicit_env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert ok.returncode == 0
    assert "InvarLock integration run complete" in ok.stdout
    assert (tmp_path / "reports" / "backend_inventory.json").is_file()
    assert (tmp_path / "reports" / "runtime_quantization_proof.json").is_file()
    assert (tmp_path / "reports" / "verify.json").is_file()
    assert "status: success" in (tmp_path / "reports" / "run_summary.txt").read_text(
        encoding="utf-8"
    )
    run_command = (tmp_path / "reports" / "run_command.txt").read_text(encoding="utf-8")
    assert "--out" in run_command
    assert ".invarlock-evaluation-runs" in run_command
    assert "--expected-runtime-image-digest" in run_command
    assert "--baseline-revision 0123456789abcdef" in run_command
    assert "--subject-revision fedcba9876543210" in run_command
    assert f"--baseline-report {baseline_report}" in run_command
    assert f"--baseline {baseline_report}" in run_command
    assert f"--policy-pack {policy_pack}" in run_command

    host_env = dict(explicit_env)
    host_env.pop("INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST")
    host = subprocess.run(
        [
            str(wrapper),
            "--baseline",
            "dense",
            "--subject",
            "quant",
            "--report-out",
            str(tmp_path / "host-reports"),
            "--lane",
            "host",
            "--no-html",
        ],
        cwd=REPO_ROOT,
        env=host_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert host.returncode == 0
    host_command = (tmp_path / "host-reports" / "run_command.txt").read_text(
        encoding="utf-8"
    )
    assert "--assurance off" in host_command
    assert "--execution-mode host" in host_command
    assert "--baseline-report" not in host_command
    assert "--policy-pack" not in host_command
    assert "--expected-runtime-image-digest" not in host_command


def test_shared_source_archive_helper_avoids_macos_xattrs() -> None:
    helper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "create_source_archive.sh"
    )
    subprocess.run(["bash", "-n", str(helper)], check=True)

    text = helper.read_text(encoding="utf-8")
    assert 'git -C "$REPO_ROOT" archive --format=tar.gz' in text
    assert "--include-worktree" in text
    assert "COPYFILE_DISABLE=1" in text
    assert "--no-xattrs" in text
    assert "ls-files -z --cached --modified --others --exclude-standard" in text


def test_shared_expected_artifacts_documents_backend_inventory() -> None:
    text = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "expected-artifacts.md"
    ).read_text(encoding="utf-8")

    assert "`backend_inventory.json`" in text
    assert "`external_edit_summary.json`" in text
    assert "`adapter_runtime_summary.json`" in text
    assert "`fixture_summary.json`" in text
    assert "InvarLock report persistence" in text
    assert "adapter provenance is available" in text
    assert "reports/<target>/<artifact-lane>/evaluation.report.json" in text
    assert "--runtime-provenance container" in text
    assert "For the primary CUDA/container strict lane" in text


def test_shared_evidence_scope_documents_source_matrix_contract() -> None:
    text = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "evidence-scope.md"
    ).read_text(encoding="utf-8")

    assert "`source_matrix.json` is the source-controlled contract" in text
    assert "strict-lane\nrequirements" in text
    assert "`source_matrix.json` has an entry" in text
    assert "`checkpoint_refs.json`, `external_edit_summary.json`" in text
    assert "`adapter_runtime_summary.json`, `training_receipt.json`," in text
    assert "`training_binding.json`, `training_evidence_proof.json`," in text
    assert "`training_profile_snapshot.json`, and `fixture_summary.json`" in text
    assert "`fixture_summary.json`" in text


def test_shared_preflight_helper_defines_host_lane_contract() -> None:
    helper = REPO_ROOT / "examples" / "integrations" / "_shared" / "preflight.sh"
    subprocess.run(["bash", "-n", str(helper)], check=True)

    text = helper.read_text(encoding="utf-8")
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_preflight_gptqmodel_host_runtime" in text
    assert "integration_lane_artifact_label" in text
    assert "integration_lane_report_out" in text
    assert "integration_effective_assurance" in text
    assert "integration_require_strict_runtime_pin" in text
    assert "integration_require_strict_acceptance_inputs" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "integration_log_kv" in text
    assert "cuda-container-strict" in text
    assert "cuda-host-off" in text
    assert "cpu-host-off" in text


def test_torchao_runner_wires_local_fixture() -> None:
    runner = TORCHAO_DIR / "run_tiny_hf_torchao_int8.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "prepare_tiny_hf_torchao_fixture.py" in text
    assert "--model-dir" in text
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert '--baseline "$model_dir"' in text
    assert '--subject "$model_dir"' in text
    assert "--baseline-adapter hf_causal" in text
    assert "--subject-adapter hf_torchao" in text
    assert "--edit-label torchao_int8_runtime_quantization" in text
    assert "adapter_runtime_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "integration_default_host_device" in text
    assert "integration_preflight_host_cuda_device" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_torchao_readme_frames_hf_torchao_as_primary_path() -> None:
    text = (TORCHAO_DIR / "README.md").read_text(encoding="utf-8")

    assert "torchao Int8 Runtime Integration Example" in text
    assert "`hf_torchao` adapter" in text
    assert "`cuda-container-strict` result requires" in text
    assert "same checkpoint as the subject loaded through `hf_torchao`" in text
    assert "runnable evidence path is the `hf_torchao` subject" in text
    assert "scoped to the configured tiny HF checkpoint" in text
    assert "shared integration evidence" in text
    assert "run_tiny_hf_torchao_int8.sh" in text
