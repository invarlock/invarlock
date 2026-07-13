from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.scripts._support_peft_torchao_integration_examples import (
    EXAMPLE_RUNNERS,
    FINE_TUNE_DIR,
    INTEGRATIONS_DIR,
    MAGNITUDE_PRUNE_DIR,
    PEFT_DIR,
    README_EXAMPLES,
    REPO_ROOT,
    TORCHAO_DIR,
    _load_source_matrix,
)


def test_peft_lora_runner_wires_local_fixture() -> None:
    runner = PEFT_DIR / "run_tiny_peft_lora.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--runtime-provenance" in text
    assert "--device" in text
    assert "--training-profile" in text
    assert "train-profile" not in text
    assert "integration_run_training_profile" in text
    assert "verify-training-profile" not in text
    assert "training_receipt.json" in text
    assert "training_binding.json" in text
    assert "integration_finalize_training_binding" in text
    assert "external_edit_summary.json" not in text
    assert "--baseline-revision" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text
    assert "import peft, torch, transformers" in text
    assert 'for candidate in python "$REPO_ROOT/.venv/bin/python" python3' in text


def test_fine_tune_runner_wires_local_fixture() -> None:
    runner = FINE_TUNE_DIR / "run_tiny_fine_tune.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--runtime-provenance" in text
    assert "--device" in text
    assert "--training-profile" in text
    assert "integration_run_training_profile" in text
    assert "training_receipt.json" in text
    assert "training_binding.json" in text
    assert "integration_finalize_training_binding" in text
    assert "external_edit_summary.json" not in text
    assert "--baseline-revision" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text
    assert "import torch, transformers" in text
    assert 'for candidate in python "$REPO_ROOT/.venv/bin/python" python3' in text
    assert "--edit-label fine_tune" in text


def test_magnitude_prune_runner_wires_local_fixture() -> None:
    runner = MAGNITUDE_PRUNE_DIR / "run_tiny_magnitude_prune.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)

    text = runner.read_text(encoding="utf-8")
    assert "--fixture-dir" in text
    assert "--preset" in text
    assert "fixture_summary.json" in text
    assert "--lane MODE" in text
    assert 'compare_cmd+=(--lane "$lane")' in text
    assert "--runtime-provenance" in text
    assert "--device" in text
    assert "--prune-fraction" in text
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text
    assert "select_python_bin transformers" in text
    assert 'for candidate in python "$REPO_ROOT/.venv/bin/python" python3' in text
    assert "import ${required_module}" in text
    assert "--edit-label magnitude_prune" in text


def test_integration_example_readmes_document_run_lanes() -> None:
    expected_headings = {
        "awq": ["### cuda-host-off lane", "### cuda-container-strict lane"],
        "compressed_tensors": [
            "### cpu-host-off lane",
            "### Strict assurance is unavailable",
        ],
        "gptqmodel": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "hf_bnb": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "hqq": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "lm_eval_harness": [
            "### cpu-host-off lane",
            "### cuda-host-off lane",
            "### mps-host-off lane",
        ],
        "peft_lora": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "fine_tune": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "magnitude_prune": [
            "### cpu-host-off lane",
            "### cuda-container-strict lane",
        ],
        "quanto": ["### cpu-host-off lane", "### cuda-container-strict lane"],
        "torchao_int8_runtime": [
            "### cpu-host-off lane",
            "### cuda-container-strict lane",
        ],
    }

    integrations = REPO_ROOT / "examples" / "integrations"
    for example, headings in expected_headings.items():
        text = (integrations / example / "README.md").read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text, f"{example} missing run lane {heading!r}"

    for example in [
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "peft_lora",
        "fine_tune",
        "magnitude_prune",
        "quanto",
        "torchao_int8_runtime",
    ]:
        text = (integrations / example / "README.md").read_text(encoding="utf-8")
        assert "`cpu-host-off`" in text
        assert "`cuda-host-off`" in text
        assert "`cuda-container-strict`" in text
        assert "--lane host" in text
        assert "--lane cuda" in text
        assert "--device cpu" in text
        assert "--device cuda" in text
        lane_support = text.split("### Lane Support", 1)[1]
        assert lane_support.index("`cuda-container-strict`") < lane_support.index(
            "`cuda-host-off`"
        )
        assert lane_support.index("`cuda-container-strict`") < lane_support.index(
            "`cpu-host-off`"
        )
        assert "run_summary.txt" in text
        assert "verifier status, runtime provenance status" in text
        assert "shared completion block" in text
        assert "INVARLOCK_ACCEPTANCE_BASELINE_REPORT" in text
        assert "INVARLOCK_ACCEPTANCE_POLICY_PACK" in text

    compressed_tensors_text = (
        integrations / "compressed_tensors" / "README.md"
    ).read_text(encoding="utf-8")
    assert "`cpu-host-off`" in compressed_tensors_text
    assert "`cuda-host-off`" in compressed_tensors_text
    assert "`cuda-container-strict`" in compressed_tensors_text
    assert "Strict assurance is unavailable" in compressed_tensors_text
    assert "--lane host" in compressed_tensors_text
    assert "--lane cuda" in compressed_tensors_text
    assert "--device cpu" in compressed_tensors_text
    assert "--device cuda" in compressed_tensors_text
    assert "not a packed-storage\nartifact proof" in compressed_tensors_text

    awq_text = (integrations / "awq" / "README.md").read_text(encoding="utf-8")
    assert "`cpu-host-off`" not in awq_text
    assert "`cuda-host-off`" in awq_text
    assert "`cuda-container-strict`" in awq_text
    assert "--lane host" in awq_text
    assert "--lane host --device cuda" in awq_text
    assert "--lane cuda" in awq_text
    assert "--device cpu" not in awq_text
    assert awq_text.index("`cuda-container-strict`") < awq_text.index("`cuda-host-off`")
    assert "run_summary.txt" in awq_text
    assert "verifier status, runtime provenance status" in awq_text
    assert "INVARLOCK_ACCEPTANCE_BASELINE_REPORT" in awq_text
    assert "INVARLOCK_ACCEPTANCE_POLICY_PACK" in awq_text

    lm_eval_text = (integrations / "lm_eval_harness" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "`mps-host-off`" in lm_eval_text
    assert "--device mps" in lm_eval_text
    assert 'uv run --extra hf --with "lm_eval[hf]"' in lm_eval_text
    assert "uv run --extra hf --with peft" in lm_eval_text
    assert lm_eval_text.index("`cuda-container-strict`") < lm_eval_text.index(
        "`cuda-host-off`"
    )
    assert "primary evidence" in lm_eval_text
    assert "run_summary.txt" in lm_eval_text
    assert "verifier status, runtime provenance status" in lm_eval_text

    peft_text = (integrations / "peft_lora" / "README.md").read_text(encoding="utf-8")
    assert "uv run --extra training" in peft_text


def test_integration_runners_default_reports_are_lane_scoped() -> None:
    for runner in EXAMPLE_RUNNERS:
        subprocess.run(["bash", "-n", str(runner)], check=True)

        text = runner.read_text(encoding="utf-8")
        assert "<artifact-lane>" in text, f"{runner} missing help contract"
        assert "report_out_was_default=1" in text, f"{runner} missing default flag"
        assert "report_out_was_default=0" in text, f"{runner} missing override flag"
        assert "integration_lane_report_out" in text, f"{runner} missing lane output"
        if runner.parent.name != "lm_eval_harness":
            assert "integration_require_strict_acceptance_inputs" in text
            assert "INVARLOCK_ACCEPTANCE_BASELINE_REPORT" in text
            assert "INVARLOCK_ACCEPTANCE_POLICY_PACK" in text


def test_source_archive_git_warning_filter_is_shared_for_external_materializers() -> (
    None
):
    preflight = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "preflight.sh"
    ).read_text(encoding="utf-8")

    assert "integration_filter_source_archive_stderr" in preflight
    assert "integration_run_source_archive_clean" in preflight
    assert "fatal: not a git repository" in preflight

    for runner in [
        INTEGRATIONS_DIR / "awq" / "run_tiny_awq.sh",
        INTEGRATIONS_DIR / "gptqmodel" / "run_tiny_gptqmodel.sh",
        INTEGRATIONS_DIR / "lm_eval_harness" / "run_tiny_lm_eval_sidecar.sh",
        INTEGRATIONS_DIR / "peft_lora" / "run_tiny_peft_lora.sh",
        INTEGRATIONS_DIR / "fine_tune" / "run_tiny_fine_tune.sh",
        INTEGRATIONS_DIR / "magnitude_prune" / "run_tiny_magnitude_prune.sh",
    ]:
        text = runner.read_text(encoding="utf-8")
        assert "integration_run_source_archive_clean" in text


def test_integration_readmes_use_run_lane_subsections() -> None:
    for example in README_EXAMPLES:
        readme = INTEGRATIONS_DIR / example / "README.md"
        text = readme.read_text(encoding="utf-8")

        assert "## Run\n\n## Lane Support" not in text
        assert "## Run\n\n### Lane Support" in text


def test_integration_readme_report_paths_are_lane_scoped() -> None:
    for example in README_EXAMPLES:
        readme = INTEGRATIONS_DIR / example / "README.md"
        text = readme.read_text(encoding="utf-8")
        report_paths = re.findall(r"`(reports/tiny-[^`]+)`", text)

        assert report_paths, f"{example} README has no report artifact paths"
        for report_path in report_paths:
            assert "/<artifact-lane>/" in report_path, (
                f"{example} report path is not lane-scoped: {report_path}"
            )


def test_strict_evidence_claim_readmes_have_artifact_source_matrix() -> None:
    matrix_entries = _load_source_matrix()
    assert "compressed_tensors" not in matrix_entries
    common_artifacts = {
        "evaluation.report.json",
        "verify.json",
        "runtime.manifest.json",
        "evaluation.html",
        "lane_artifact.json",
        "run_command.txt",
        "run_summary.txt",
    }
    quantized_strict_targets = {
        "awq",
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "quanto",
        "torchao_int8_runtime",
    }
    target_provenance_artifacts = {
        "awq": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "gptqmodel": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "hf_bnb": {"fixture_summary.json"},
        "hqq": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
        "peft_lora": {
            "training_receipt.json",
            "training_evidence_proof.json",
            "training_profile_snapshot.json",
            "fixture_summary.json",
        },
        "fine_tune": {
            "training_receipt.json",
            "training_evidence_proof.json",
            "training_profile_snapshot.json",
            "fixture_summary.json",
        },
        "magnitude_prune": {
            "checkpoint_refs.json",
            "external_edit_summary.json",
            "fixture_summary.json",
        },
        "quanto": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
        "torchao_int8_runtime": {
            "checkpoint_refs.json",
            "adapter_runtime_summary.json",
            "fixture_summary.json",
        },
    }

    claimed_readmes = {}
    for readme in INTEGRATIONS_DIR.rglob("README.md"):
        text = readme.read_text(encoding="utf-8")
        if "`cuda-container-strict` result requires" in text:
            claimed_readmes[readme.parent.name] = text

    assert set(claimed_readmes) == set(matrix_entries)

    shared_expected_artifacts = (
        INTEGRATIONS_DIR / "_shared" / "expected-artifacts.md"
    ).read_text(encoding="utf-8")
    shared_training_profiles = (
        INTEGRATIONS_DIR / "_shared" / "training_profiles.sh"
    ).read_text(encoding="utf-8")
    shared_training_artifacts = {
        "training_evidence_proof.json",
        "training_profile_snapshot.json",
    }
    for artifact in shared_training_artifacts:
        assert artifact in shared_expected_artifacts

    for example, text in claimed_readmes.items():
        entry = matrix_entries[example]
        readme = INTEGRATIONS_DIR / example / "README.md"
        runner = Path(entry["runner"])
        runtime_image = entry["runtime_image"]
        expected = entry["expected"]
        required_artifacts = set(entry["required_artifacts"])
        provenance_artifacts = set(entry["provenance_artifacts"])

        assert Path(entry["readme"]) == readme.relative_to(REPO_ROOT)
        assert (REPO_ROOT / runner).is_file()
        assert entry["strict_claim_phrase"] in text
        assert entry["lane"] == "cuda-container-strict"
        assert entry["verification_profile"] in {"ci", "release"}
        assert entry["command_shape"] == "--lane cuda"
        assert "`cuda-container-strict`" in text
        assert str(entry["report_path"]) in text
        assert runtime_image["source_command"] in text
        assert runtime_image["declared_digest_source"] == "runtime.manifest.json"
        assert (
            runtime_image["expected_digest_source"]
            == "wrapper_input_from_independent_policy"
        )
        assert expected["lane_artifact_label"] == "cuda-container-strict"
        assert expected["verify_status"] == "ok"
        assert expected["runtime_provenance_declared"] == "container"
        assert expected["runtime_provenance_verified"] is True
        assert expected["runtime_provenance_status"] == (
            "expected_image_digest_matched"
        )
        assert expected["runtime_expected_digest_matched"] is True
        assert common_artifacts <= required_artifacts
        assert provenance_artifacts == target_provenance_artifacts[example]
        assert provenance_artifacts <= required_artifacts
        if example in {"peft_lora", "fine_tune"}:
            assert "training_binding.json" in required_artifacts
            assert shared_training_artifacts <= required_artifacts
            expected_profile = (
                "tiny_gpt2_lora_cuda_v1"
                if example == "peft_lora"
                else "tiny_gpt2_full_ft_cuda_v1"
            )
            expected_scope = "attn" if example == "peft_lora" else "all"
            assert entry["training_profile"] == expected_profile
            assert entry["training_scope"] == expected_scope

        for artifact in required_artifacts:
            assert artifact in text

        runner_text = (REPO_ROOT / runner).read_text(encoding="utf-8")
        for artifact in provenance_artifacts:
            if artifact in shared_training_artifacts:
                assert artifact in shared_training_profiles
            else:
                assert artifact in runner_text

        if example in {"peft_lora", "fine_tune"}:
            assert "integration_stage_training_evidence" in runner_text
            assert f'"{entry["training_scope"]}"' in runner_text

        if example in quantized_strict_targets:
            assert "backend_inventory.json" in required_artifacts
            assert "runtime_quantization_proof.json" in required_artifacts
            assert (
                entry["runner_enforcement"]["backend_inventory"]
                == "--require-backend-inventory"
            )
            assert (
                entry["runner_enforcement"]["runtime_quantization_proof"]
                == "--require-runtime-quantization-proof"
            )
            assert "--require-backend-inventory" in runner_text
            assert "--require-runtime-quantization-proof" in runner_text
            assert 'lane_artifact_label" == "cuda-container-strict"' in runner_text
        else:
            assert "backend_inventory.json" not in required_artifacts
            assert "runtime_quantization_proof.json" not in required_artifacts
            assert entry["runner_enforcement"] == {}


def test_materialized_subject_readmes_define_evidence_boundary() -> None:
    expectations = {
        "awq": ["`hf_awq`", "`external_edit_summary.json`"],
        "compressed_tensors": ["`hf_ct`", "`adapter_runtime_summary.json`"],
        "gptqmodel": ["`hf_gptq`", "`external_edit_summary.json`"],
        "peft_lora": ["`hf_causal`", "`training_receipt.json`"],
        "fine_tune": ["`hf_causal`", "`training_receipt.json`"],
        "magnitude_prune": ["`hf_causal`", "`external_edit_summary.json`"],
    }

    for example, phrases in expectations.items():
        text = (INTEGRATIONS_DIR / example / "README.md").read_text(encoding="utf-8")

        assert "## Evidence Boundary" in text, f"{example} lacks evidence boundary"
        if example in {"peft_lora", "fine_tune"}:
            assert "The runner invokes its immutable profile before" in text
            assert "optimizer-history" in text
            assert "`training_evidence_proof.json` verifies" in text
            assert "saved state" in text
            assert "behavior for that subject" in text
            assert "not independent proof of training execution" in text
            assert "exact Python version" in text
            assert "exact Torch build string" in text
        else:
            assert "The subject checkpoint is materialized before" in text
        if example == "compressed_tensors":
            assert "not a packed-storage\nartifact proof" in text
        else:
            assert "verifier result for that\nproduced subject" in text
        for phrase in phrases:
            assert phrase in text


def test_torchao_readme_documents_backend_inventory_sidecar() -> None:
    text = (TORCHAO_DIR / "README.md").read_text(encoding="utf-8")

    assert (
        "`reports/tiny-hf-torchao-int8/<artifact-lane>/backend_inventory.json`" in text
    )
    assert "The shell runner relies on InvarLock report persistence to emit" in text
    assert "`backend_inventory.json` when adapter provenance is available" in text
    assert "adapter provenance is available" in text
    assert (
        "`reports/tiny-hf-torchao-int8/<artifact-lane>/"
        "runtime_quantization_proof.json`" in text
    )
    assert "v1 process receipt listing recognized torchao runtime types" in text
    assert "not an independent runtime observation" in text


def test_shared_example_docs_scope_source_archives_and_image_digests() -> None:
    shared_readme = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "README.md"
    ).read_text(encoding="utf-8")
    image_readme = (
        REPO_ROOT / "examples" / "integrations" / "_runtime_images" / "README.md"
    ).read_text(encoding="utf-8")

    assert "Use `--committed` when sharing an archive" in shared_readme
    assert "Use `--include-worktree` only" in shared_readme
    assert "deliberately including local changes" in shared_readme
    assert "Rebuilding an example image may" in image_readme
    assert "INVARLOCK_EXPECTED_RUNTIME_IMAGE_DIGEST" in image_readme
    assert "not itself an independent source" in image_readme


def test_peft_readme_scopes_strict_evidence_to_tiny_runtime() -> None:
    text = (PEFT_DIR / "README.md").read_text(encoding="utf-8")

    assert "`cuda-container-strict` result requires" in text
    assert "scoped to the configured tiny merged dense checkpoint" in text
    assert "shared integration evidence" in text


def test_integration_example_docs_use_canonical_lane_wording() -> None:
    scanned_paths = list((REPO_ROOT / "examples" / "integrations").rglob("README.md"))
    scanned_paths.extend(
        [
            REPO_ROOT
            / "examples"
            / "integrations"
            / "_shared"
            / "run_invarlock_compare.sh",
            REPO_ROOT / "examples" / "integrations" / "_shared" / "preflight.sh",
        ]
    )

    stale_phrases = [
        "host" + "/" + "off",
        "CPU host" + "/" + "off",
        "CUDA host" + "/" + "off",
        "CUDA" + "/" + "container",
        "strict-cuda" + "-container",
        "gpu-host" + "-off",
    ]

    for path in scanned_paths:
        text = path.read_text(encoding="utf-8")
        for phrase in stale_phrases:
            assert phrase not in text, f"{path} contains stale lane wording {phrase!r}"
