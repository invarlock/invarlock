from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS_DIR = REPO_ROOT / "examples" / "integrations"
PEFT_DIR = REPO_ROOT / "examples" / "integrations" / "peft_lora"
TORCHAO_DIR = REPO_ROOT / "examples" / "integrations" / "torchao_int8_runtime"

EXAMPLE_RUNNERS = [
    INTEGRATIONS_DIR / "awq" / "run_tiny_awq.sh",
    INTEGRATIONS_DIR / "compressed_tensors" / "run_tiny_hf_ct.sh",
    INTEGRATIONS_DIR / "gptqmodel" / "run_tiny_gptqmodel.sh",
    INTEGRATIONS_DIR / "hf_bnb" / "run_tiny_hf_bnb_8bit.sh",
    INTEGRATIONS_DIR / "hqq" / "run_tiny_hf_hqq.sh",
    INTEGRATIONS_DIR / "lm_eval_harness" / "run_tiny_lm_eval_sidecar.sh",
    INTEGRATIONS_DIR / "peft_lora" / "run_tiny_peft_lora.sh",
    INTEGRATIONS_DIR / "quanto" / "run_tiny_hf_quanto.sh",
    INTEGRATIONS_DIR / "torchao_int8_runtime" / "run_tiny_hf_torchao_int8.sh",
]

README_EXAMPLES = [
    "awq",
    "compressed_tensors",
    "gptqmodel",
    "hf_bnb",
    "hqq",
    "lm_eval_harness",
    "peft_lora",
    "quanto",
    "torchao_int8_runtime",
]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    assert "integration_log_header" in text
    assert "integration_log_step" in text
    assert "lane_artifact_label" in text


def test_integration_example_readmes_document_run_lanes() -> None:
    expected_headings = {
        "awq": ["### cuda-host-off lane", "### cuda-container-strict lane"],
        "compressed_tensors": [
            "### cpu-host-off lane",
            "### cuda-container-strict lane",
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
        "compressed_tensors",
        "gptqmodel",
        "hf_bnb",
        "hqq",
        "peft_lora",
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
        assert text.index("`cuda-container-strict`") < text.index("`cuda-host-off`")
        assert text.index("`cuda-container-strict`") < text.index("`cpu-host-off`")
        assert "run_summary.txt" in text
        assert "verifier status, runtime provenance status" in text
        assert "shared completion block" in text

    awq_text = (integrations / "awq" / "README.md").read_text(encoding="utf-8")
    assert "`cpu-host-off`" not in awq_text
    assert "`cuda-host-off`" in awq_text
    assert "`cuda-container-strict`" in awq_text
    assert "--lane host" in awq_text
    assert "--lane cuda" in awq_text
    assert "--device cpu" not in awq_text
    assert awq_text.index("`cuda-container-strict`") < awq_text.index("`cuda-host-off`")
    assert "run_summary.txt" in awq_text
    assert "verifier status, runtime provenance status" in awq_text

    lm_eval_text = (integrations / "lm_eval_harness" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "`mps-host-off`" in lm_eval_text
    assert "--device mps" in lm_eval_text
    assert lm_eval_text.index("`cuda-container-strict`") < lm_eval_text.index(
        "`cuda-host-off`"
    )
    assert "primary evidence" in lm_eval_text
    assert "run_summary.txt" in lm_eval_text
    assert "verifier status, runtime provenance status" in lm_eval_text


def test_integration_runners_default_reports_are_lane_scoped() -> None:
    for runner in EXAMPLE_RUNNERS:
        subprocess.run(["bash", "-n", str(runner)], check=True)

        text = runner.read_text(encoding="utf-8")
        assert "<artifact-lane>" in text, f"{runner} missing help contract"
        assert "report_out_was_default=1" in text, f"{runner} missing default flag"
        assert "report_out_was_default=0" in text, f"{runner} missing override flag"
        assert "integration_lane_report_out" in text, f"{runner} missing lane output"


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
    expected_sources = {
        "awq": "build_example_runtime_image.sh cuda-gptqmodel",
        "gptqmodel": "build_example_runtime_image.sh cuda-gptqmodel",
        "hf_bnb": "build_example_runtime_image.sh cuda-bnb",
        "peft_lora": "make runtime-image-cuda",
        "torchao_int8_runtime": "build_example_runtime_image.sh cuda-torchao",
    }

    claimed_readmes = {}
    for readme in INTEGRATIONS_DIR.rglob("README.md"):
        text = readme.read_text(encoding="utf-8")
        if "strict container evidence is verified" in text:
            claimed_readmes[readme.parent.name] = text

    assert set(claimed_readmes) == set(expected_sources)

    for example, source_command in expected_sources.items():
        text = claimed_readmes[example]

        assert source_command in text
        assert "INVARLOCK_RUNTIME_IMAGE=" in text
        assert "`cuda-container-strict`" in text
        assert "`runtime.manifest.json`" in text
        assert "/<artifact-lane>/" in text
        assert "evaluation.report.json" in text
        assert "verify.json" in text
        assert "lane_artifact.json" in text
        assert "run_command.txt" in text
        assert "run_summary.txt" in text


def test_materialized_subject_readmes_define_evidence_boundary() -> None:
    expectations = {
        "awq": ["`hf_awq`", "`external_edit_summary.json`"],
        "compressed_tensors": ["`hf_ct`", "`adapter_runtime_summary.json`"],
        "gptqmodel": ["`hf_gptq`", "`external_edit_summary.json`"],
        "peft_lora": ["`hf_causal`", "`external_edit_summary.json`"],
    }

    for example, phrases in expectations.items():
        text = (INTEGRATIONS_DIR / example / "README.md").read_text(encoding="utf-8")

        assert "## Evidence Boundary" in text, f"{example} lacks evidence boundary"
        assert "The subject checkpoint is materialized before" in text
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
    assert "may produce a different image digest" in image_readme
    assert "digest recorded in `runtime.manifest.json`" in image_readme


def test_peft_readme_scopes_strict_evidence_to_tiny_runtime() -> None:
    text = (PEFT_DIR / "README.md").read_text(encoding="utf-8")

    assert "strict container evidence is verified on CUDA for this tiny" in text
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


def test_shared_compare_wrapper_checks_report_materialization() -> None:
    wrapper = (
        REPO_ROOT / "examples" / "integrations" / "_shared" / "run_invarlock_compare.sh"
    )
    subprocess.run(["bash", "-n", str(wrapper)], check=True)

    text = wrapper.read_text(encoding="utf-8")
    assert "Evaluate completed but did not write the expected report" in text
    assert '[[ ! -s "$report_json" ]]' in text
    assert 'rm -f "$report_json" "$verify_json"' in text
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
    assert "integration_lane_artifact_label" in text
    assert "integration_lane_report_out" in text
    assert "report_out_was_default=1" in text
    assert "report_out_was_default=0" in text
    assert "<artifact-lane>" in text


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
    assert "InvarLock report persistence" in text
    assert "adapter provenance is available" in text


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
    assert "strict container evidence is verified" in text
    assert "this tiny\n`hf_torchao` runtime-load example" in text
    assert "runnable evidence path is the `hf_torchao` subject" in text
    assert "scoped to the configured tiny HF checkpoint" in text
    assert "shared integration evidence" in text
    assert "run_tiny_hf_torchao_int8.sh" in text


def test_peft_lora_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-gpt2-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-gpt2-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "peft-lora-fixture-v1"


def test_peft_lora_helper_isolates_dense_lora_from_quantized_dispatch(
    monkeypatch,
) -> None:
    helper = _load_module(
        PEFT_DIR / "materialize_tiny_peft_lora_subject.py",
        "peft_lora_example_dispatch",
    )

    class DenseModel:
        config = object()
        is_quantized = False

    calls = {"count": 0}

    def fake_get_peft_model(_model, _config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ImportError(
                "cannot import name 'AwqGEMMQuantLinear' from "
                "'gptqmodel.nn_modules.qlinear.gemm_awq'"
            )
        return "peft-model"

    monkeypatch.setattr(
        helper,
        "_disable_quantized_peft_dispatch_for_dense_example",
        lambda: True,
    )

    assert (
        helper._get_dense_peft_model(DenseModel(), object(), fake_get_peft_model)
        == "peft-model"
    )
    assert calls["count"] == 2


def test_torchao_helper_writes_local_jsonl_and_preset(tmp_path: Path) -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_example",
    )
    summary = helper.write_text_fixture(
        tmp_path,
        model_id="/tmp/tiny-llama-baseline",
        rows=6,
        terms_per_row=5,
        seq_len=32,
        preview_n=3,
        final_n=3,
    )

    data_path = Path(summary["data_path"])
    preset_path = Path(summary["preset_path"])
    assert data_path.exists()
    assert preset_path.exists()
    assert (tmp_path / "fixture_summary.json").exists()

    rows = [
        json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    preset = preset_path.read_text(encoding="utf-8")
    assert 'kind: "local_jsonl"' in preset
    assert 'id: "/tmp/tiny-llama-baseline"' in preset
    assert f'file: "{data_path}"' in preset
    assert "preview_n: 3" in preset
    assert summary["format_version"] == "torchao-fixture-v1"


def test_torchao_helper_prefers_non_deprecated_config_version() -> None:
    helper = _load_module(
        TORCHAO_DIR / "prepare_tiny_hf_torchao_fixture.py",
        "torchao_config_helper",
    )

    class _ModernConfig:
        def __init__(self, *, version=None):
            self.version = version

    class _LegacyConfig:
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("unexpected keyword")
            self.version = 1

    modern = helper._torchao_int8_weight_only_config(_ModernConfig)
    legacy = helper._torchao_int8_weight_only_config(_LegacyConfig)

    assert modern.version == 2
    assert legacy.version == 1
