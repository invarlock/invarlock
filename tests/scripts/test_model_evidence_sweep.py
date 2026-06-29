from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from tests.scripts._support_model_evidence_sweep import (
    load_script_module,
    write_fake_python,
)


def test_manifest_lane_ids_match_supported_experimental_support_matrix() -> None:
    mod = load_script_module("model_evidence_sweep")

    expected = mod.supported_experimental_lane_ids()
    actual = mod.manifest_lane_ids(mod.CURRENT_SUPPORTED_EXPERIMENTAL_LANES)

    assert actual == expected
    for lane in mod.CURRENT_SUPPORTED_EXPERIMENTAL_LANES:
        assert lane.preset_path.is_file(), lane.preset_relpath


def test_repo_mentioned_gpu_suite_includes_basis_canaries_and_experimental() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = mod.select_specs(
        mod.REPO_MENTIONED_GPU_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=0,
        shard_count=1,
    )

    assert len(specs) == 28
    slugs = {lane.slug for lane in specs}
    assert {
        "gpt2_public",
        "bert_base_uncased_public",
        "roberta_base_public",
        "tiny_gpt2_canary",
        "bert_tiny_canary",
        "mistral_7b_public",
        "ministral3_3b_public",
        "ministral3_8b_public",
        "ministral3_14b_public",
        "tinyllama_1_1b_public",
        "olmo2_13b_public",
        "open_llama_7b_public",
        "falcon_7b_public",
        "qwen2_7b_public",
        "qwen2_5_7b_public",
        "qwen2_5_14b_public",
        "qwen3_8b_public",
        "qwen3_5_9b_public",
        "granite4_1_3b_public",
        "granite4_1_8b_public",
        "deepseek_r1_distill_qwen_7b_public",
        "deepseek_r1_0528_qwen3_8b_public",
        "deepseek_r1_distill_qwen_14b_public",
        "phi4_reasoning_plus_public",
        "flan_t5_base_public",
        "olmo2_7b_public",
        "gemma4_e2b_public",
        "mixtral_8x7b_public",
    }.issubset(slugs)


def test_repo_mentioned_gpu_basis_lanes_use_lane_specific_profiles_and_presets() -> (
    None
):
    mod = load_script_module("model_evidence_sweep")
    basis = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.REPO_MENTIONED_GPU_SUITE,
            slugs=[
                "gpt2_public",
                "bert_base_uncased_public",
                "mistral_7b_public",
                "ministral3_3b_public",
                "ministral3_8b_public",
                "ministral3_14b_public",
                "tinyllama_1_1b_public",
                "olmo2_7b_public",
                "olmo2_13b_public",
                "open_llama_7b_public",
                "falcon_7b_public",
                "qwen2_7b_public",
                "qwen2_5_7b_public",
                "qwen2_5_14b_public",
                "qwen3_8b_public",
                "qwen3_5_9b_public",
                "granite4_1_3b_public",
                "granite4_1_8b_public",
                "mixtral_8x7b_public",
                "gemma4_e2b_public",
                "deepseek_r1_distill_qwen_7b_public",
                "deepseek_r1_0528_qwen3_8b_public",
                "deepseek_r1_distill_qwen_14b_public",
                "phi4_reasoning_plus_public",
                "flan_t5_base_public",
            ],
            lane_ids=[],
            shard_index=0,
            shard_count=1,
        )
    }

    assert basis["gpt2_public"].verify_profile == "dev"
    assert (
        basis["gpt2_public"].preset_relpath
        == "configs/presets/causal_lm/wikitext2_512.yaml"
    )
    assert basis["bert_base_uncased_public"].verify_profile == "dev"
    assert (
        basis["bert_base_uncased_public"].preset_relpath
        == "configs/presets/masked_lm/wikitext2_128.yaml"
    )
    assert basis["mistral_7b_public"].verify_profile == "ci"
    assert (
        basis["mistral_7b_public"].preset_relpath
        == "configs/presets/causal_lm/mistral_7b_512.yaml"
    )
    assert basis["ministral3_3b_public"].verify_profile == "release"
    assert (
        basis["ministral3_3b_public"].preset_relpath
        == "configs/presets/causal_lm/ministral3_3b_512.yaml"
    )
    assert basis["ministral3_8b_public"].verify_profile == "ci"
    assert (
        basis["ministral3_8b_public"].preset_relpath
        == "configs/presets/causal_lm/ministral3_8b_512.yaml"
    )
    assert basis["ministral3_14b_public"].verify_profile == "ci"
    assert (
        basis["ministral3_14b_public"].preset_relpath
        == "configs/presets/causal_lm/ministral3_14b_512.yaml"
    )
    assert basis["tinyllama_1_1b_public"].verify_profile == "ci"
    assert (
        basis["tinyllama_1_1b_public"].preset_relpath
        == "configs/presets/causal_lm/tinyllama_1_1b_512.yaml"
    )
    assert basis["olmo2_7b_public"].verify_profile == "ci"
    assert (
        basis["olmo2_7b_public"].preset_relpath
        == "configs/presets/causal_lm/olmo2_7b_512.yaml"
    )
    assert basis["olmo2_13b_public"].verify_profile == "ci"
    assert (
        basis["olmo2_13b_public"].preset_relpath
        == "configs/presets/causal_lm/olmo2_13b_512.yaml"
    )
    assert basis["open_llama_7b_public"].verify_profile == "release"
    assert (
        basis["open_llama_7b_public"].preset_relpath
        == "configs/presets/causal_lm/open_llama_7b_512.yaml"
    )
    assert basis["falcon_7b_public"].verify_profile == "release"
    assert (
        basis["falcon_7b_public"].preset_relpath
        == "configs/presets/causal_lm/falcon_7b_512.yaml"
    )
    assert basis["qwen2_7b_public"].verify_profile == "ci"
    assert (
        basis["qwen2_7b_public"].preset_relpath
        == "configs/presets/causal_lm/qwen2_7b_512.yaml"
    )
    assert basis["qwen2_5_7b_public"].verify_profile == "ci"
    assert (
        basis["qwen2_5_7b_public"].preset_relpath
        == "configs/presets/causal_lm/qwen2_5_7b_512.yaml"
    )
    assert basis["qwen2_5_14b_public"].verify_profile == "ci"
    assert (
        basis["qwen2_5_14b_public"].preset_relpath
        == "configs/presets/causal_lm/qwen2_5_14b_512.yaml"
    )
    assert basis["qwen3_8b_public"].verify_profile == "ci"
    assert (
        basis["qwen3_8b_public"].preset_relpath
        == "configs/presets/causal_lm/qwen3_8b_512.yaml"
    )
    assert basis["qwen3_5_9b_public"].verify_profile == "ci"
    assert (
        basis["qwen3_5_9b_public"].preset_relpath
        == "configs/presets/causal_lm/qwen3_5_9b_512.yaml"
    )
    assert basis["granite4_1_3b_public"].verify_profile == "release"
    assert (
        basis["granite4_1_3b_public"].preset_relpath
        == "configs/presets/causal_lm/granite4_1_3b_512.yaml"
    )
    assert basis["granite4_1_8b_public"].verify_profile == "release"
    assert (
        basis["granite4_1_8b_public"].preset_relpath
        == "configs/presets/causal_lm/granite4_1_8b_512.yaml"
    )
    assert basis["gemma4_e2b_public"].verify_profile == "release"
    assert (
        basis["gemma4_e2b_public"].preset_relpath
        == "configs/presets/causal_lm/gemma4_e2b_512.yaml"
    )
    assert basis["deepseek_r1_distill_qwen_7b_public"].verify_profile == "ci"
    assert (
        basis["deepseek_r1_distill_qwen_7b_public"].preset_relpath
        == "configs/presets/causal_lm/deepseek_r1_distill_qwen_7b_512.yaml"
    )
    assert basis["deepseek_r1_0528_qwen3_8b_public"].verify_profile == "release"
    assert (
        basis["deepseek_r1_0528_qwen3_8b_public"].preset_relpath
        == "configs/presets/causal_lm/deepseek_r1_0528_qwen3_8b_512.yaml"
    )
    assert basis["deepseek_r1_distill_qwen_14b_public"].verify_profile == "release"
    assert (
        basis["deepseek_r1_distill_qwen_14b_public"].preset_relpath
        == "configs/presets/causal_lm/deepseek_r1_distill_qwen_14b_512.yaml"
    )
    assert basis["phi4_reasoning_plus_public"].verify_profile == "ci"
    assert (
        basis["phi4_reasoning_plus_public"].preset_relpath
        == "configs/presets/causal_lm/phi4_reasoning_plus_512.yaml"
    )
    assert basis["flan_t5_base_public"].verify_profile == "release"
    assert (
        basis["flan_t5_base_public"].preset_relpath
        == "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml"
    )
    assert basis["flan_t5_base_public"].adapter == "hf_seq2seq"


def test_model_evidence_sweep_dry_run_uses_preset_override(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "override-dry-run"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "huggingfacetb_smollm3_3b",
            "--output-root",
            str(output_root),
            "--preset-override",
            "huggingfacetb_smollm3_3b=tmp/smollm3_release.yaml",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    evaluate = payload[0]["evaluate"]
    assert evaluate[evaluate.index("--preset") + 1] == "tmp/smollm3_release.yaml"


def test_model_evidence_sweep_materialized_lane_prepares_preset_from_override(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "override-materialized"
    override_preset = tmp_path / "qwen-linear-only.yaml"
    override_preset.write_text(
        """
model:
  id: Qwen/Qwen3.5-2B
dataset:
  provider:
    kind: vision_text
    path: should-be-replaced.jsonl
guards:
  order: [spectral, rmt]
  spectral:
    module_include_patterns:
      - override.linear_attn.spectral
  rmt:
    module_include_patterns:
      - override.linear_attn.rmt
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "qwen_qwen3_5_2b",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
            "--preset-override",
            f"qwen_qwen3_5_2b={override_preset}",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 1, proc.stderr
    prepared = output_root / "eval" / "qwen_qwen3_5_2b" / "prepared_preset.yaml"
    prepared_data = yaml.safe_load(prepared.read_text(encoding="utf-8"))
    provider = prepared_data["dataset"]["provider"]
    assert provider["kind"] == "vision_text"
    assert provider["path"].endswith("eval/qwen_qwen3_5_2b/dataset/manifest.jsonl")
    assert prepared_data["guards"]["spectral"]["module_include_patterns"] == [
        "override.linear_attn.spectral"
    ]
    assert prepared_data["guards"]["rmt"]["module_include_patterns"] == [
        "override.linear_attn.rmt"
    ]


def test_model_evidence_sweep_rejects_invalid_preset_override(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--output-root",
            str(tmp_path / "bad-override"),
            "--preset-override",
            "missing_separator",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 2
    assert "SLUG=PATH" in proc.stderr


def test_build_prefetch_command_uses_model_profile_tokenizer_resolution() -> None:
    mod = load_script_module("model_evidence_sweep")
    spec = next(
        lane
        for lane in mod.SUITES[mod.REPO_MENTIONED_GPU_SUITE]
        if lane.slug == "bert_tiny_canary"
    )

    command = mod.build_prefetch_command(spec, python_exe=sys.executable)

    assert command[1] == "-c"
    assert "detect_model_profile" in command[2]
    assert "make_tokenizer()" in command[2]
    assert "snapshot_download(model_id)" in command[2]
    assert "AutoModel" not in command[2]
    assert "AutoTokenizer" not in command[2]
    assert command[3] == "prajjwal1/bert-tiny"


def test_model_evidence_sweep_dry_run_uses_lane_specific_verify_profile_when_omitted(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-lane-profile"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gpt2_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    verify = payload[0]["verify"]
    assert verify[verify.index("--profile") + 1] == "dev"
    evaluate = payload[0]["evaluate"]
    assert evaluate[evaluate.index("--assurance") + 1] == "off"


def test_model_evidence_sweep_container_dev_profile_disables_strict_assurance(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-container-dev"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gpt2_public",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    evaluate = payload[0]["evaluate"]
    assert evaluate[evaluate.index("--profile") + 1] == "dev"
    assert evaluate[evaluate.index("--assurance") + 1] == "off"


def test_select_specs_sharding_is_stable() -> None:
    mod = load_script_module("model_evidence_sweep")

    first_shard = mod.select_specs(
        mod.DEFAULT_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=0,
        shard_count=3,
    )
    second_shard = mod.select_specs(
        mod.DEFAULT_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=1,
        shard_count=3,
    )

    assert first_shard == []
    assert second_shard == []


def test_lane_resource_preflight_warns_for_underprovisioned_moe() -> None:
    mod = load_script_module("model_evidence_sweep")
    spec = next(
        lane
        for lane in mod.SUITES[mod.SUPPORT_MATRIX_BACKLOG_GPU_SUITE]
        if lane.slug == "mistralai_mixtral_8x7b_v0_1"
    )

    underprovisioned = mod.lane_resource_preflight(
        spec,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        device="cuda",
    )

    assert underprovisioned is not None
    assert underprovisioned["ok"] is False
    assert underprovisioned["visible_cuda_devices"] == 1
    assert underprovisioned["recommended_min_gpus_80gb"] >= 3
    assert "recommended minimum" in underprovisioned["warning"]

    provisioned = mod.lane_resource_preflight(
        spec,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        device="cuda",
    )

    assert provisioned is not None
    assert provisioned["ok"] is True
    assert "warning" not in provisioned


def test_model_evidence_sweep_logs_resource_preflight_warning(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "evidence-moe-preflight"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "mistralai_mixtral_8x7b_v0_1",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert proc.returncode == 1, proc.stderr
    status_log = (output_root / "status.log").read_text(encoding="utf-8")
    assert "WARN mistralai_mixtral_8x7b_v0_1 resource_preflight=" in status_log
    assert "visible CUDA device count 1" in status_log


def test_model_evidence_sweep_returns_failure_when_verify_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "evidence"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 1, proc.stderr
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "container"
    assert summary["ok"] is False
    assert len(summary["results"]) == 1
    result = summary["results"][0]
    assert result["slug"] == "gemma4_e2b_public"
    assert result["evaluate_exit"] == 0
    assert result["verify_exit"] == 1
    assert result["detail"] == "policy_fail"
    assert result["ok"] is False
    assert (output_root / "eval" / "gemma4_e2b_public" / "verify.json").is_file()


def test_model_evidence_sweep_verify_failure_detail_is_sanitized(
    tmp_path: Path,
) -> None:
    mod = load_script_module("model_evidence_sweep")
    verify_path = tmp_path / "verify.json"

    verify_path.write_text(
        json.dumps({"summary": {"reason": "policy-fail"}}),
        encoding="utf-8",
    )
    assert mod._verify_failure_detail(verify_path) == "policy_fail"

    verify_path.write_text(
        json.dumps({"summary": {"reason": "../policy fail"}}),
        encoding="utf-8",
    )
    assert mod._verify_failure_detail(verify_path) is None


def test_model_evidence_sweep_evaluate_failure_detail_classifies_no_samples() -> None:
    mod = load_script_module("model_evidence_sweep")

    assert (
        mod._evaluate_failure_detail(
            "[FAIL] [INVARLOCK:E306] NO-SAMPLES: vision_text produced no samples"
        )
        == "no_samples"
    )


def test_model_evidence_sweep_evaluate_failure_detail_classifies_dataset_cache() -> (
    None
):
    mod = load_script_module("model_evidence_sweep")

    assert (
        mod._evaluate_failure_detail(
            "Couldn't find cache for Salesforce/wikitext for config "
            "'wikitext-103-v1'. Available configs in the cache: "
            "['wikitext-2-raw-v1']"
        )
        == "dataset_cache_missing"
    )


def test_model_evidence_sweep_host_mode_rejects_ci_profile(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-host-ci"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--execution-mode",
            "host",
            "--profile",
            "ci",
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 2
    assert "incompatible with --profile ci/release" in proc.stderr
