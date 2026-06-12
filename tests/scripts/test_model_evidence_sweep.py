from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scripts._support_model_evidence_sweep import (
    load_script_module,
    write_fake_python,
    write_flaky_fake_python,
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

    assert len(specs) == 19
    slugs = {lane.slug for lane in specs}
    assert {
        "gpt2_public",
        "bert_base_uncased_public",
        "roberta_base_public",
        "tiny_gpt2_canary",
        "bert_tiny_canary",
        "mistral_7b_public",
        "ministral3_8b_public",
        "ministral3_14b_public",
        "tinyllama_1_1b_public",
        "olmo2_13b_public",
        "qwen2_7b_public",
        "qwen2_5_7b_public",
        "qwen2_5_14b_public",
        "qwen3_8b_public",
        "qwen3_5_9b_public",
        "deepseek_r1_distill_qwen_7b_public",
        "phi4_reasoning_plus_public",
        "gemma4_e2b",
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
                "ministral3_8b_public",
                "ministral3_14b_public",
                "tinyllama_1_1b_public",
                "olmo2_13b_public",
                "qwen2_7b_public",
                "qwen2_5_7b_public",
                "qwen2_5_14b_public",
                "qwen3_8b_public",
                "qwen3_5_9b_public",
                "deepseek_r1_distill_qwen_7b_public",
                "phi4_reasoning_plus_public",
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
    assert basis["olmo2_13b_public"].verify_profile == "ci"
    assert (
        basis["olmo2_13b_public"].preset_relpath
        == "configs/presets/causal_lm/olmo2_13b_512.yaml"
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
    assert basis["deepseek_r1_distill_qwen_7b_public"].verify_profile == "ci"
    assert (
        basis["deepseek_r1_distill_qwen_7b_public"].preset_relpath
        == "configs/presets/causal_lm/deepseek_r1_distill_qwen_7b_512.yaml"
    )
    assert basis["phi4_reasoning_plus_public"].verify_profile == "ci"
    assert (
        basis["phi4_reasoning_plus_public"].preset_relpath
        == "configs/presets/causal_lm/phi4_reasoning_plus_512.yaml"
    )


def test_model_catalog_gpu_suite_covers_public_catalog_representative_models() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = mod.select_specs(
        mod.MODEL_CATALOG_GPU_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=0,
        shard_count=1,
    )

    slugs = {lane.slug for lane in specs}
    assert len(specs) == len(slugs)
    assert {
        "openai_community_gpt2",
        "bert_base_uncased",
        "roberta_base",
        "mistralai_mistral_7b_v0_1",
        "openai_gpt_oss_20b",
        "qwen_qwen2_7b",
        "qwen_qwen2_5_7b",
        "microsoft_phi_4_reasoning_plus",
        "google_gemma_4_e4b_it",
        "facebook_bart_base",
    }.issubset(slugs)


def test_model_catalog_gpu_suite_maps_family_specific_presets() -> None:
    mod = load_script_module("model_evidence_sweep")
    specs = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.MODEL_CATALOG_GPU_SUITE,
            slugs=[
                "microsoft_deberta_v3_base",
                "t5_small",
                "google_gemma_4_e4b_it",
                "mistralai_mixtral_8x7b_v0_1",
                "openlm_research_open_llama_7b",
                "facebook_opt_1_3b",
                "tiiuae_falcon_7b",
                "thudm_glm_4_9b_chat",
                "ibm_granite_granite_4_1_8b",
                "ibm_granite_granite_4_1_3b",
                "huggingfacetb_smollm3_3b",
                "microsoft_phi_4_mini_instruct",
                "deepseek_ai_deepseek_r1_distill_qwen_14b",
                "deepseek_ai_deepseek_r1_0528_qwen3_8b",
                "tiiuae_falcon_h1r_7b",
            ],
            lane_ids=[],
            shard_index=0,
            shard_count=1,
        )
    }

    assert specs["microsoft_deberta_v3_base"].preset_relpath == (
        "configs/presets/masked_lm/wikitext2_128.yaml"
    )
    assert specs["microsoft_deberta_v3_base"].adapter == "hf_mlm"
    assert specs["t5_small"].preset_relpath == "configs/presets/seq2seq/synth_128.yaml"
    assert specs["t5_small"].adapter == "hf_seq2seq"
    assert specs["google_gemma_4_e4b_it"].preset_relpath == (
        "configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml"
    )
    assert specs["google_gemma_4_e4b_it"].adapter == "hf_multimodal"
    assert specs["mistralai_mixtral_8x7b_v0_1"].preset_relpath == (
        "configs/presets/causal_lm/wikitext2_512.yaml"
    )
    assert specs["mistralai_mixtral_8x7b_v0_1"].adapter == "auto"
    assert specs["openlm_research_open_llama_7b"].preset_relpath == (
        "configs/presets/causal_lm/open_llama_7b_512.yaml"
    )
    assert specs["openlm_research_open_llama_7b"].adapter == "hf_causal"
    assert specs["facebook_opt_1_3b"].preset_relpath == (
        "configs/presets/causal_lm/opt_1_3b_512.yaml"
    )
    assert specs["facebook_opt_1_3b"].adapter == "hf_causal"
    assert specs["tiiuae_falcon_7b"].preset_relpath == (
        "configs/presets/causal_lm/falcon_7b_512.yaml"
    )
    assert specs["tiiuae_falcon_7b"].adapter == "hf_causal"
    assert specs["thudm_glm_4_9b_chat"].preset_relpath == (
        "configs/presets/causal_lm/glm4_9b_chat_512.yaml"
    )
    assert specs["thudm_glm_4_9b_chat"].adapter == "hf_causal"
    assert specs["ibm_granite_granite_4_1_8b"].preset_relpath == (
        "configs/presets/causal_lm/granite4_1_8b_512.yaml"
    )
    assert specs["ibm_granite_granite_4_1_8b"].adapter == "hf_causal"
    assert specs["ibm_granite_granite_4_1_3b"].preset_relpath == (
        "configs/presets/causal_lm/granite4_1_3b_512.yaml"
    )
    assert specs["huggingfacetb_smollm3_3b"].preset_relpath == (
        "configs/presets/causal_lm/smollm3_3b_512.yaml"
    )
    assert specs["microsoft_phi_4_mini_instruct"].preset_relpath == (
        "configs/presets/causal_lm/phi4_mini_512.yaml"
    )
    assert specs["deepseek_ai_deepseek_r1_distill_qwen_14b"].preset_relpath == (
        "configs/presets/causal_lm/deepseek_r1_distill_qwen_14b_512.yaml"
    )
    assert specs["deepseek_ai_deepseek_r1_0528_qwen3_8b"].preset_relpath == (
        "configs/presets/causal_lm/deepseek_r1_0528_qwen3_8b_512.yaml"
    )
    assert specs["tiiuae_falcon_h1r_7b"].preset_relpath == (
        "configs/presets/causal_lm/falcon_h1r_7b_512.yaml"
    )


def test_support_matrix_backlog_gpu_suite_targets_prepared_candidate_rows() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = mod.select_specs(
        mod.SUPPORT_MATRIX_BACKLOG_GPU_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=0,
        shard_count=1,
    )

    assert [lane.slug for lane in specs] == [
        "mistralai_ministral_3_3b_instruct_2512_bf16",
        "ibm_granite_granite_4_1_8b",
        "ibm_granite_granite_4_1_3b",
        "huggingfacetb_smollm3_3b",
        "microsoft_phi_4_mini_instruct",
        "deepseek_ai_deepseek_r1_distill_qwen_14b",
        "deepseek_ai_deepseek_r1_0528_qwen3_8b",
        "tiiuae_falcon_h1r_7b",
    ]
    for lane in specs:
        assert lane.adapter == "hf_causal"
        assert lane.verify_profile == "dev"
        assert lane.preset_path.is_file(), lane.preset_relpath


def test_promotion_gap_gpu_suite_targets_repo_prepared_blocked_lanes() -> None:
    mod = load_script_module("model_evidence_sweep")

    specs = mod.select_specs(
        mod.PROMOTION_GAP_GPU_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=0,
        shard_count=1,
    )

    assert [lane.slug for lane in specs] == [
        "openlm_research_open_llama_7b",
        "facebook_opt_1_3b",
        "tiiuae_falcon_7b",
        "thudm_glm_4_9b_chat",
        "distilbert_base_uncased",
    ]
    distilbert = specs[-1]
    assert distilbert.adapter == "hf_mlm"
    assert distilbert.preset_relpath == (
        "configs/presets/masked_lm/distilbert_base_uncased_128.yaml"
    )
    for lane in specs:
        assert lane.verify_profile == "dev"
        assert lane.preset_path.is_file(), lane.preset_relpath


def test_model_evidence_sweep_dry_run_supports_promotion_gap_suite_candidates(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "promotion-gap-dry-run"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "promotion-gap-gpu",
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
    slugs = [entry["slug"] for entry in payload]
    assert slugs == [
        "openlm_research_open_llama_7b",
        "facebook_opt_1_3b",
        "tiiuae_falcon_7b",
        "thudm_glm_4_9b_chat",
        "distilbert_base_uncased",
    ]
    distilbert = payload[-1]
    assert distilbert["evaluate"][distilbert["evaluate"].index("--baseline") + 1] == (
        "distilbert-base-uncased"
    )
    assert (
        distilbert["evaluate"][distilbert["evaluate"].index("--baseline-adapter") + 1]
        == "hf_mlm"
    )
    assert (
        distilbert["evaluate"][distilbert["evaluate"].index("--subject-adapter") + 1]
        == "hf_mlm"
    )


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

    shard = mod.select_specs(
        mod.DEFAULT_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=1,
        shard_count=3,
    )

    assert [lane.slug for lane in shard] == [
        "gemma4_e2b",
    ]


def test_model_evidence_sweep_host_mode_emits_explicit_runtime_flags(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-host"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "olmo2_7b",
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
    assert len(payload) == 1
    assert payload[0]["execution_mode"] == "host"
    assert payload[0]["prefetch"][-1] == "allenai/OLMo-2-1124-7B"
    assert payload[0]["prefetch"][1] == "-c"
    assert "--execution-mode" in payload[0]["evaluate"]
    assert (
        payload[0]["evaluate"][payload[0]["evaluate"].index("--execution-mode") + 1]
        == "host"
    )
    assert payload[0]["evaluate"][payload[0]["evaluate"].index("--assurance") + 1] == (
        "off"
    )
    assert "--allow-host-execution" not in payload[0]["evaluate"]
    assert "--runtime-provenance" in payload[0]["verify"]
    assert (
        payload[0]["verify"][payload[0]["verify"].index("--runtime-provenance") + 1]
        == "host"
    )
    assert payload[0]["verify"][payload[0]["verify"].index("--profile") + 1] == "dev"
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert payload[0]["evaluate"][preset_idx] == str(
        repo_root / "configs/presets/causal_lm/olmo2_7b_512.yaml"
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "host"


def test_model_evidence_sweep_host_mode_prefetches_before_evaluate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "evidence-host-prefetch"
    log_path = tmp_path / "fake-python.log"

    env = dict(os.environ)
    env["FAKE_PYTHON_LOG"] = str(log_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "olmo2_7b",
            "--execution-mode",
            "host",
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
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert "allenai/OLMo-2-1124-7B" in invocations[0]
    assert "-m invarlock evaluate" in invocations[1]


def test_model_evidence_sweep_retries_evaluate_once_after_sigterm(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "flaky-fake-python"
    write_flaky_fake_python(fake_python)
    output_root = tmp_path / "evidence-host-retry"
    log_path = tmp_path / "fake-python-retry.log"
    state_path = tmp_path / "retry-state"

    env = dict(os.environ)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    env["FAKE_PYTHON_STATE"] = str(state_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "olmo2_7b",
            "--execution-mode",
            "host",
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

    assert proc.returncode == 0, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    evaluate_invocations = [
        line for line in invocations if "-m invarlock evaluate" in line
    ]
    verify_invocations = [line for line in invocations if "-m invarlock verify" in line]
    assert len(evaluate_invocations) == 2
    assert len(verify_invocations) == 1
    lane_log = (output_root / "logs" / "olmo2_7b.log").read_text(encoding="utf-8")
    assert "evaluate exited with -15; retrying once." in lane_log


def test_model_evidence_sweep_marks_gated_prefetch_as_skipped(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "gated-fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  echo "GatedRepoError: Cannot access gated repo" >&2
  exit 1
fi
echo "unexpected invocation" >&2
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    output_root = tmp_path / "evidence-host-gated"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "model-catalog-gpu",
            "--slug",
            "google_gemma_3_4b_it",
            "--execution-mode",
            "host",
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

    assert proc.returncode == 0, proc.stderr
    summary_tsv = (output_root / "summary.tsv").read_text(encoding="utf-8")
    assert "status\tdetail" in summary_tsv
    assert "skipped\tgated_repo" in summary_tsv
    status_log = (output_root / "status.log").read_text(encoding="utf-8")
    assert "status=skipped detail=gated_repo" in status_log


def test_build_evaluate_command_uses_container_safe_repo_relative_paths(
    tmp_path: Path,
) -> None:
    mod = load_script_module("model_evidence_sweep")
    spec = next(
        lane
        for lane in mod.CURRENT_SUPPORTED_EXPERIMENTAL_LANES
        if lane.slug == "olmo2_7b"
    )
    external_output_root = tmp_path / "external-container-evidence"
    execution_root = mod._execution_root(
        external_output_root, execution_mode="container"
    )
    lane_root = execution_root / "eval" / spec.slug

    command = mod.build_evaluate_command(
        spec,
        python_exe=sys.executable,
        profile="ci",
        device="cuda",
        execution_mode="container",
        lane_root=lane_root,
    )

    out_idx = command.index("--out") + 1
    report_idx = command.index("--report-out") + 1
    base = f"tmp/model_evidence_container/{execution_root.name}"
    assert command[out_idx] == f"{base}/eval/olmo2_7b/runs"
    assert command[report_idx] == f"{base}/eval/olmo2_7b/report"


def test_runtime_env_preserves_container_default_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = load_script_module("model_evidence_sweep")
    config_root = tmp_path / "config-root"
    config_root.mkdir()
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("INVARLOCK_ALLOW_NETWORK", raising=False)
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "auto")

    env = mod.runtime_env()

    assert env["PYTHONPATH"] == str(mod.REPO_ROOT / "src")
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert env["INVARLOCK_CONFIG_ROOT"] == str(config_root)
    assert env["HF_HOME"] == str(hf_home)
    assert env["TMPDIR"] == str(tmpdir)
    assert env["INVARLOCK_EXPORT_DIR"] == str(export_dir)
    assert env["INVARLOCK_STORE_EVAL_WINDOWS"] == "1"
    assert env["INVARLOCK_SNAPSHOT_MODE"] == "auto"


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
            "--slug",
            "olmo2_7b",
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
    assert result["slug"] == "olmo2_7b"
    assert result["evaluate_exit"] == 0
    assert result["verify_exit"] == 1
    assert result["ok"] is False
    assert (output_root / "eval" / "olmo2_7b" / "verify.json").is_file()


def test_model_evidence_sweep_host_mode_rejects_ci_profile(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-host-ci"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "olmo2_7b",
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


def test_model_evidence_sweep_container_mode_publishes_external_output_root(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "external-container-evidence"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "olmo2_7b",
            "--execution-mode",
            "container",
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
    published_lane = output_root / "eval" / "olmo2_7b"
    assert (published_lane / "report" / "evaluation.report.json").is_file()
    assert (published_lane / "verify.json").is_file()
