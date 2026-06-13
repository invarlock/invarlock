from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.scripts._support_model_evidence_sweep import load_script_module


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
        "google_flan_t5_base",
    }.issubset(slugs)


def test_model_catalog_gpu_suite_maps_family_specific_presets() -> None:
    mod = load_script_module("model_evidence_sweep")
    specs = {
        lane.slug: lane
        for lane in mod.select_specs(
            mod.MODEL_CATALOG_GPU_SUITE,
            slugs=[
                "microsoft_deberta_v3_base",
                "google_flan_t5_base",
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
    assert specs["google_flan_t5_base"].preset_relpath == (
        "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml"
    )
    assert specs["google_flan_t5_base"].adapter == "hf_seq2seq"
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
        "google_gemma_4_12b_it",
        "huggingfacetb_smollm3_3b",
        "microsoft_phi_4_mini_instruct",
        "tiiuae_falcon_h1r_7b",
        "google_flan_t5_base",
    ]
    adapters = {lane.slug: lane.adapter for lane in specs}
    assert adapters["google_gemma_4_12b_it"] == "hf_multimodal"
    assert adapters["google_flan_t5_base"] == "hf_seq2seq"
    for lane in specs:
        if lane.slug in {
            "huggingfacetb_smollm3_3b",
            "microsoft_phi_4_mini_instruct",
            "tiiuae_falcon_h1r_7b",
        }:
            assert lane.adapter == "hf_causal"
            assert lane.verify_profile == "dev"
        elif lane.slug == "google_gemma_4_12b_it":
            assert lane.verify_profile == "release"
            assert lane.preset_relpath == (
                "configs/presets/multimodal/gemma4_12b_public_vqav2_256.yaml"
            )
            assert lane.vision_text_materialization is not None
        else:
            assert lane.slug == "google_flan_t5_base"
            assert lane.verify_profile == "release"
            assert lane.preset_relpath == (
                "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml"
            )
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
        "facebook_opt_1_3b",
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
        "facebook_opt_1_3b",
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
