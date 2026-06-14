"""Lane catalog and selection helpers for model-evidence sweeps."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORT_MATRIX_PATH = REPO_ROOT / "contracts" / "support_matrix.json"
MODEL_FAMILY_CATALOG_PATH = REPO_ROOT / "contracts" / "model_family_catalog.json"
DEFAULT_SUITE = "current-supported-experimental"
REPO_MENTIONED_GPU_SUITE = "repo-mentioned-gpu"
MODEL_CATALOG_GPU_SUITE = "model-catalog-gpu"
PROMOTION_GAP_GPU_SUITE = "promotion-gap-gpu"
SUPPORT_MATRIX_BACKLOG_GPU_SUITE = "support-matrix-backlog-gpu"
EXECUTION_MODES = ("container", "host")
GPU_80GB_USABLE_MEMORY_GB = 72.0
DEFAULT_DENSE_RUNTIME_MULTIPLIER = 1.35
DEFAULT_MOE_RUNTIME_MULTIPLIER = 2.0


def estimate_model_weight_gb(model_id: str) -> float | None:
    """Return an approximate BF16 checkpoint footprint for planning GPU groups."""
    model_lower = model_id.lower()
    if "mixtral" in model_lower or "8x7b" in model_lower:
        return 90.0
    if "qwen3-30b-a3b" in model_lower or "30b-a3b" in model_lower:
        return 60.0
    if "gemma-4-26b-a4b" in model_lower or "26b-a4b" in model_lower:
        return 52.0
    if "olmoe-1b-7b" in model_lower or "1b-7b" in model_lower:
        return 14.0

    params_b = [
        float(match.group(1).replace("_", "."))
        for match in re.finditer(r"(\d+(?:[._]\d+)?)\s*b\b", model_lower)
    ]
    if not params_b:
        return None
    return max(params_b) * 2.0


def _is_moe_model(model_id: str) -> bool:
    model_lower = model_id.lower()
    return any(
        token in model_lower
        for token in ("moe", "mixtral", "8x7b", "a3b", "a4b", "olmoe")
    )


def lane_resource_estimate(model_id: str) -> dict[str, object] | None:
    """Build an approximate memory planning hint for evidence sweeps."""
    weights_gb = estimate_model_weight_gb(model_id)
    if weights_gb is None:
        return None
    is_moe = _is_moe_model(model_id)
    multiplier = (
        DEFAULT_MOE_RUNTIME_MULTIPLIER if is_moe else DEFAULT_DENSE_RUNTIME_MULTIPLIER
    )
    recommended = max(
        1, math.ceil((weights_gb * multiplier) / GPU_80GB_USABLE_MEMORY_GB)
    )
    return {
        "schema": "invarlock/model-evidence-resource-estimate-v1",
        "estimated_weight_gb_bf16": round(weights_gb, 2),
        "runtime_multiplier": multiplier,
        "recommended_min_gpus_80gb": recommended,
        "gpu_80gb_usable_memory_gb": GPU_80GB_USABLE_MEMORY_GB,
        "moe_model": is_moe,
        "basis": "heuristic_from_model_name_for_launch_planning",
    }


@dataclass(frozen=True)
class EvidenceLane:
    slug: str
    lane_id: str
    family: str
    model_id: str
    preset_relpath: str
    adapter: str = "auto"
    verify_profile: str = "ci"
    vision_text_materialization: dict[str, object] | None = None

    @property
    def preset_path(self) -> Path:
        return REPO_ROOT / self.preset_relpath

    def preset_arg(self, *, execution_mode: str) -> str:
        if execution_mode == "container":
            return self.preset_relpath
        return str(self.preset_path)

    def to_manifest_entry(self) -> dict[str, object]:
        entry = {
            "slug": self.slug,
            "lane_id": self.lane_id,
            "family": self.family,
            "model_id": self.model_id,
            "preset": self.preset_relpath,
            "adapter": self.adapter,
            "verify_profile": self.verify_profile,
        }
        if self.vision_text_materialization:
            entry["vision_text_materialization"] = self.vision_text_materialization
        resource_estimate = lane_resource_estimate(self.model_id)
        if resource_estimate is not None:
            entry["resource_estimate"] = resource_estimate
        return entry


CURRENT_SUPPORTED_EXPERIMENTAL_LANES: tuple[EvidenceLane, ...] = ()

CURRENT_PUBLISHED_BASIS_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="gpt2_public",
        lane_id="published-gpt2-causal-hf",
        family="GPT-2 causal LM",
        model_id="gpt2",
        preset_relpath="configs/presets/causal_lm/wikitext2_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="bert_base_uncased_public",
        lane_id="published-bert-base-uncased-mlm-hf",
        family="BERT / RoBERTa MLM",
        model_id="bert-base-uncased",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="roberta_base_public",
        lane_id="published-roberta-base-mlm-hf",
        family="BERT / RoBERTa MLM",
        model_id="roberta-base",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="mistral_7b_public",
        lane_id="mistral-7b-causal-hf",
        family="Mistral 7B causal LM",
        model_id="mistralai/Mistral-7B-v0.1",
        preset_relpath="configs/presets/causal_lm/mistral_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="ministral3_8b_public",
        lane_id="ministral-3-8b-text-causal-hf",
        family="Ministral 3 8B causal LM (text-only eval)",
        model_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        preset_relpath="configs/presets/causal_lm/ministral3_8b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="ministral3_3b_public",
        lane_id="ministral-3-3b-text-causal-hf",
        family="Ministral 3 3B causal LM (text-only eval)",
        model_id="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        preset_relpath="configs/presets/causal_lm/ministral3_3b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="ministral3_14b_public",
        lane_id="ministral-3-14b-text-causal-hf",
        family="Ministral 3 14B causal LM (text-only eval)",
        model_id="mistralai/Ministral-3-14B-Instruct-2512-BF16",
        preset_relpath="configs/presets/causal_lm/ministral3_14b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="tinyllama_1_1b_public",
        lane_id="tinyllama-1-1b-causal-hf",
        family="TinyLlama 1.1B causal LM",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        preset_relpath="configs/presets/causal_lm/tinyllama_1_1b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="gemma4_e2b_public",
        lane_id="gemma4-e2b-text-causal-hf",
        family="Gemma 4 E2B causal LM (text-only eval)",
        model_id="google/gemma-4-E2B-it",
        preset_relpath="configs/presets/causal_lm/gemma4_e2b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="olmo2_7b_public",
        lane_id="olmo-2-7b-causal-hf",
        family="OLMo 2 7B causal LM",
        model_id="allenai/OLMo-2-1124-7B",
        preset_relpath="configs/presets/causal_lm/olmo2_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="olmo2_13b_public",
        lane_id="olmo-2-13b-causal-hf",
        family="OLMo 2 13B causal LM",
        model_id="allenai/OLMo-2-1124-13B-Instruct",
        preset_relpath="configs/presets/causal_lm/olmo2_13b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="open_llama_7b_public",
        lane_id="open-llama-7b-causal-hf",
        family="OpenLLaMA 7B causal LM",
        model_id="openlm-research/open_llama_7b",
        preset_relpath="configs/presets/causal_lm/open_llama_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="falcon_7b_public",
        lane_id="falcon-7b-causal-hf",
        family="Falcon 7B causal LM",
        model_id="tiiuae/falcon-7b",
        preset_relpath="configs/presets/causal_lm/falcon_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="mixtral_8x7b_public",
        lane_id="mixtral-8x7b-moe-causal-hf",
        family="Mixtral 8x7B MoE causal LM",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        preset_relpath="configs/presets/causal_lm/mixtral_8x7b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="qwen2_7b_public",
        lane_id="qwen2-7b-causal-hf",
        family="Qwen2 7B causal LM",
        model_id="Qwen/Qwen2-7B",
        preset_relpath="configs/presets/causal_lm/qwen2_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="qwen2_5_7b_public",
        lane_id="qwen2-5-7b-causal-hf",
        family="Qwen2.5 7B causal LM",
        model_id="Qwen/Qwen2.5-7B",
        preset_relpath="configs/presets/causal_lm/qwen2_5_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="qwen2_5_14b_public",
        lane_id="qwen2-5-14b-causal-hf",
        family="Qwen2.5 14B causal LM",
        model_id="Qwen/Qwen2.5-14B",
        preset_relpath="configs/presets/causal_lm/qwen2_5_14b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="qwen3_8b_public",
        lane_id="qwen3-causal-hf",
        family="Qwen3 causal LM",
        model_id="Qwen/Qwen3-8B",
        preset_relpath="configs/presets/causal_lm/qwen3_8b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="qwen3_5_9b_public",
        lane_id="qwen3-5-causal-hf",
        family="Qwen3.5 causal LM",
        model_id="Qwen/Qwen3.5-9B",
        preset_relpath="configs/presets/causal_lm/qwen3_5_9b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="granite4_1_3b_public",
        lane_id="granite-4-1-3b-causal-hf",
        family="Granite 4.1 3B causal LM",
        model_id="ibm-granite/granite-4.1-3b",
        preset_relpath="configs/presets/causal_lm/granite4_1_3b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="granite4_1_8b_public",
        lane_id="granite-4-1-8b-causal-hf",
        family="Granite 4.1 8B causal LM",
        model_id="ibm-granite/granite-4.1-8b",
        preset_relpath="configs/presets/causal_lm/granite4_1_8b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="deepseek_r1_distill_qwen_7b_public",
        lane_id="deepseek-r1-distill-qwen-causal-hf",
        family="DeepSeek-R1-Distill-Qwen causal LM",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        preset_relpath="configs/presets/causal_lm/deepseek_r1_distill_qwen_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="deepseek_r1_0528_qwen3_8b_public",
        lane_id="deepseek-r1-0528-qwen3-8b-causal-hf",
        family="DeepSeek-R1-0528-Qwen3 8B causal LM",
        model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        preset_relpath="configs/presets/causal_lm/deepseek_r1_0528_qwen3_8b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="deepseek_r1_distill_qwen_14b_public",
        lane_id="deepseek-r1-distill-qwen-14b-causal-hf",
        family="DeepSeek-R1-Distill-Qwen 14B causal LM",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        preset_relpath="configs/presets/causal_lm/deepseek_r1_distill_qwen_14b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="phi4_reasoning_plus_public",
        lane_id="phi-4-text-causal-hf",
        family="Phi-4 causal LM (text-only eval)",
        model_id="microsoft/Phi-4-reasoning-plus",
        preset_relpath="configs/presets/causal_lm/phi4_reasoning_plus_512.yaml",
        adapter="hf_causal",
        verify_profile="ci",
    ),
    EvidenceLane(
        slug="flan_t5_base_public",
        lane_id="flan-t5-base-seq2seq-hf",
        family="FLAN-T5 base seq2seq LM",
        model_id="google/flan-t5-base",
        preset_relpath="configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml",
        adapter="hf_seq2seq",
        verify_profile="release",
    ),
)

DOCUMENTED_SMOKE_CANARY_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="tiny_gpt2_canary",
        lane_id="smoke-tiny-gpt2-causal-hf",
        family="GPT-2 causal LM smoke canary",
        model_id="sshleifer/tiny-gpt2",
        preset_relpath="configs/presets/causal_lm/wikitext2_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="bert_tiny_canary",
        lane_id="smoke-bert-tiny-mlm-hf",
        family="BERT MLM smoke canary",
        model_id="prajjwal1/bert-tiny",
        preset_relpath="configs/presets/masked_lm/wikitext2_128.yaml",
        adapter="hf_mlm",
        verify_profile="dev",
    ),
)

MODEL_FAMILY_CATALOG_SECTIONS = (
    "declared_support",
    "implemented_coverage",
    "usage_only",
    "recommended_additions",
)

CATALOG_PRESET_OVERRIDES: dict[str, tuple[str, str]] = {
    "distilbert-base-uncased": (
        "configs/presets/masked_lm/distilbert_base_uncased_128.yaml",
        "hf_mlm",
    ),
    "openlm-research/open_llama_7b": (
        "configs/presets/causal_lm/open_llama_7b_512.yaml",
        "hf_causal",
    ),
    "facebook/opt-1.3b": (
        "configs/presets/causal_lm/opt_1_3b_512.yaml",
        "hf_causal",
    ),
    "tiiuae/falcon-7b": (
        "configs/presets/causal_lm/falcon_7b_512.yaml",
        "hf_causal",
    ),
    "THUDM/glm-4-9b-chat": (
        "configs/presets/causal_lm/glm4_9b_chat_512.yaml",
        "hf_causal",
    ),
    "mistralai/Ministral-3-3B-Instruct-2512-BF16": (
        "configs/presets/causal_lm/ministral3_3b_512.yaml",
        "hf_causal",
    ),
    "google/gemma-4-12B-it": (
        "configs/presets/multimodal/gemma4_12b_vision_text_256.yaml",
        "hf_multimodal",
    ),
    "Qwen/Qwen3.5-4B": (
        "configs/presets/multimodal/qwen3_5_4b_public_vqav2_256.yaml",
        "hf_multimodal",
    ),
    "Qwen/Qwen3.5-2B": (
        "configs/presets/multimodal/qwen3_5_2b_public_vqav2_256.yaml",
        "hf_multimodal",
    ),
    "google/gemma-3n-E4B-it": (
        "configs/presets/multimodal/gemma3n_e4b_public_vqav2_256.yaml",
        "hf_multimodal",
    ),
    "google/gemma-3-4b-it": (
        "configs/presets/multimodal/gemma3_4b_it_public_vqav2_256.yaml",
        "hf_multimodal",
    ),
    "ibm-granite/granite-4.1-8b": (
        "configs/presets/causal_lm/granite4_1_8b_512.yaml",
        "hf_causal",
    ),
    "ibm-granite/granite-4.1-3b": (
        "configs/presets/causal_lm/granite4_1_3b_512.yaml",
        "hf_causal",
    ),
    "HuggingFaceTB/SmolLM3-3B": (
        "configs/presets/causal_lm/smollm3_3b_512.yaml",
        "hf_causal",
    ),
    "microsoft/Phi-4-mini-instruct": (
        "configs/presets/causal_lm/phi4_mini_512.yaml",
        "hf_causal",
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (
        "configs/presets/causal_lm/deepseek_r1_distill_qwen_14b_512.yaml",
        "hf_causal",
    ),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": (
        "configs/presets/causal_lm/deepseek_r1_0528_qwen3_8b_512.yaml",
        "hf_causal",
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (
        "configs/presets/causal_lm/qwen3_30b_a3b_instruct_2507_512.yaml",
        "hf_causal",
    ),
    "mistralai/Mixtral-8x7B-v0.1": (
        "configs/presets/causal_lm/mixtral_8x7b_512.yaml",
        "hf_causal",
    ),
    "tiiuae/Falcon-H1R-7B": (
        "configs/presets/causal_lm/falcon_h1r_7b_512.yaml",
        "hf_causal",
    ),
    "allenai/OLMoE-1B-7B-0924": (
        "configs/presets/causal_lm/olmoe_1b_7b_0924_512.yaml",
        "hf_causal",
    ),
    "google/gemma-4-26B-A4B-it": (
        "configs/presets/multimodal/gemma4_26b_a4b_public_vqav2_256.yaml",
        "hf_multimodal",
    ),
    "google/flan-t5-base": (
        "configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml",
        "hf_seq2seq",
    ),
}


def _public_vqav2_materialization() -> dict[str, object]:
    return {
        "dataset": "Multimodal-Fatima/VQAv2_sample_validation",
        "split": "validation",
        "revision": "99487d2651df3799002b2fb3e455741744514a02",
        "max_samples": 800,
        "image_field": "image",
        "prompt_field": "question",
        "answer_field": "multiple_choice_answer",
        "answers_field": "answers",
        "id_field": "question_id",
        "prompt_template": "{question}\nAnswer with a short phrase.",
        "image_format": "png",
    }


SUPPORT_MATRIX_BACKLOG_GPU_LANES: tuple[EvidenceLane, ...] = (
    EvidenceLane(
        slug="google_gemma_4_12b_it",
        lane_id="gemma4-12b-any-to-any-hf",
        family="Gemma 4 12B any-to-any LM",
        model_id="google/gemma-4-12B-it",
        preset_relpath="configs/presets/multimodal/gemma4_12b_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
    EvidenceLane(
        slug="qwen_qwen3_5_4b",
        lane_id="qwen3-5-4b-image-text-hf",
        family="Qwen3.5 4B image-text LM",
        model_id="Qwen/Qwen3.5-4B",
        preset_relpath="configs/presets/multimodal/qwen3_5_4b_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
    EvidenceLane(
        slug="qwen_qwen3_5_2b",
        lane_id="qwen3-5-2b-image-text-hf",
        family="Qwen3.5 2B image-text LM",
        model_id="Qwen/Qwen3.5-2B",
        preset_relpath="configs/presets/multimodal/qwen3_5_2b_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
    EvidenceLane(
        slug="google_gemma_3n_e4b_it",
        lane_id="gemma3n-e4b-image-text-hf",
        family="Gemma 3n E4B image-text LM",
        model_id="google/gemma-3n-E4B-it",
        preset_relpath="configs/presets/multimodal/gemma3n_e4b_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
    EvidenceLane(
        slug="google_gemma_3_4b_it",
        lane_id="gemma3-4b-it-image-text-hf",
        family="Gemma 3 4B IT image-text LM",
        model_id="google/gemma-3-4b-it",
        preset_relpath="configs/presets/multimodal/gemma3_4b_it_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
    EvidenceLane(
        slug="huggingfacetb_smollm3_3b",
        lane_id="smollm3-3b-causal-hf",
        family="SmolLM3 3B causal LM",
        model_id="HuggingFaceTB/SmolLM3-3B",
        preset_relpath="configs/presets/causal_lm/smollm3_3b_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="microsoft_phi_4_mini_instruct",
        lane_id="phi-4-mini-causal-hf",
        family="Phi-4 mini causal LM",
        model_id="microsoft/Phi-4-mini-instruct",
        preset_relpath="configs/presets/causal_lm/phi4_mini_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="tiiuae_falcon_h1r_7b",
        lane_id="falcon-h1r-7b-causal-hf",
        family="Falcon-H1R 7B causal LM",
        model_id="tiiuae/Falcon-H1R-7B",
        preset_relpath="configs/presets/causal_lm/falcon_h1r_7b_512.yaml",
        adapter="hf_causal",
        verify_profile="dev",
    ),
    EvidenceLane(
        slug="google_flan_t5_base",
        lane_id="flan-t5-base-seq2seq-hf",
        family="FLAN-T5 base seq2seq LM",
        model_id="google/flan-t5-base",
        preset_relpath="configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml",
        adapter="hf_seq2seq",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="qwen_qwen3_30b_a3b_instruct_2507",
        lane_id="qwen3-30b-a3b-moe-causal-hf",
        family="Qwen3 30B-A3B MoE causal LM",
        model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        preset_relpath="configs/presets/causal_lm/qwen3_30b_a3b_instruct_2507_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="mistralai_mixtral_8x7b_v0_1",
        lane_id="mixtral-8x7b-moe-causal-hf",
        family="Mixtral 8x7B MoE causal LM",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        preset_relpath="configs/presets/causal_lm/mixtral_8x7b_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="allenai_olmoe_1b_7b_0924",
        lane_id="olmoe-1b-7b-0924-causal-hf",
        family="OLMoE 1B-active/7B-total causal LM",
        model_id="allenai/OLMoE-1B-7B-0924",
        preset_relpath="configs/presets/causal_lm/olmoe_1b_7b_0924_512.yaml",
        adapter="hf_causal",
        verify_profile="release",
    ),
    EvidenceLane(
        slug="google_gemma_4_26b_a4b_it",
        lane_id="gemma4-26b-a4b-moe-image-text-hf",
        family="Gemma 4 26B-A4B MoE image-text LM",
        model_id="google/gemma-4-26B-A4B-it",
        preset_relpath="configs/presets/multimodal/gemma4_26b_a4b_public_vqav2_256.yaml",
        adapter="hf_multimodal",
        verify_profile="release",
        vision_text_materialization=_public_vqav2_materialization(),
    ),
)


def _load_model_family_catalog(
    path: Path = MODEL_FAMILY_CATALOG_PATH,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model family catalog must be a JSON object")
    return payload


def _catalog_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "_")
    for old, new in ((".", "_"), ("-", "_"), ("+", "_")):
        slug = slug.replace(old, new)
    return slug


def _catalog_lane_defaults(model_id: str) -> tuple[str, str]:
    model_lower = model_id.lower()
    override = CATALOG_PRESET_OVERRIDES.get(model_id)
    if override is not None:
        return override
    if any(
        keyword in model_lower
        for keyword in (
            "bert",
            "roberta",
            "deberta",
            "distilbert",
            "albert",
            "electra",
        )
    ):
        return ("configs/presets/masked_lm/wikitext2_128.yaml", "hf_mlm")
    if any(
        keyword in model_lower
        for keyword in ("t5", "bart", "mbart", "pegasus", "marian", "opus-mt")
    ):
        return ("configs/presets/seq2seq/synth_128.yaml", "hf_seq2seq")
    if model_lower == "google/gemma-4-e4b-it":
        return (
            "configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml",
            "hf_multimodal",
        )
    return ("configs/presets/causal_lm/wikitext2_512.yaml", "auto")


def _build_model_catalog_gpu_lanes(
    payload: dict[str, object] | None = None,
) -> tuple[EvidenceLane, ...]:
    catalog = payload or _load_model_family_catalog()
    lanes: list[EvidenceLane] = []
    seen: set[str] = set()
    for section in MODEL_FAMILY_CATALOG_SECTIONS:
        families = catalog.get(section) or []
        if not isinstance(families, list):
            raise ValueError(f"model_family_catalog.{section} must be a list")
        for family in families:
            if not isinstance(family, dict):
                continue
            display_name = family.get("display_name")
            family_label = display_name if isinstance(display_name, str) else section
            models = family.get("representative_models") or []
            if not isinstance(models, list):
                continue
            for model_id in models:
                if not isinstance(model_id, str) or not model_id or model_id in seen:
                    continue
                preset_relpath, adapter = _catalog_lane_defaults(model_id)
                lanes.append(
                    EvidenceLane(
                        slug=_catalog_slug(model_id),
                        lane_id=f"catalog::{_catalog_slug(model_id)}",
                        family=family_label,
                        model_id=model_id,
                        preset_relpath=preset_relpath,
                        adapter=adapter,
                        verify_profile="dev",
                    )
                )
                seen.add(model_id)
    return tuple(lanes)


MODEL_CATALOG_GPU_LANES = _build_model_catalog_gpu_lanes()


def _build_promotion_gap_gpu_lanes(
    payload: dict[str, object] | None = None,
) -> tuple[EvidenceLane, ...]:
    catalog = payload or _load_model_family_catalog()
    section = catalog.get("promotion_candidates_text_le_14b") or {}
    if not isinstance(section, dict):
        raise ValueError(
            "model_family_catalog.promotion_candidates_text_le_14b must be an object"
        )
    candidates = section.get("candidates") or []
    if not isinstance(candidates, list):
        raise ValueError(
            "model_family_catalog.promotion_candidates_text_le_14b.candidates must be a list"
        )

    lanes: list[EvidenceLane] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("decision") != "blocked_missing_artifacts":
            continue
        if candidate.get("current_catalog_state") != "implemented_coverage":
            continue
        criteria = candidate.get("criteria_status") or {}
        if not isinstance(criteria, dict):
            continue
        if criteria.get("included_preset") != "pass":
            continue
        if criteria.get("included_calibration_config") != "pass":
            continue
        model_id = candidate.get("representative_model")
        if not isinstance(model_id, str) or not model_id:
            continue
        family = candidate.get("display_name")
        family_label = family if isinstance(family, str) and family else model_id
        preset_relpath, adapter = _catalog_lane_defaults(model_id)
        lanes.append(
            EvidenceLane(
                slug=_catalog_slug(model_id),
                lane_id=f"promotion-gap::{_catalog_slug(model_id)}",
                family=family_label,
                model_id=model_id,
                preset_relpath=preset_relpath,
                adapter=adapter,
                verify_profile="dev",
            )
        )
    return tuple(lanes)


PROMOTION_GAP_GPU_LANES = _build_promotion_gap_gpu_lanes()

SUITES: dict[str, tuple[EvidenceLane, ...]] = {
    DEFAULT_SUITE: CURRENT_SUPPORTED_EXPERIMENTAL_LANES,
    REPO_MENTIONED_GPU_SUITE: (
        CURRENT_PUBLISHED_BASIS_LANES
        + DOCUMENTED_SMOKE_CANARY_LANES
        + CURRENT_SUPPORTED_EXPERIMENTAL_LANES
    ),
    MODEL_CATALOG_GPU_SUITE: MODEL_CATALOG_GPU_LANES,
    PROMOTION_GAP_GPU_SUITE: PROMOTION_GAP_GPU_LANES,
    SUPPORT_MATRIX_BACKLOG_GPU_SUITE: SUPPORT_MATRIX_BACKLOG_GPU_LANES,
}


@cache
def _preset_model_config(preset_path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(preset_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    model_cfg = data.get("model")
    return model_cfg if isinstance(model_cfg, dict) else {}


def lane_requires_remote_code(spec: EvidenceLane) -> bool:
    model_cfg = _preset_model_config(str(spec.preset_path))
    return bool(model_cfg.get("trust_remote_code") is True)


def _load_support_matrix(path: Path = SUPPORT_MATRIX_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Support matrix must be a JSON object")
    return payload


def supported_experimental_lane_ids(
    support_matrix: dict[str, object] | None = None,
) -> set[str]:
    payload = support_matrix or _load_support_matrix()
    lanes = payload.get("lanes") or []
    if not isinstance(lanes, list):
        raise ValueError("support_matrix.lanes must be a list")
    lane_ids: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if lane.get("support_tier") != "supported_experimental":
            continue
        lane_id = lane.get("lane_id")
        if isinstance(lane_id, str) and lane_id:
            lane_ids.add(lane_id)
    return lane_ids


def manifest_lane_ids(specs: tuple[EvidenceLane, ...] | list[EvidenceLane]) -> set[str]:
    return {spec.lane_id for spec in specs}


def validate_manifest_coverage(
    specs: tuple[EvidenceLane, ...] | list[EvidenceLane],
    support_matrix: dict[str, object] | None = None,
) -> None:
    expected = supported_experimental_lane_ids(support_matrix)
    actual = manifest_lane_ids(specs)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing lane_ids: " + ", ".join(missing))
        if extra:
            parts.append("unexpected lane_ids: " + ", ".join(extra))
        raise ValueError("Model evidence manifest drift: " + "; ".join(parts))


def select_specs(
    suite: str,
    *,
    slugs: list[str],
    lane_ids: list[str],
    shard_index: int,
    shard_count: int,
) -> list[EvidenceLane]:
    if shard_count < 1:
        raise ValueError("shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must be within [0, shard-count)")

    specs = list(SUITES[suite])
    if slugs:
        slug_set = set(slugs)
        specs = [spec for spec in specs if spec.slug in slug_set]
        missing = sorted(slug_set - {spec.slug for spec in specs})
        if missing:
            raise ValueError("Unknown slugs: " + ", ".join(missing))
    if lane_ids:
        lane_id_set = set(lane_ids)
        specs = [spec for spec in specs if spec.lane_id in lane_id_set]
        missing = sorted(lane_id_set - {spec.lane_id for spec in specs})
        if missing:
            raise ValueError("Unknown lane_ids: " + ", ".join(missing))

    return [spec for idx, spec in enumerate(specs) if idx % shard_count == shard_index]
