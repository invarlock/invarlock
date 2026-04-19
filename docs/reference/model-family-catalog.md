# Model Family Catalog

## Overview

This page is the human-readable rendering of
`contracts/model_family_catalog.json`.

Use it to answer three distinct questions without weakening the public meaning
of the support matrix:

- What is supported as a public lane?
- What families are implemented in code but not publicly supported?
- What families or capabilities should be added next?

## Support Tier vs Coverage State

| Term | Meaning | Source of truth |
| --- | --- | --- |
| `support tier` | Public support/assurance posture for a declared lane. Values stay aligned with `support_matrix.json`. | `contracts/support_matrix.json` |
| `coverage state` | Repo implementation maturity outside the public support matrix, such as `profile_first_class`, `profile_shared_alias`, `auto_or_loader_only`, `loader_only`, or backlog states. | `contracts/model_family_catalog.json` |

The support matrix remains strict. The model family catalog is broader by
design and records code-level visibility, usage-only checkpoints, and
recommended additions. Access-gated vendor checkpoints are intentionally kept
out of declared support lanes and included preset inventory.

## Declared Support

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| GPT-2 causal LM | `published_basis` | `openai-community/gpt2` | Public lane derived from `gpt2-causal-hf`. |
| BERT / RoBERTa MLM | `published_basis` | `bert-base-uncased`, `roberta-base` | Public lane derived from `bert-mlm-hf`. |
| Mistral 7B causal LM | `supported_experimental` | `mistralai/Mistral-7B-v0.1` | Pilot preset and calibration config are included. |
| Ministral 3 causal LM (text-only eval) | `supported_experimental` | `mistralai/Ministral-3-8B-Instruct-2512-BF16`, `mistralai/Ministral-3-14B-Instruct-2512-BF16` | Text-only pilot presets and calibration configs are included for both 8B and 14B checkpoints. |
| Qwen2 7B causal LM | `supported_experimental` | `Qwen/Qwen2-7B` | Pilot preset and calibration config are included. |
| Qwen2.5 14B causal LM | `supported_experimental` | `Qwen/Qwen2.5-14B` | Pilot preset and calibration config are included. |
| Qwen3 causal LM | `supported_experimental` | `Qwen/Qwen3-8B` | Pilot preset and calibration config are included. |
| DeepSeek-R1-Distill-Qwen causal LM | `supported_experimental` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Pilot preset and calibration config are included. |
| Phi-4 causal LM (text-only eval) | `supported_experimental` | `microsoft/Phi-4-reasoning-plus` | Text-only pilot preset and calibration config are included. Current HF runtime validation closes cleanly when the lane opts into `trust_remote_code`. |
| Gemma 4 E2B causal LM (text-only eval) | `supported_experimental` | `google/gemma-4-E2B-it` | Text-only pilot preset and calibration config are included. Image-text evaluation uses the explicit `hf_multimodal` + `vision_text` path. |
| TinyLlama 1.1B causal LM | `supported_experimental` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Ungated Llama-family pilot lane with included preset and calibration config. |
| OLMo 2 causal LM | `supported_experimental` | `allenai/OLMo-2-1124-7B`, `allenai/OLMo-2-1124-13B-Instruct` | Pilot presets and calibration configs are included for both 7B and 13B scale points. |
| Qwen3.5 causal LM | `supported_experimental` | `Qwen/Qwen3.5-9B` | Pilot preset and calibration config are included. |
| Seq2Seq / local pairs | `community_experimental` | `t5-small`, `facebook/bart-base` | Generic seq2seq lane without a published-basis claim. |

## Implemented Coverage

| Family | Coverage state | Representative models | Notes |
| --- | --- | --- | --- |
| Mixtral | `profile_first_class` | `mistralai/Mixtral-8x7B-v0.1` | Profile and loader code recognize the family directly. |
| Llama | `profile_first_class` | `openlm-research/open_llama_7b`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Generic Llama-family profile handling is first-class. TinyLlama provides the ungated declared support lane, while access-gated vendor checkpoints remain omitted. |
| Qwen family aliases (Qwen1.5/Qwen2.5/Qwen3 naming) | `profile_first_class` | `Qwen/Qwen2.5-14B`, `Qwen/Qwen3.5-9B` | Shared qwen-family heuristics cover aliases beyond the declared Qwen2, Qwen2.5 14B, Qwen3, and Qwen3.5 lanes, including usage-only Qwen2.5 checkpoints. |
| Yi | `profile_first_class` | `01-ai/Yi-34B` | Treated as a RoPE decoder family in profile logic. |
| Phi family | `profile_first_class` | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-4-reasoning-plus` | Dedicated phi-family selectors exist. Phi-4 has a declared text-only lane, while multimodal Phi-4 remains backlog-only. |
| Gemma family | `profile_first_class` | `google/gemma-3-4b-it`, `google/gemma-4-E2B-it` | Gemma 3/4 selectors and loaders are first-class. Gemma 4 E2B has a declared text-only lane, image-text evaluation uses `hf_multimodal` + `vision_text`, and audio remains deferred. |
| OPT / GPT-NeoX / GPT-J | `profile_shared_alias` | `facebook/opt-1.3b`, `EleutherAI/gpt-neox-20b` | Available through shared GPT-style paths. |
| GPT-OSS | `profile_first_class` | `openai/gpt-oss-20b` | Dedicated profile selectors and HF causal decoder spec now cover the open-weight checkpoint directly. |
| Falcon | `auto_or_loader_only` | `tiiuae/falcon-7b` | Visible through adapter-auto heuristics only. |
| GLM | `auto_or_loader_only` | `THUDM/glm-4-9b-chat` | Visible through adapter-auto heuristics only. |
| DeepSeek | `profile_first_class` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek distill checkpoints share the qwen-family route. Oversized FP8 checkpoint-specific repo hooks and included configs are omitted because they do not fit the supported hardware/runtime path. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `auto_or_loader_only` | `distilbert-base-uncased`, `microsoft/deberta-v3-base` | Loader/auto support exceeds the public BERT / RoBERTa lane. |
| Broader seq2seq families (mBART/PEGASUS/Marian) | `auto_or_loader_only` | `facebook/mbart-large-50`, `Helsinki-NLP/opus-mt-en-de` | Loader support is broader than the generic seq2seq public lane. |

## Usage Only

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| Qwen2.5 7B | `usage_only` | `Qwen/Qwen2.5-7B` | Shipped preset/config and smoke scaffolding exist, but the `<=14B` promotion wave did not close approved calibration/evaluation evidence for a declared public lane. |
| Qwen2.5 32B | `usage_only` | `Qwen/Qwen2.5-32B` | Used in evidence-pack suites and validation defaults outside the declared Qwen2.5 14B support lane. |
| Qwen1.5 72B | `usage_only` | `Qwen/Qwen1.5-72B` | Used concretely in evidence-pack suites. |
| Yi 34B | `usage_only` | `01-ai/Yi-34B` | Used in workshop and full evidence-pack suites. |
| Mixtral 8x7B | `usage_only` | `mistralai/Mixtral-8x7B-v0.1` | Used in evidence-pack flows without a public support lane. |

## <=14B Text Promotion Wave

This wave audits ungated text-only Hugging Face families at `<=14B` that are
visible in repo coverage but still sit outside declared support. Promotion
still requires the full six-part bar: explicit recognition, a shipped preset,
a shipped calibration config, targeted tests, CLI smoke evidence, and approved
calibration/evaluation evidence.

No new family cleared the full promotion bar in this wave. Qwen2.5 7B closed
the shipped artifact and smoke pieces, but the empirical evidence gate still
failed, so it remains usage-only for now. The remaining `<=14B` text families
are either missing shipped lane artifacts or intentionally kept out of scope
for this pass.

| Family | Representative model | Decision | Current state | Notes |
| --- | --- | --- | --- | --- |
| Qwen2.5 7B causal LM | `Qwen/Qwen2.5-7B` | `blocked_missing_artifacts` | `usage_only` | Shipped preset/config/tests/smoke now exist, but the reduced promotion matrix finished `FAIL`: three clean controls drifted, and the SVD clean scenario id did not align with the shared manifest. |
| OpenLLaMA 7B causal LM | `openlm-research/open_llama_7b` | `blocked_missing_artifacts` | `implemented_coverage` | Recognition and targeted tests exist, but the shipped lane artifacts and evidence are missing. |
| Phi-3 Mini 4K Instruct causal LM | `microsoft/Phi-3-mini-4k-instruct` | `explicitly_out_of_scope` | `implemented_coverage` | This wave keeps the public Phi surface at the shipped Phi-4 text-only lane. |
| Gemma 3 4B IT | `google/gemma-3-4b-it` | `explicitly_out_of_scope` | `implemented_coverage` | The broader Gemma family remains multimodal/audio-capable, so it stays out of this text-only wave. |
| OPT 1.3B causal LM | `facebook/opt-1.3b` | `blocked_missing_artifacts` | `implemented_coverage` | Shared GPT-style recognition exists, but the public lane artifacts and evidence are missing. |
| Falcon 7B causal LM | `tiiuae/falcon-7b` | `blocked_missing_artifacts` | `implemented_coverage` | Recognition and targeted variant-path tests exist, but the public lane artifacts are absent. |
| GLM 4 9B Chat | `THUDM/glm-4-9b-chat` | `blocked_missing_artifacts` | `implemented_coverage` | Recognition and targeted variant-path tests exist, but the public lane artifacts are absent. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `distilbert-base-uncased` | `blocked_missing_artifacts` | `implemented_coverage` | Loader and adapter tests exist for DistilBERT and DeBERTa, but the shipped family lane artifacts are absent. |
| mBART large 50 seq2seq | `facebook/mbart-large-50` | `explicitly_out_of_scope` | `implemented_coverage` | Generic seq2seq/community lanes are excluded from this text-only wave. |

The machine-readable criterion-by-criterion ledger for this wave lives under
`promotion_candidates_text_le_14b` in `contracts/model_family_catalog.json`.

## Recommended Additions

| Priority | Family | Planned support mode | Representative models | Notes |
| --- | --- | --- | --- | --- |
| `P2` | Audio-text evaluation pipeline | `phase2_audio_eval` | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` | Image-text evaluation is included. Audio-capable evaluation for the smaller Gemma 4 checkpoints remains deferred. |

## Promotion Criteria

A family only moves into `support_matrix.json` after all of the following are
present:

1. explicit adapter/profile recognition
2. an included preset
3. an included calibration config
4. targeted tests
5. CLI smoke evidence
6. approved calibration/evaluation evidence

## Related Documentation

- [Public Contracts](contracts.md)
- [Model Adapters](model-adapters.md)
- [Tier Policy Tuning CLI (Calibration)](calibration.md)
