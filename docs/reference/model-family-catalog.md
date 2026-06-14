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
| `lifecycle classification` | Whether a lane, catalog family, or candidate is published, backlog, blocked, smoke-only, usage-only, or out of scope. | `contracts/model_classification.json` |

The support matrix remains strict. The model family catalog is broader by
design and records code-level visibility, usage-only checkpoints, and
recommended additions. The model classification contract records the promotion
decision and blocker state for those entries, including the blocked named
checkpoint list used by repo checks. The support matrix is ordered by evidence readiness:
baseline fixtures, published modern decoder evidence, repo-maintained
experimental lanes, community candidate backlog, and concrete task-family
bases. Within those sections, related families and scale points stay adjacent
without adding another visible column. Access-gated vendor checkpoints are
intentionally kept out of included preset inventory.

## Declared Support

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| GPT-2 causal LM | `published_basis` | `openai-community/gpt2` | Public lane derived from `gpt2-causal-hf`. |
| BERT / RoBERTa MLM | `published_basis` | `bert-base-uncased`, `roberta-base` | Public lane derived from `bert-mlm-hf`. |
| Mistral 7B causal LM | `published_basis` | `mistralai/Mistral-7B-v0.1` | Public container-backed evidence fixture is included, plus a real guard-value scenario package with PM-pass, baseline-relative spectral/RMT/variance evidence. |
| Ministral 3 8B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | Public container-backed evidence fixture is included. |
| Ministral 3 14B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | Public container-backed evidence fixture is included. |
| TinyLlama 1.1B causal LM | `published_basis` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Public container-backed evidence fixture is included. |
| OLMo 2 7B causal LM | `published_basis` | `allenai/OLMo-2-1124-7B` | Public container-backed evidence fixture is included. |
| OLMo 2 13B causal LM | `published_basis` | `allenai/OLMo-2-1124-13B-Instruct` | Public container-backed evidence fixture is included. |
| OLMoE 1B-active/7B-total causal LM | `published_basis` | `allenai/OLMoE-1B-7B-0924` | Public release-profile container-backed evidence fixture is included from the smaller MoE H100 validation run. |
| OpenLLaMA 7B causal LM | `published_basis` | `openlm-research/open_llama_7b` | Public release-profile container-backed evidence fixture is included. |
| Falcon 7B causal LM | `published_basis` | `tiiuae/falcon-7b` | Public release-profile container-backed evidence fixture is included. |
| Qwen2 7B causal LM | `published_basis` | `Qwen/Qwen2-7B` | Public container-backed evidence fixture is included. |
| Qwen2.5 7B causal LM | `published_basis` | `Qwen/Qwen2.5-7B` | Public container-backed evidence fixture is included. |
| Qwen2.5 14B causal LM | `published_basis` | `Qwen/Qwen2.5-14B` | Public container-backed evidence fixture is included. |
| Qwen3 causal LM | `published_basis` | `Qwen/Qwen3-8B` | Public container-backed evidence fixture is included. |
| DeepSeek-R1-Distill-Qwen causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Public container-backed evidence fixture is included. |
| DeepSeek-R1-0528-Qwen3 8B causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Container-backed public report, runtime manifest, and signed evidence pack are included. |
| DeepSeek-R1-Distill-Qwen 14B causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Public release-profile container-backed evidence fixture is included. |
| Phi-4 causal LM (text-only eval) | `published_basis` | `microsoft/Phi-4-reasoning-plus` | Public text-only container-backed evidence fixture is included; guard-overhead measurement is skipped by preset policy. |
| Gemma 4 E2B causal LM (text-only eval) | `published_basis` | `google/gemma-4-E2B-it` | Public release-profile container-backed evidence fixture is included. Image-text evaluation uses the explicit `hf_multimodal` + `vision_text` path. |
| Qwen3.5 causal LM | `published_basis` | `Qwen/Qwen3.5-9B` | Public container-backed evidence fixture is included. |
| Ministral 3 3B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | Public release-profile container-backed evidence fixture is included. |
| Granite 4.1 3B causal LM | `published_basis` | `ibm-granite/granite-4.1-3b` | Public release-profile container-backed evidence fixture is included. |
| Granite 4.1 8B causal LM | `published_basis` | `ibm-granite/granite-4.1-8b` | Public release-profile container-backed evidence fixture is included. |
| Gemma 4 12B any-to-any LM | `published_basis` | `google/gemma-4-12B-it` | Public release-profile H100 image-text evidence fixture is included on pinned public VQAv2 materialization. The no-op report passes strict policy with 0.565 final accuracy over 400 examples and no guard warnings; audio and broader any-to-any behavior remain out of scope. |
| Qwen3.5 4B image-text LM | `published_basis` | `Qwen/Qwen3.5-4B` | Public release-profile H100 image-text evidence fixture is included on pinned public VQAv2 materialization after the structured JSON-answer prompt fix. The no-op report passes strict policy with 0.855 final accuracy over 400 examples and no guard warnings. |
| Qwen3.5 2B image-text LM | `published_basis` | `Qwen/Qwen3.5-2B` | Public release-profile container-backed image-text evidence fixture is included on pinned public VQAv2 materialization; it is no-op preservation/null-behavior evidence, not guard-value proof. |
| SmolLM3 3B causal LM | `published_basis` | `HuggingFaceTB/SmolLM3-3B` | Public release-profile container-backed evidence fixture is included; guard-overhead measurement is skipped by preset policy. |
| Phi-4 mini causal LM | `published_basis` | `microsoft/Phi-4-mini-instruct` | Public release-profile container-backed evidence fixture is included after the FFN projection family-classification fix. |
| FLAN-T5 base seq2seq LM | `published_basis` | `google/flan-t5-base` | Public release-profile container-backed evidence fixture is included on pinned CNN/DailyMail validation data through `hf_seq2seq`. |

## Implemented Coverage

| Family | Coverage state | Representative models | Notes |
| --- | --- | --- | --- |
| Qwen3 30B-A3B MoE causal LM | `published_basis` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Public release-profile H100 evidence fixture is included on public WikiText-103 with all-8 H100 sharding and scoped attention/router/shared-expert guard scans; it is no-op preservation evidence, not benchmark-quality, exhaustive expert-bank, or MoE routing-quality assurance. |
| Gemma 4 26B-A4B MoE image-text LM | `published_basis` | `google/gemma-4-26B-A4B-it` | Public release-profile H100 image-text evidence fixture is included on pinned public VQAv2 materialization. The no-op report passes strict policy with 0.555 final accuracy over 400 examples and no guard warnings; it is not audio, exhaustive expert-bank, or MoE routing-quality evidence. |
| Mixtral 8x7B MoE causal LM | `published_basis` | `mistralai/Mixtral-8x7B-v0.1` | Public release-profile H100 evidence fixture is included as a no-op preservation basis with full guard scans; it is not a benchmark-quality or MoE routing-quality claim, and guard-overhead measurement is skipped by preset policy. |
| Llama | `profile_first_class` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Generic Llama-family profile handling is first-class. OpenLLaMA and TinyLlama provide ungated declared support lanes, while access-gated vendor checkpoints remain omitted. |
| Qwen family aliases (Qwen1.5/Qwen2.5/Qwen3 naming) | `profile_first_class` | `Qwen/Qwen2.5-14B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B` | Shared qwen-family heuristics cover aliases beyond the declared text-only Qwen2, Qwen2.5 14B, Qwen3, and Qwen3.5 9B public lanes. Qwen3.5 2B and Qwen3.5 4B have published image-text evidence through `hf_multimodal` and pinned public VQAv2 materialization. |
| Yi | `profile_first_class` | `01-ai/Yi-34B` | Treated as a RoPE decoder family in profile logic. |
| Phi family | `profile_first_class` | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-4-reasoning-plus` | Dedicated phi-family selectors exist. Phi-4 has a declared published text-only lane, while multimodal Phi-4 remains backlog-only. |
| Gemma family | `profile_first_class` | `google/gemma-4-E2B-it`, `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it` | Gemma-family selectors and loaders remain first-class for compatible local or user-supplied checkpoints. Repo-declared published Gemma support includes Gemma 4 E2B text-only plus Gemma 4 12B and Gemma 4 26B-A4B image-text evidence. |
| OPT / GPT-NeoX / GPT-J | `profile_shared_alias` | `EleutherAI/gpt-neox-20b` | Available through shared GPT-style paths. The common OPT-1.3B hosted checkpoint is intentionally not named in repo support inventory because its license is not Apache-2.0 or MIT. |
| GPT-OSS | `profile_first_class` | `openai/gpt-oss-20b` | Dedicated profile selectors and HF causal decoder spec now cover the open-weight checkpoint directly. |
| Falcon | `profile_shared_alias` | `tiiuae/falcon-7b` | Falcon 7B now has a declared support lane; remaining Falcon-family coverage is available through adapter-auto heuristics and variant-path tests. |
| GLM | `auto_or_loader_only` | `local-glm-compatible-checkpoint` | Visible through adapter-auto heuristics only. The public catalog intentionally avoids naming hosted GLM chat checkpoints that fall outside the repo's Apache-2.0/MIT named-checkpoint policy. |
| DeepSeek | `profile_first_class` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek distill checkpoints share the qwen-family route. DeepSeek-R1-Distill-Qwen 7B has a declared published lane; oversized FP8 checkpoint-specific repo hooks and included configs are omitted because they do not fit the supported hardware/runtime path. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `auto_or_loader_only` | `distilbert-base-uncased`, `microsoft/deberta-v3-base` | Loader/auto support exceeds the public BERT / RoBERTa lane. |
| Broader seq2seq families (mBART/PEGASUS/Marian) | `auto_or_loader_only` | `facebook/mbart-large-50` | Loader support is broader than the FLAN-T5 public seq2seq basis. CC-BY-4.0-only hosted checkpoints are intentionally not named because they are outside the repo's strict Apache-2.0/MIT named-checkpoint policy. |

## Usage Only

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| Qwen2.5 32B | `usage_only` | `Qwen/Qwen2.5-32B` | Used in evidence-pack suites and validation defaults outside the declared Qwen2.5 14B support lane. |
| Yi 34B | `usage_only` | `01-ai/Yi-34B` | Used in workshop and full evidence-pack suites. |

## <=14B Text Candidate Inventory

This section summarizes the contract-tracked `<=14B` text and MLM candidates
that sit outside, adjacent to, or have recently graduated into declared
support.

It is a catalog view, not a run ledger. Exact criterion-by-criterion status and
decision codes live under `promotion_candidates_text_le_14b` in
`contracts/model_family_catalog.json`.

| Family | Representative model | Promotion status | Catalog location | Notes |
| --- | --- | --- | --- | --- |
| Qwen2.5 7B causal LM | `Qwen/Qwen2.5-7B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed public report, runtime manifest, and signed evidence pack. |
| Qwen2.5 14B causal LM | `Qwen/Qwen2.5-14B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed public report, runtime manifest, and signed evidence pack. |
| Qwen3 8B causal LM | `Qwen/Qwen3-8B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed public report, runtime manifest, and signed evidence pack. |
| DeepSeek-R1-Distill-Qwen causal LM | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed public report, runtime manifest, and signed evidence pack. |
| Phi-4 reasoning-plus causal LM | `microsoft/Phi-4-reasoning-plus` | `promoted_published_basis` | `published_basis` | Promoted with container-backed public report, runtime manifest, and signed evidence pack; this fixture is text-only and skips guard-overhead measurement by preset policy. |
| OpenLLaMA 7B causal LM | `openlm-research/open_llama_7b` | `promoted_published_basis` | `published_basis` | Promoted with release-profile container-backed public report, runtime manifest, and signed evidence pack. |
| Phi-3 Mini 4K Instruct causal LM | `microsoft/Phi-3-mini-4k-instruct` | `explicitly_out_of_scope` | `implemented_coverage` | The current declared Phi support surface remains the shipped Phi-4 text-only lane. |
| Falcon 7B causal LM | `tiiuae/falcon-7b` | `promoted_published_basis` | `published_basis` | Promoted with release-profile container-backed public report, runtime manifest, and signed evidence pack. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `distilbert-base-uncased` | `blocked_missing_artifacts` | `implemented_coverage` | Loader and adapter tests exist for DistilBERT and DeBERTa, and the repo ships a lane preset plus calibration config with dry-run sweep coverage, but approved calibration/evaluation evidence is still missing. |
| mBART large 50 seq2seq | `facebook/mbart-large-50` | `explicitly_out_of_scope` | `implemented_coverage` | FLAN-T5 base now supplies the concrete public seq2seq basis; mBART still needs its own evidence. |

The machine-readable criterion-by-criterion ledger for this candidate set lives
under `promotion_candidates_text_le_14b` in
`contracts/model_family_catalog.json`.

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
