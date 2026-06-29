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
| `coverage state` | Repo implementation maturity outside declared public support lanes, such as `profile_first_class`, `profile_shared_alias`, `auto_or_loader_only`, `loader_only`, or backlog states. | `contracts/model_family_catalog.json` |
| `lifecycle classification` | Whether a lane, catalog family, or candidate is published, backlog, blocked, smoke-only, usage-only, or out of scope. | `contracts/model_classification.json` |

The support matrix remains strict. In this catalog, `declared_support` is the
complete public `published_basis` support set and follows the support-matrix
lane order. `implemented_coverage` is reserved for code-level visibility that
does not itself create a public support claim. The model classification
contract records promotion decisions and checkpoint status used by repo checks.
Access-gated vendor checkpoints are not included as repo-shipped presets.

Catalog notes use a compact format: evidence surface first, then scope caveats
only when the public evidence lane needs them. Exact metrics, hardware details,
and rebuild instructions live in the public evidence artifacts and linked
contracts.

## Declared Support

These rows are the complete public `published_basis` support set from
`contracts/support_matrix.json`, rendered in support-matrix order. A row here
means the repo includes a public evidence fixture for that lane; the row notes
state the evidence scope and non-goals.

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| GPT-2 causal LM | `published_basis` | `openai-community/gpt2` | Derived from support_matrix lane `gpt2-causal-hf`. |
| BERT / RoBERTa MLM | `published_basis` | `bert-base-uncased`, `roberta-base` | Derived from support_matrix lane `bert-mlm-hf`. |
| Mistral 7B causal LM | `published_basis` | `mistralai/Mistral-7B-v0.1` | Promoted from support_matrix lane `mistral-7b-causal-hf` with container-backed public evidence. |
| Ministral 3 8B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | Promoted from support_matrix lane `ministral-3-8b-text-causal-hf` with container-backed public evidence. The separate 14B Ministral lane has its own published evidence fixture. |
| Ministral 3 14B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | Promoted from support_matrix lane `ministral-3-14b-text-causal-hf` with container-backed public evidence. |
| Qwen2 7B causal LM | `published_basis` | `Qwen/Qwen2-7B` | Promoted from support_matrix lane `qwen2-7b-causal-hf` with container-backed public evidence. |
| Qwen2.5 7B causal LM | `published_basis` | `Qwen/Qwen2.5-7B` | Promoted from support_matrix lane `qwen2-5-7b-causal-hf` with container-backed public evidence. |
| Qwen2.5 14B causal LM | `published_basis` | `Qwen/Qwen2.5-14B` | Promoted from support_matrix lane `qwen2-5-14b-causal-hf` with container-backed public evidence. |
| Qwen3 causal LM | `published_basis` | `Qwen/Qwen3-8B` | Promoted from support_matrix lane `qwen3-causal-hf` with container-backed public evidence. |
| DeepSeek-R1-Distill-Qwen causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Promoted from support_matrix lane `deepseek-r1-distill-qwen-causal-hf` with container-backed public evidence. |
| Phi-4 causal LM (text-only eval) | `published_basis` | `microsoft/Phi-4-reasoning-plus` | Promoted from support_matrix lane `phi-4-text-causal-hf` with container-backed public evidence. The public fixture is text-only and skips guard-overhead measurement by preset policy. |
| Gemma 4 E2B causal LM (text-only eval) | `published_basis` | `google/gemma-4-E2B-it` | Promoted from support_matrix lane `gemma4-e2b-text-causal-hf` with release-profile container-backed public evidence. Public support is text-only; image-text evaluation remains on the explicit hf_multimodal + vision_text path. |
| TinyLlama 1.1B causal LM | `published_basis` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Promoted from support_matrix lane `tinyllama-1-1b-causal-hf` with container-backed public evidence. |
| OLMo 2 7B causal LM | `published_basis` | `allenai/OLMo-2-1124-7B` | Promoted from support_matrix lane `olmo-2-7b-causal-hf` with container-backed public evidence. |
| OLMo 2 13B causal LM | `published_basis` | `allenai/OLMo-2-1124-13B-Instruct` | Promoted from support_matrix lane `olmo-2-13b-causal-hf` with container-backed public evidence. |
| OpenLLaMA 7B causal LM | `published_basis` | `openlm-research/open_llama_7b` | Promoted from support_matrix lane `open-llama-7b-causal-hf` with release-profile container-backed public evidence. |
| Falcon 7B causal LM | `published_basis` | `tiiuae/falcon-7b` | Promoted from support_matrix lane `falcon-7b-causal-hf` with release-profile container-backed public evidence. |
| Qwen3.5 causal LM | `published_basis` | `Qwen/Qwen3.5-9B` | Promoted from support_matrix lane `qwen3-5-causal-hf` with container-backed public evidence. |
| Granite 4.1 3B causal LM | `published_basis` | `ibm-granite/granite-4.1-3b` | Promoted from support_matrix lane `granite-4-1-3b-causal-hf` with container-backed release-profile public evidence. |
| FLAN-T5 base seq2seq LM | `published_basis` | `google/flan-t5-base` | Promoted with release-profile container-backed public report, runtime manifest, signed evidence pack, and pinned CNN/DailyMail validation data through hf_seq2seq. |
| Gemma 4 12B any-to-any LM | `published_basis` | `google/gemma-4-12B-it` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.565 final accuracy over 400 examples, ratio_vs_baseline 1.009, and no guard warnings. Gemma 4 12B uses the explicit hf_multimodal path and requires the multimodal runtime stack (transformers>=5.12.0 and torchvision>=0.26.0). This is no-op preservation/null-behavior evidence; audio and broader any-to-any behavior remain out of scope, and it is not guard-value proof. |
| Gemma 4 E4B image-text LM | `published_basis` | `google/gemma-4-E4B-it` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.500 final accuracy over 400 examples, ratio_vs_baseline 1.010, and no guard warnings. This is no-op preservation/null-behavior evidence; audio and broader any-to-any behavior remain out of scope, and it is not guard-value proof. |
| Gemma 4 E2B image-text LM | `published_basis` | `google/gemma-4-E2B-it` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.388 final accuracy over 400 examples, ratio_vs_baseline 1.013, and no guard warnings. This is no-op preservation/null-behavior evidence; audio and broader multimodal behavior remain out of scope, and it is not guard-value proof. |
| Qwen3.5 4B image-text LM | `published_basis` | `Qwen/Qwen3.5-4B` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.855 final accuracy over 400 examples, ratio_vs_baseline 1.036, and no guard warnings. This is no-op preservation/null-behavior evidence; it is not guard-value proof. |
| Qwen3.5 2B image-text LM | `published_basis` | `Qwen/Qwen3.5-2B` | Promoted from support_matrix lane `qwen3-5-2b-image-text-hf` with release-profile container-backed public VQAv2 image-text evidence, runtime manifest, and signed evidence pack. This is no-op preservation/null-behavior evidence, not guard-value proof. |
| Qwen3.5 27B image-text LM (scoped) | `published_basis` | `Qwen/Qwen3.5-27B` | Promoted with release-profile container-backed public VQAv2 image-text evidence using scoped self-attention and MLP guard scans. The no-op report passes strict policy with 0.8975 final accuracy over 400 examples and no guard warnings. Linear-attention module coverage remains a separate strict spectral-cap finding in the larger-model follow-on addendum; this is no-op preservation/null-behavior evidence and not guard-value proof. |
| Qwen3.6 27B image-text LM (scoped) | `published_basis` | `Qwen/Qwen3.6-27B` | Promoted with release-profile container-backed public VQAv2 image-text evidence using scoped self-attention and MLP guard scans. The no-op report passes strict policy with 0.8825 final accuracy over 400 examples and no guard warnings. Linear-attention module coverage remains a separate strict spectral-cap finding in the larger-model follow-on addendum; this is no-op preservation/null-behavior evidence and not guard-value proof. |
| Gemma 4 26B-A4B MoE image-text LM | `published_basis` | `google/gemma-4-26B-A4B-it` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.555 final accuracy over 400 examples, ratio_vs_baseline 1.009, and no guard warnings. This is no-op preservation/null-behavior evidence; audio, exhaustive expert-bank behavior, and MoE routing quality remain out of scope, and it is not guard-value proof. |
| Gemma 4 31B image-text LM | `published_basis` | `google/gemma-4-31B-it` | Promoted with release-profile container-backed public VQAv2 image-text evidence. The no-op report passes strict policy with 0.610 final accuracy over 400 examples and no guard warnings. This is no-op preservation/null-behavior evidence; audio and broader any-to-any behavior remain out of scope, and it is not guard-value proof. |
| Mixtral 8x7B MoE causal LM | `published_basis` | `mistralai/Mixtral-8x7B-v0.1` | Promoted as a Mixtral MoE causal LM published basis with explicit hf_causal preset, release-profile public report, runtime manifest, and signed evidence pack. The preset keeps the full guard scan surface while using BF16, automatic device placement, low-CPU-memory loading, disabled optional HF loading-info collection, and a skipped optional guard-overhead check for large-model resource discipline. |
| Qwen3 30B-A3B MoE causal LM | `published_basis` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Promoted as a recent Qwen3 MoE causal LM published basis with explicit hf_causal preset, public WikiText-103 release-profile report, runtime manifest, and signed evidence pack. The fixture uses BF16/low-CPU-memory loading, disabled optional Hugging Face loading-info collection, automatic device placement, and scoped attention/router/shared-expert guard scans. This is no-op preservation evidence; it is not a benchmark-quality, fine-tuning, deployment, exhaustive expert-bank, or MoE routing-quality claim. |
| GPT-OSS 20B causal LM | `published_basis` | `openai/gpt-oss-20b` | Promoted with release-profile container-backed public WikiText-2 validation evidence for the pinned fixture. The no-op report passes strict policy with final perplexity 86.273 over 400 paired windows and no guard warnings. This is no-op preservation evidence; it is not a fine-tuning, benchmark-quality, deployment, safety, or alternate-seed/window robustness claim. |
| OLMoE 1B-active/7B-total causal LM | `published_basis` | `allenai/OLMoE-1B-7B-0924` | Promoted with release-profile container-backed public report, runtime manifest, and signed evidence pack. This fixture is the smaller MoE validation basis; larger MoE assurance is covered by Mixtral, Qwen3 30B-A3B, and Gemma 4 26B-A4B published bases. Gemma 4 26B-A4B is image-text preservation/null-behavior evidence, not audio, exhaustive expert-bank, or MoE routing-quality evidence. |
| Ministral 3 3B causal LM (text-only eval) | `published_basis` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | Promoted from support_matrix lane `ministral-3-3b-text-causal-hf` with container-backed public evidence. |
| Granite 4.1 8B causal LM | `published_basis` | `ibm-granite/granite-4.1-8b` | Public release-profile container-backed evidence fixture is included. |
| SmolLM3 3B causal LM | `published_basis` | `HuggingFaceTB/SmolLM3-3B` | Promoted with release-profile container-backed public report, runtime manifest, signed evidence pack, and recorded 400-window preservation evidence. The preset explicitly skips guard-overhead measurement, so this is preservation evidence and not guard-overhead evidence. |
| Phi-4 mini causal LM | `published_basis` | `microsoft/Phi-4-mini-instruct` | Promoted with release-profile container-backed public report, runtime manifest, signed evidence pack, and recorded 400-window preservation evidence. |
| DeepSeek-R1-Distill-Qwen 14B causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Promoted from support_matrix lane `deepseek-r1-distill-qwen-14b-causal-hf` with container-backed public evidence using baseline-report reuse recovery. |
| DeepSeek-R1-0528-Qwen3 8B causal LM | `published_basis` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Promoted from support_matrix lane `deepseek-r1-0528-qwen3-8b-causal-hf` with container-backed public evidence. |

## Implemented Coverage

These rows are implementation, adapter, profile, or loader coverage that exists
outside declared public support lanes. They are not public support claims unless
they later move into `declared_support` and `support_matrix.json`.

| Family | Coverage state | Representative models | Notes |
| --- | --- | --- | --- |
| Llama | `profile_first_class` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Generic Llama-family profile handling is first-class. OpenLLaMA and TinyLlama now provide ungated declared support lanes, while access-gated vendor checkpoints remain omitted. |
| Qwen family aliases (Qwen1.5/Qwen2.5/Qwen3 naming) | `profile_first_class` | `Qwen/Qwen2.5-14B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.6-27B` | Shared qwen-family heuristics cover aliases beyond the declared text-only Qwen2, Qwen2.5 14B, Qwen3, and Qwen3.5 9B public lanes. Qwen3.5 2B, Qwen3.5 4B, and scoped Qwen3.5/Qwen3.6 27B have published image-text evidence through hf_multimodal and pinned public VQAv2 materialization. |
| Yi | `profile_first_class` | `01-ai/Yi-34B` | Yi is treated as a RoPE decoder family in profile and adapter-auto logic. |
| Phi family | `profile_first_class` | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-4-reasoning-plus` | Phi-family selectors are first-class. Phi-4 now has a declared text-only lane via the reasoning-plus pilot, while multimodal Phi-4 remains backlog-only. |
| Gemma family | `profile_first_class` | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` | Gemma-family selectors and loaders are first-class for compatible local or user-supplied checkpoints. Repo-declared published Gemma support includes Gemma 4 E2B text-only plus Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B image-text published-basis evidence. |
| OPT / GPT-NeoX / GPT-J | `profile_shared_alias` | `EleutherAI/gpt-neox-20b` | These families are available through shared GPT-style loader and profile paths rather than dedicated public support lanes. |
| Falcon | `profile_shared_alias` | `tiiuae/falcon-7b` | Falcon 7B now has a declared support lane; remaining Falcon-family coverage is available through adapter-auto heuristics and variant-path tests. |
| GLM | `auto_or_loader_only` | `local-glm-compatible-checkpoint` | Family visibility currently comes from adapter-auto heuristics only. |
| DeepSeek | `profile_first_class` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek distill checkpoints continue to share the qwen-family route. Oversized FP8 checkpoint-specific repo hooks and shipped configs were removed after bring-up showed that they do not fit the supported hardware/runtime path. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `auto_or_loader_only` | `distilbert-base-uncased`, `microsoft/deberta-v3-base` | Loader and adapter-auto support exceeds the public BERT / RoBERTa support lane. |
| Broader seq2seq families (mBART/PEGASUS/Marian) | `auto_or_loader_only` | `facebook/mbart-large-50` | Concrete loader families exist beyond the public FLAN-T5 base seq2seq basis; mBART/PEGASUS/Marian still need dedicated presets and public evidence before they become declared support lanes. |

## Usage Only

These checkpoints appear in repo workflows or historical validation inputs, but
they are not public support lanes.

| Family | State | Representative models | Notes |
| --- | --- | --- | --- |
| Qwen2.5 32B | `usage_only` | `Qwen/Qwen2.5-32B` | Used in evidence-pack suites and validation defaults outside the declared Qwen2.5 14B support lane. |
| Yi 34B | `usage_only` | `01-ai/Yi-34B` | Used in workshop and full evidence-pack suites without a public support lane. |

## <=14B Text Candidate Inventory

This section summarizes the contract-tracked `<=14B` text and MLM candidates
that sit outside, adjacent to, or have recently graduated into declared
support.

It is a catalog view, not a run ledger. Exact criterion-by-criterion status and
decision codes live under `promotion_candidates_text_le_14b` in
`contracts/model_family_catalog.json`.

| Family | Representative model | Promotion status | Catalog location | Notes |
| --- | --- | --- | --- | --- |
| Qwen2.5 7B causal LM | `Qwen/Qwen2.5-7B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed Qwen2.5 7B public report, runtime manifest, and signed evidence pack. |
| Qwen2.5 14B causal LM | `Qwen/Qwen2.5-14B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed Qwen2.5 14B public report, runtime manifest, and signed evidence pack. |
| Qwen3 8B causal LM | `Qwen/Qwen3-8B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed Qwen3 8B public report, runtime manifest, and signed evidence pack. |
| DeepSeek-R1-Distill-Qwen causal LM | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `promoted_published_basis` | `published_basis` | Promoted with container-backed DeepSeek-R1-Distill-Qwen 7B public report, runtime manifest, and signed evidence pack. |
| Phi-4 reasoning-plus causal LM | `microsoft/Phi-4-reasoning-plus` | `promoted_published_basis` | `published_basis` | Promoted with container-backed Phi-4 reasoning-plus public report, runtime manifest, and signed evidence pack. This fixture is text-only and skips guard-overhead measurement by preset policy. |
| OpenLLaMA 7B causal LM | `openlm-research/open_llama_7b` | `promoted_published_basis` | `published_basis` | Promoted with release-profile container-backed public report, runtime manifest, and signed evidence pack. |
| Phi-3 Mini 4K Instruct causal LM | `microsoft/Phi-3-mini-4k-instruct` | `explicitly_out_of_scope` | `implemented_coverage` | The public Phi surface remains the shipped Phi-4 text-only lane for this wave. |
| Falcon 7B causal LM | `tiiuae/falcon-7b` | `promoted_published_basis` | `published_basis` | Promoted with release-profile container-backed public report, runtime manifest, and signed evidence pack. |
| Broader BERT-like MLMs (DistilBERT/ALBERT/DeBERTa/ELECTRA) | `distilbert-base-uncased` | `blocked_missing_artifacts` | `implemented_coverage` | Loader and adapter tests exist for representative DistilBERT and DeBERTa checkpoints, and the repo now ships a lane preset plus calibration config with dry-run sweep coverage, but approved calibration/evaluation evidence is still missing. |
| mBART large 50 seq2seq | `facebook/mbart-large-50` | `explicitly_out_of_scope` | `implemented_coverage` | FLAN-T5 base now supplies the concrete public seq2seq basis; this broader mBART candidate stays out of scope until it has its own preset, calibration config, smoke evidence, and public run. |

The machine-readable criterion-by-criterion ledger for this candidate set lives
under `promotion_candidates_text_le_14b` in
`contracts/model_family_catalog.json`.

## Recommended Additions

| Priority | Family | Planned support mode | Representative models | Notes |
| --- | --- | --- | --- | --- |
| `P2` | Audio-text evaluation pipeline | `phase2_audio_eval` | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` | Image-text evaluation is included. The remaining multimodal backlog item is audio-capable evaluation for the smaller Gemma 4 checkpoints. |

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
