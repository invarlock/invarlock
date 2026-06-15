# Dataset Providers

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Deterministic dataset providers for preview/final evaluation windows. |
| **Audience** | CLI users configuring `dataset` blocks and Python callers building evaluation windows. |
| **Supported providers** | `wikitext2`, `synthetic`, `hf_text`, `local_jsonl`, `vision_text`, `hf_seq2seq`, `local_jsonl_pairs`, `seq2seq`. |
| **Requires** | `invarlock[eval]` or `invarlock[hf]` for Hugging Face datasets providers. |
| **Network** | Offline by default; CLI runs use `evaluate --allow-network` for first download, while programmatic callers can set `INVARLOCK_ALLOW_NETWORK=1`. |
| **Inputs** | Dataset provider name plus provider-specific fields. |
| **Outputs / Artifacts** | Evaluation windows stored in `report.evaluation_windows` and dataset metadata in `report.data.*`. `vision_text` persists example records instead of token windows. |
| **Source of truth** | `src/invarlock/eval/data.py`, `src/invarlock/eval/data_support.py`, `src/invarlock/eval/data_tokenization.py`, and `src/invarlock/eval/data_providers.py`. |

## Quick Start

```yaml
dataset:
  provider: wikitext2
  split: validation
  seq_len: 512
  stride: 512
  preview_n: 64
  final_n: 64
  seed: 42
```

For Compare & evaluate, reuse the same `dataset` block in baseline and subject runs.

## Concepts

- **Preview vs final windows**: the runner computes the primary metric on two
  deterministic splits; counts are recorded in run reports and evaluation reports.
- **Pairing**: `invarlock evaluate` requires baseline window evidence to pair
  windows. Missing/invalid evidence fails closed in CI/Release profiles.
- **Offline-first**: downloads are opt-in. CLI runs use `evaluate --allow-network`;
  programmatic callers can set `INVARLOCK_ALLOW_NETWORK=1`. Cached datasets can
  be enforced via `HF_DATASETS_OFFLINE=1`.
- **Vision-text manifests**: `vision_text` is local-files-only and
  expects JSONL records with `id`, `image_path`, `prompt`, and either `answer`
  or `answers`. Records are single-image examples; provider batching can still
  group multiple records when callers request `batch_size > 1`.
- **Public image-text datasets**: public Hugging Face datasets can be used by
  materializing them first into a local `vision_text` manifest. The model
  evidence workflow uses
  `scripts/model_evidence/materialize_vision_text_dataset.py` for this pattern,
  so evaluation remains offline/hashable after the download step.
- **Image-text primary metric**: `vision_text` uses answer accuracy as the
  primary metric because the evidence claim is whether the generated answer
  matches the image question. Token log loss is still recorded as supporting
  telemetry, but perplexity is not the public VQA gate.
- **Tokenizer contract**: dataset providers expect either a callable tokenizer
  that returns `input_ids` plus optional `attention_mask`, or an `encode(...)`
  method that accepts `truncation=True`, `max_length=...`, and
  `padding="max_length"`.
- **Default runtime-container execution**: dataset-backed model-loading commands run in the
  runtime container by default; public host-side execution uses
  `invarlock evaluate --execution-mode host`.
- **Dedupe & capacity**: `INVARLOCK_DEDUP_TEXTS=1` removes exact duplicates;
  `INVARLOCK_CAPACITY_FAST=1` speeds up capacity checks for quick runs.
- **HF cache fallback**: if a local rerun hits a Hugging Face datasets
  shared-cache lock/permission error, InvarLock retries with its own writable
  datasets cache. Set `INVARLOCK_HF_DATASETS_CACHE` to choose that fallback
  location explicitly.

### Pairing invariants (E001)

| Invariant | Failure condition |
| --- | --- |
| `window_pairing_reason` | Must be empty / `None`. |
| `paired_windows` | Must be > 0. |
| `window_match_fraction` | Must be 1.0. |
| `window_overlap_fraction` | Must be 0.0. |

Counts mismatches are enforced via `coverage.preview.used`,
`coverage.final.used`, and `paired_windows` in `dataset.windows.stats`.

## Reference

### Provider matrix

| Provider | Kind | Network | Required keys | Notes |
| --- | --- | --- | --- | --- |
| `wikitext2` | text | Cache/Net | `provider`, `seq_len`, `stride`, `preview_n`, `final_n` | Deterministic n‑gram stratification; requires `datasets`. |
| `synthetic` | text | Offline | `provider`, `seq_len`, `preview_n`, `final_n` | Generated text; good for smoke tests. |
| `hf_text` | text | Cache/Net | `dataset_name`, `text_field` | Generic HF dataset loader; uses first N rows. |
| `local_jsonl` | text | Offline | `file`/`path`/`data_files`, `text_field` | Reads JSONL from disk; default `text_field: text`. |
| `vision_text` | image-text | Offline | `file`/`path`/`data_files` | Local JSONL manifest of single-image VQA-style examples; `stride` is ignored. |
| `hf_seq2seq` | seq2seq | Cache/Net | `dataset_name`, `src_field`, `tgt_field` | Provides encoder ids + decoder labels; supports pinned dataset `revision` and source/target prefixes. |
| `local_jsonl_pairs` | seq2seq | Offline | `file`/`path`/`data_files`, `src_field`, `tgt_field` | Paired JSONL for seq2seq. |
| `seq2seq` | seq2seq | Offline | optional `n`, `src_len`, `tgt_len` | Synthetic seq2seq generator. |

### Provider field map

| Provider | Required keys | Evidence fields (run report / evaluation report) |
| --- | --- | --- |
| `wikitext2` | `provider`, `seq_len`, `stride`, `preview_n`, `final_n` | `report.data.*` + `report.dataset.windows.stats` |
| `synthetic` | `provider`, `seq_len`, `preview_n`, `final_n` | `report.data.*` + `report.dataset.windows.stats` |
| `hf_text` | `dataset_name`, `text_field` | `report.data.*` + `report.dataset.windows.stats` |
| `local_jsonl` | `file`/`path`/`data_files`, `text_field` | `report.data.*` + `report.dataset.windows.stats` |
| `vision_text` | `file`/`path`/`data_files` | `report.data.*` + `report.evaluation_windows.{preview,final}.records` |
| `hf_seq2seq` | `dataset_name`, `src_field`, `tgt_field` | `report.data.*` + `report.dataset.windows.stats` |
| `local_jsonl_pairs` | `file`/`path`/`data_files`, `src_field`, `tgt_field` | `report.data.*` + `report.dataset.windows.stats` |
| `seq2seq` | optional `n`, `src_len`, `tgt_len` | `report.data.*` + `report.dataset.windows.stats` |

Provider-specific config fields (dataset name, paths, fields) are recorded under
`report.data` when available.

### Pairing evidence matrix

| Config keys | Report fields | report fields | Verify gate |
| --- | --- | --- | --- |
| `dataset.provider`, `seq_len`, `stride`, `split` | `report.data.{dataset,seq_len,stride,split}` | `report.dataset.{provider,seq_len,windows}` | Schema + pairing context. |
| `dataset.preview_n/final_n` | `report.data.{preview_n,final_n}`, `report.evaluation_windows` | `report.dataset.windows.{preview,final}` | Pairing + count checks. |
| Pairing stats (derived) | `report.dataset.windows.stats` | `report.dataset.windows.stats` | `_validate_pairing` + `_validate_counts`. |
| Provider digest | `report.provenance.provider_digest` | `report.provenance.provider_digest` | Required in CI/Release. |

### HF text provider example

```yaml
dataset:
  provider: hf_text
  dataset_name: Salesforce/wikitext
  config_name: wikitext-2-raw-v1
  text_field: text
  split: validation
  preview_n: 64
  final_n: 64
```

### Local JSONL provider example

```yaml
dataset:
  provider: local_jsonl
  path: /data/my_corpus
  text_field: text
  preview_n: 64
  final_n: 64
```

### Vision-text provider example

```yaml
dataset:
  provider:
    kind: vision_text
    path: tests/fixtures/vision_text/demo_manifest.jsonl
  split: validation
  seq_len: 256
  preview_n: 1
  final_n: 1
```

### Public VQA materialization example

```bash
python scripts/model_evidence/materialize_vision_text_dataset.py \
  --dataset Multimodal-Fatima/VQAv2_sample_validation \
  --split validation \
  --revision 99487d2651df3799002b2fb3e455741744514a02 \
  --output-dir artifacts/model-evidence/public_datasets/vqav2_sample_validation_800 \
  --max-samples 800 \
  --image-field image \
  --prompt-field question \
  --answer-field multiple_choice_answer \
  --answers-field answers \
  --id-field question_id \
  --prompt-template '{question}
Return exactly one JSON object like {{"answer":"short phrase"}}. Use a short phrase only. Do not explain.' \
  --overwrite
```

The generated `manifest.jsonl`, `images/`, and
`materialization_summary.json` are then consumed by `vision_text`. For evidence
promotion, pin the dataset revision and keep the materialization summary with
the run artifacts. Public VQA evidence prompts should prefer a structured
answer field such as `{"answer":"..."}`; the evaluator extracts that field
before exact-answer scoring and falls back to the raw generation when no JSON
answer is present.

### Seq2seq provider example (HF)

```yaml
dataset:
  provider:
    kind: hf_seq2seq
    dataset_name: abisee/cnn_dailymail
    config_name: 3.0.0
    revision: 96df5e686bee6baa90b8bee7c28b81fa3fa6223d
    src_field: article
    tgt_field: highlights
    src_prefix: "summarize: "
    max_samples: 1024
  split: validation
  seq_len: 256
  preview_n: 32
  final_n: 32
```

The FLAN-T5 public seq2seq basis uses this provider shape with
`google/flan-t5-base` pinned to model revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.

### Environment variables

- `INVARLOCK_ALLOW_NETWORK=1` — allow dataset downloads.
- `HF_DATASETS_OFFLINE=1` — force cached-only datasets.
- `INVARLOCK_DEDUP_TEXTS=1` — exact-text dedupe before tokenization.
- `INVARLOCK_CAPACITY_FAST=1` — approximate capacity estimation for quick runs.
- `INVARLOCK_HF_DATASETS_CACHE=/path/to/cache` — override the writable fallback
  cache used after shared-cache lock/permission failures.

## Troubleshooting

- **`DEPENDENCY-MISSING: datasets`**: install `invarlock[eval]` or `invarlock[hf]`.
- **`NO-SAMPLES` / `NO-PAIRS` errors**: verify dataset fields and split names.
- **HF cache `.lock` / permission errors on local reruns**: rerun as-is to use
  the automatic writable-cache fallback, or set
  `INVARLOCK_HF_DATASETS_CACHE` to a writable directory you control.
- **`vision_text image file is missing`**: ensure manifest `image_path` values
  resolve relative to the JSONL file and point to readable local files.
- **Pairing failures (`E001`)**: ensure baseline `report.json` contains
  `evaluation_windows` and was produced with matching dataset settings.

## Observability

- `report.data.*` stores provider name, split, and window counts.
- `report.evaluation_windows` stores preview/final token windows.
- reports preserve dataset metadata and window pairing stats under `dataset.*`.

## Related Documentation

- [Configuration Schema](config-schema.md)
- [Environment Variables](env-vars.md)
- [CLI Reference](cli.md)
- [reports](reports.md) — Schema, telemetry, and HTML export
- [Coverage & Pairing](../assurance/02-coverage-and-pairing.md) — Window requirements and pairing math
- [Bring Your Own Data](../user-guide/bring-your-own-data.md) — Custom dataset workflows
