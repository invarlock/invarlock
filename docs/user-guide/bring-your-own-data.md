# Bring Your Own Data (BYOD)

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Run InvarLock on custom datasets with offline-capable providers. |
| **Audience** | Users with proprietary or custom text corpora for evaluation. |
| **Supported providers** | `local_jsonl` (hermetic, no deps), `hf_text` (requires `datasets`). |
| **Network** | `local_jsonl` is fully offline; `hf_text` needs network on first fetch only. |
| **Source of truth** | `src/invarlock/eval/providers/text_lm.py`. |

Run InvarLock on your own small text files with zero network access. Two options:


- hf_text: use Hugging Face Datasets (json/text) with `datasets` installed
- local_jsonl: read a local JSONL file with a `text` field (no extra deps)

## Option A: Local JSONL (hermetic)

Config snippet:

```yaml
model:
  id: gpt2
  adapter: hf_causal

dataset:
  provider: { kind: local_jsonl }
  file: /absolute/path/to/data.jsonl   # each line: {"text": "..."}
  seq_len: 128
  stride: 64
  preview_n: 50
  final_n: 50

guards:
  order: []

eval:
  metric: { kind: ppl_causal }
```

Tips:

- The file must exist and contain one JSON object per line with a `text` field. Doctor validates this and will report "path does not exist" if missing.
- Works fully offline; great for CI and local smokes.
- You can set `text_field` when your column name differs from `text`.

## Option B: Hugging Face Datasets (JSON)

Requires `pip install datasets` and network only when first fetching data.

```yaml
model:
  id: gpt2
  adapter: hf_causal

dataset:
  provider: { kind: hf_text }
  dataset_name: json
  # When using json builder, pass local files via CLIs (use invarlock doctor for hints)
  text_field: text
  seq_len: 128
  stride: 64
  preview_n: 50
  final_n: 50

guards:
  order: []

eval:
  metric: { kind: ppl_causal }
```

## End-to-end example (local JSONL)

```bash
# 1) Create a small JSONL
printf '{"text":"hello world"}\n{"text":"custom data"}\n' > /tmp/byod.jsonl

# 2) Write config (start from configs/presets/causal_lm/wikitext2_512.yaml and adjust dataset to your BYOD)
# 3) Run baseline and subject
invarlock run -c byod.yaml --profile dev --out runs/base
invarlock run -c byod.yaml --profile dev --out runs/subj

# 4) Generate certificate
invarlock report --run runs/subj --format cert --baseline runs/base -o cert
```

Common pitfalls:

- Set `dataset.split` explicitly for HF datasets to avoid split fallback warnings.
- Keep `preview_n` and `final_n` modest for quick smokes; increase in release profiles.
- When using BYOD providers, configs should validate against the dataset provider schema (see tests/schemas/dataset_provider.schema.json for a reference shape).

Evidence debug:

- Set `INVARLOCK_EVIDENCE_DEBUG=1` to also emit a small `guards_evidence.json` with decision inputs next to the certificate; the manifest will include a pointer to this file.
