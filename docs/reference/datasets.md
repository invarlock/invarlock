# Dataset Providers

InvarLock uses pluggable dataset providers with deterministic windowing:

| Provider   | Description                                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `wikitext2`| Default language modeling dataset used for certification examples (downloads via HuggingFace `datasets` if allowed).                         |
| `synthetic`| Minimal random-text provider for smoke tests and offline CI flows.                                                                           |
| `hf_text`  | Generic HuggingFace text dataset provider (set `dataset_name`, optional `config_name`, and `text_field`).                                    |

The `wikitext2` provider uses a fixed byte‑level n‑gram difficulty scorer to
stratify candidate windows. It is deterministic, offline, and tokenizer‑agnostic
to keep window selection consistent across model families.

## Online vs Offline

- Online: set `dataset.provider: wikitext2` and export `INVARLOCK_ALLOW_NETWORK=1`
  so the HuggingFace `datasets` library can fetch the data on first use.
- Offline:
  - Pre-download the dataset (e.g., using `datasets` in a connected
    environment), so it exists in the local cache.
  - Unset `INVARLOCK_ALLOW_NETWORK` and optionally set `HF_DATASETS_OFFLINE=1` to
    enforce offline reads.
  - For completely synthetic/offline smoke tests, set `dataset.provider:
    synthetic`.

## Example (CI preset)

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

> Use the same `dataset` section for both baseline and edited runs; pass
> `--baseline <baseline report.json>` on the edit run to pair windows.
>
> Baseline pairing is evidence-based: the baseline `report.json` must preserve
> `evaluation_windows` (tokenized window inputs/masks + stable window IDs). In
> `--profile ci` / `--profile release`, `--baseline` fails closed with
> `[INVARLOCK:E001]` if the baseline is missing or has invalid window evidence.
> In dev profile, InvarLock may fall back to an unpaired schedule with a loud
> warning.

Inspect providers through the library API:

```python
from invarlock.eval import data as eval_data

print(eval_data.list_providers())
provider = eval_data.get_provider("wikitext2")
preview, final = provider.windows(
    tokenizer,
    preview_n=64,
    final_n=64,
    seq_len=512,
    stride=512,
)
```

This matches the registry the runner uses, so programmatic experiments and
scripts see the same providers as CLI runs.

## Generic HF Text Dataset

Use the `hf_text` provider to load a HuggingFace dataset by name:

```yaml
dataset:
  provider: hf_text
  dataset_name: wikitext          # any HF dataset name
  config_name: wikitext-2-raw-v1  # optional
  text_field: text                # defaults to 'text'
  split: validation
  preview_n: 64
  final_n: 64
```

Notes:

- Online mode requires `INVARLOCK_ALLOW_NETWORK=1` on first use; subsequent runs
  read from HF cache.
- `hf_text` uses a simple deterministic selection (first N for preview and next
  N for final) and a straightforward tokenizer pass.

### Pre-download snippet (offline cache)

Use the `datasets` library to pre-download data in a connected environment:

```python
from datasets import load_dataset

# Pick dataset/config/split
ds = load_dataset("wikitext", name="wikitext-2-raw-v1", split="validation")

# Access a few rows to force materialization/caching
_ = [row["text"] for row in ds.select(range(10))]

print("Cached at:", ds.cache_files)
```

Then move the HF cache directory to your offline machine (or set
`HF_HOME`/`HF_DATASETS_CACHE` to a persistent location). Run InvarLock with
`INVARLOCK_ALLOW_NETWORK` unset and optionally `HF_DATASETS_OFFLINE=1`.

---

## Environment Variables

- INVARLOCK_CAPACITY_FAST=1 — approximate capacity estimator that skips the full
  capacity/dedupe pass to speed up pilots and smoke tests.
  This is intended for experiments and quick checks, not certification runs.

  Example:

  ```bash
  INVARLOCK_CAPACITY_FAST=1 invarlock run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci
  ```

  Note: use for quick checks; not suitable for release evidence.

- INVARLOCK_DEDUP_TEXTS=1 — exact-text dedupe before tokenization to reduce
  duplicate windows and stabilize overlap metrics; preserves first occurrence order.

  Example:

  ```bash
  INVARLOCK_DEDUP_TEXTS=1 invarlock run -c configs/tasks/causal_lm/ci_cpu.yaml --profile ci
  ```
