# Configuration Gallery

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Quick pointers to common presets and overlays. |
| **Audience** | Users looking for ready-to-use configurations. |
| **Note** | Presets are repo assets, not included in wheels. |
| **Source** | `configs/presets/` and `configs/overlays/`. |

Pointers to common presets in this repository you can start from. Presets are
repo assets (not included in wheels). Use flag‑only `invarlock evaluate` when
installing from PyPI, or clone this repo to reference these files.

Note: HF-backed `invarlock evaluate` flows require the evaluation stack from
`invarlock[hf]`. The narrower `invarlock[adapters]` extra is enough for local
adapter loading and plugin inspection, but it intentionally omits dataset and
download helpers. The core install (`pip install invarlock`) remains torch-free.

The `evaluate` examples below use the runtime container by default. Add
`--execution-mode host` only for host-side workflows that intentionally bypass that
boundary.

Most preset files intentionally keep small YAML `preview_n` / `final_n` values
so they remain fast and portable for repo smokes. For balanced-tier evaluations
that are expected to clear the normal token-floor gates, keep the preset and
run it with `--profile ci` or `--profile release`.

## Presets (Runnable)

### Causal LM (decoder-only)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/causal_lm/wikitext2_512.yaml` | Standard evaluation | Decoder-only causal | WikiText-2 |

**When to use:** Primary preset for causal language models. 512-token sequences
provide good coverage while keeping runtime reasonable.

```bash
invarlock evaluate --allow-network --baseline gpt2 --subject /path/to/edited \
  --preset configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

### Masked LM (BERT, RoBERTa, etc.)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/masked_lm/wikitext2_128.yaml` | Standard MLM evaluation | BERT/RoBERTa | WikiText-2 |
| `configs/presets/masked_lm/synthetic_128.yaml` | Offline testing | BERT/RoBERTa | Synthetic |

**When to use:** MLM presets for BERT-family models. Use synthetic preset when
network access is unavailable or for CI smoke tests.

```bash
invarlock evaluate --allow-network --baseline bert-base-uncased --subject /path/to/edited \
  --preset configs/presets/masked_lm/wikitext2_128.yaml --profile ci
```

### Seq2Seq (T5, etc.)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/seq2seq/synth_64.yaml` | Quick seq2seq tests | T5/BART-style encoder-decoder | Synthetic |
| `configs/presets/seq2seq/synth_128.yaml` | Longer seq2seq smoke runs | T5/BART-style encoder-decoder | Synthetic |
| `configs/presets/seq2seq/flan_t5_base_cnn_dailymail_256.yaml` | Public-basis FLAN-T5 preservation evidence | FLAN-T5 encoder-decoder | CNN/DailyMail validation |

**When to use:** Encoder-decoder models. Synthetic data keeps runs offline and
fast for smoke testing. The FLAN-T5 preset uses a pinned public Hugging Face
dataset revision and is intended for release-style evidence runs.

## Edit Overlays (Demo RTN Quantization)

These overlays apply the built-in `quant_rtn` edit for demonstration. For
production, use [Compare & evaluate (BYOE)](compare-and-evaluate.md) with your
own pre-edited checkpoint instead.

| Overlay | Scope | Use Case |
| --- | --- | --- |
| `configs/overlays/edits/quant_rtn/8bit_attn.yaml` | Attention layers only | Conservative quantization demo |
| `configs/overlays/edits/quant_rtn/8bit_full.yaml` | All linear layers | Full model quantization demo |
| `configs/overlays/edits/quant_rtn/tiny_demo.yaml` | Minimal layers | Quick smoke test |

**Example (demo edit):**

```bash
invarlock evaluate --allow-network --baseline gpt2 --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
  --profile ci
```

## Model And Dataset Overlays

These partial overlays are useful when composing local presets:

| Overlay | Scope | Use Case |
| --- | --- | --- |
| `configs/overlays/models/hf_causal.yaml` | Model adapter | Hugging Face causal LM defaults |
| `configs/overlays/models/hf_mlm.yaml` | Model adapter | Hugging Face masked LM defaults |
| `configs/overlays/datasets/wikitext2.yaml` | Dataset | WikiText-2 validation windows |

## Profiles

Profiles control window counts and bootstrap depth:

| Profile | Windows | Bootstrap | Use Case |
| --- | --- | --- | --- |
| `ci` | 240/240 | 1200 | Standard CI evaluation |
| `release` | 400/400 | 3200 | Production releases |
| `ci_cpu` | 120/120 | 1000 fallback | CPU-only environments |

## Tips

- Use `--profile ci|release|ci_cpu` to apply runtime window counts and
  bootstrapping defaults.
- Keep `seq_len = stride` for deterministic non‑overlapping windows.
- Combine a preset with an edit overlay using `--preset` and `--edit-config`.
- For custom data, see [Bring Your Own Data](bring-your-own-data.md).

## Related Documentation

- [Configuration Schema](../reference/config-schema.md) — All config options
- [CLI Reference](../reference/cli.md) — Command flags and profiles
- [Compare & evaluate (BYOE)](compare-and-evaluate.md) — Production workflow
- [Dataset Providers](../reference/datasets.md) — Available data sources
