# Configuration Gallery

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Quick pointers to common presets and overlays. |
| **Audience** | Users looking for ready-to-use configurations. |
| **Note** | Presets are repo assets, not shipped in wheels. |
| **Source** | `configs/presets/` and `configs/overlays/`. |

Pointers to common presets in this repository you can start from. Presets are
repo assets (not shipped in wheels). Use flag‑only `invarlock certify` when
installing from PyPI, or clone this repo to reference these files.

Note: Adapter‑based flows such as `invarlock certify` and `invarlock run` with
HF models require extras like `invarlock[hf]` or `invarlock[adapters]`. The
core install (`pip install invarlock`) remains torch‑free.

## Presets (Runnable)

### Causal LM (decoder-only)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/causal_lm/wikitext2_512.yaml` | Standard certification | Decoder-only causal | WikiText-2 |

**When to use:** Primary preset for causal language models. 512-token sequences
provide good coverage while keeping runtime reasonable.

```bash
invarlock certify --baseline gpt2 --subject /path/to/edited \
  --preset configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

### Masked LM (BERT, RoBERTa, etc.)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/masked_lm/wikitext2_128.yaml` | Standard MLM certification | BERT/RoBERTa | WikiText-2 |
| `configs/presets/masked_lm/synthetic_128.yaml` | Offline testing | BERT/RoBERTa | Synthetic |

**When to use:** MLM presets for BERT-family models. Use synthetic preset when
network access is unavailable or for CI smoke tests.

```bash
invarlock certify --baseline bert-base-uncased --subject /path/to/edited \
  --preset configs/presets/masked_lm/wikitext2_128.yaml --profile ci
```

### Seq2Seq (T5, etc.)

| Preset | Use Case | Model Type | Dataset |
| --- | --- | --- | --- |
| `configs/presets/seq2seq/synth_64.yaml` | Quick seq2seq tests | T5 | Synthetic |

**When to use:** Encoder-decoder models. Synthetic data keeps runs offline and
fast for smoke testing.

## Edit Overlays (Demo RTN Quantization)

These overlays apply the built-in `quant_rtn` edit for demonstration. For
production, use [Compare & Certify (BYOE)](compare-and-certify.md) with your
own pre-edited checkpoint instead.

| Overlay | Scope | Use Case |
| --- | --- | --- |
| `configs/overlays/edits/quant_rtn/8bit_attn.yaml` | Attention layers only | Conservative quantization demo |
| `configs/overlays/edits/quant_rtn/8bit_full.yaml` | All linear layers | Full model quantization demo |
| `configs/overlays/edits/quant_rtn/tiny_demo.yaml` | Minimal layers | Quick smoke test |

**Example (demo edit):**

```bash
invarlock certify --baseline gpt2 --subject gpt2 \
  --preset configs/presets/causal_lm/wikitext2_512.yaml \
  --edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
  --profile ci
```

## Profiles

Profiles control window counts and bootstrap depth:

| Profile | Windows | Bootstrap | Use Case |
| --- | --- | --- | --- |
| `ci` | 200/200 | 1200 | Standard CI certification |
| `release` | 400/400 | 3200 | Production releases |
| `ci_cpu` | 120/120 | 1200 | CPU-only environments |

## Tips

- Use `--profile ci|release|ci_cpu` to apply runtime window counts and
  bootstrapping defaults.
- Keep `seq_len = stride` for deterministic non‑overlapping windows.
- Combine presets with edit overlays using multiple `-c` flags or `--edit-config`.
- For custom data, see [Bring Your Own Data](bring-your-own-data.md).

## Related Documentation

- [Configuration Schema](../reference/config-schema.md) — All config options
- [CLI Reference](../reference/cli.md) — Command flags and profiles
- [Compare & Certify (BYOE)](compare-and-certify.md) — Production workflow
- [Dataset Providers](../reference/datasets.md) — Available data sources
