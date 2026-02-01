# InvarLock — Edit‑agnostic robustness reports for weight edits

In short: evaluate that weight edits (e.g., quantization) preserve quality. If
they don’t, roll back safely.

Technical: edit‑agnostic guard pipeline (invariants → spectral → RMT →
variance) producing a machine‑readable Evaluation Report.

> **Status:** 0.3.7 (pre‑1.0). Until 1.0, **minor** releases may be
> breaking. See CLI help and the CHANGELOG for updates.

[![CI](https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&logo=github&label=CI)](https://github.com/invarlock/invarlock/actions/workflows/ci.yml)
[![PyPI](https://badge.fury.io/py/invarlock.svg)](https://pypi.org/project/invarlock/)
[![Docs](https://img.shields.io/badge/docs-quickstart-blue.svg)](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/quickstart.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
---

For guidance on where to ask questions, how to report bugs, and what to expect in terms of response times, see
[SUPPORT.md](https://github.com/invarlock/invarlock/blob/main/SUPPORT.md).

## 🚀 Quick start (no repo clone)

Notebooks (Colab):

- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_quickstart_cpu.ipynb)
  `invarlock_quickstart_cpu.ipynb` — install + evaluate + verify + HTML export (CPU-friendly)
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_compare_evaluate.ipynb)
  `invarlock_compare_evaluate.ipynb` — Compare & Evaluate (BYOE) end-to-end
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_evaluation_report_deep_dive.ipynb)
  `invarlock_evaluation_report_deep_dive.ipynb` — reading and interpreting evaluation reports
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_custom_datasets.ipynb)
  `invarlock_custom_datasets.ipynb` — Bring Your Own Data (BYOD) with `local_jsonl`
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_python_api.ipynb)
  `invarlock_python_api.ipynb` — programmatic Python API usage
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_policy_tiers.ipynb)
  `invarlock_policy_tiers.ipynb` — Conservative vs Balanced vs Aggressive tier comparison

```bash
# Install with HF adapters
pip install "invarlock[hf]"

# Fast dev self‑cert on GPT‑2 small (tiny‑relax; downloads require explicit network)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 INVARLOCK_TINY_RELAX=1 \
invarlock evaluate \
  --baseline gpt2 \
  --subject  gpt2 \
  --adapter auto \
  --profile dev
```

This produces `reports/.../evaluation.report.json` with paired metrics
(ppl/accuracy), structural deltas, spectral/RMT stats, variance‑estimator
provenance, seeds/hashes, pairing metrics, and a policy digest.

> **Calibration note:** tier thresholds and window sizes are piloted on GPT‑2 small
> and BERT base (see `docs/assurance/09-tier-v1-calibration.md`). For
> calibrated Balanced/Conservative evaluation reports, use the preset‑based CI/Release examples
> below. `INVARLOCK_TINY_RELAX` dev runs relax sample‑size floors and are intended
> only for small smoke tests (not release evidence).

> Need presets or matrix scripts? Clone this repo and see Presets & Demos below.

---

## 📚 Docs & Guides

- Quickstart: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/quickstart.md>
- Compare & Evaluate (BYOE): <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/compare-and-evaluate.md>
- Reading a report: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/reading-report.md>
- CLI reference: <https://github.com/invarlock/invarlock/blob/main/docs/reference/cli.md>

Quick examples (repo presets, CPU; repo clone required for preset paths):

```bash
# Install with HF adapters
pip install "invarlock[hf]"

# Preflight a config (JSON diagnostics)
invarlock doctor --config configs/presets/causal_lm/wikitext2_512.yaml --json

# Calibrated GPT‑2 small (recommended starting point; repo preset)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 \
invarlock evaluate \
  --baseline gpt2 \
  --subject  gpt2 \
  --adapter auto \
  --profile release \
  --preset configs/presets/causal_lm/wikitext2_512.yaml

# Tiny causal LM smoke (out‑of‑calibration, dev‑only)
INVARLOCK_ALLOW_NETWORK=1 \
invarlock evaluate \
  --baseline hf:sshleifer/tiny-gpt2 \
  --subject  hf:sshleifer/tiny-gpt2 \
  --profile dev
```

Notes:

- Presets and scripts live in this repo (`configs/`, `scripts/`) and are not
  shipped in wheels. Use flag‑only `evaluate` when installing from PyPI, or clone
  this repo to use presets and the matrix script.
- `python -m invarlock` works the same as `invarlock`.
- InvarLock runs offline by default; enable network per command with `INVARLOCK_ALLOW_NETWORK=1` when fetching.

---

## 🔧 Installation

```bash
# Core + HF adapter
pip install "invarlock[hf]"

# GPU extras (CUDA wheels if available)
pip install "invarlock[gpu]"

# Optional edit backends
pip install "invarlock[awq,gptq]"     # AWQ/GPTQ PTQ stacks
pip install "invarlock[dev]"          # dev tooling (ruff, pytest, mkdocs)
```

> Minimal core installs with `pip install invarlock`. The OSS core is edit‑agnostic
> (BYOE): supply baseline and subject checkpoints and run Compare & Evaluate. A
> small built‑in edit, `quant_rtn`, is provided for CI/quickstart demos only;
> optional extras (e.g., `gptq`, `awq`, `gpu`) are loaders/runtimes, not edit
> pipelines. Core installs do not pull in torch/transformers; those are only
> installed when you opt into extras such as `"invarlock[hf]"` or
> `"invarlock[adapters]"`.

Run either entry point:

```bash
invarlock --help
python -m invarlock --help
```

Common error (missing torch on adapter-based commands):

```text
❌ Torch is required for this command.
Install extras with: pip install "invarlock[hf]" or "invarlock[adapters]".
```

If you see this, install an appropriate extra (for example, `pip install "invarlock[hf]"`)
before running `invarlock run` or `invarlock evaluate` with HF adapters.

### Network Access

- Outbound network is disabled by default for safety. Enable it explicitly (per
  command) when you need to download models or datasets:

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \
  --baseline gpt2 \
  --subject  gpt2 \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
```

- Offline/air‑gapped usage: pre‑download to a cache, then run with network
  disabled. You can enforce offline reads with `HF_DATASETS_OFFLINE=1` (and
  optionally set `HF_HOME`/`HF_DATASETS_CACHE` to your cache location).

See the CLI reference and datasets guide for details:

- <https://github.com/invarlock/invarlock/blob/main/docs/reference/cli.md>
- <https://github.com/invarlock/invarlock/blob/main/docs/reference/datasets.md>

### Install via pipx (isolated)

```bash
# Ensure pipx uses Python 3.12+
pipx install --python python3.12 "invarlock[hf]"  # Python 3.12+ recommended

# With GPU extras (if supported on your platform)
pipx install --python python3.12 "invarlock[hf,gpu]"
```

### Conda environment recipe

```bash
conda create -n invarlock python=3.12 -y
conda activate invarlock

# Core + HF stack
pip install "invarlock[hf]"

# Optional extras
# pip install "invarlock[gpu]"
# pip install "invarlock[awq,gptq]"
```

---

## 💻 Support Matrix

<!-- markdownlint-disable MD060 -->
| Platform               | Status          | Notes                                     |
| ---------------------- | --------------- | ----------------------------------------- |
| Python 3.12+           | ✅ Required      |                                           |
| Linux                  | ✅ Full          | Primary dev target                        |
| macOS (Intel/M-series) | ✅ Full          | MPS supported (default on Apple Silicon)  |
| Windows                | ❌ Not supported | Use WSL2 or a Linux container if required |
| CUDA                   | ✅ Recommended   | For larger models                         |
| CPU                    | ✅ Fallback      | Slower but functional                     |
<!-- markdownlint-enable MD060 -->

**Device selection:** CUDA → MPS → CPU (auto). Override with torch env if
needed (e.g., `CUDA_VISIBLE_DEVICES`).

---

## 🧱 What InvarLock Provides

- **Runner** (torch-agnostic core): `prepare → preview → apply → guards → evaluate → report/rollback`

 - **Built-in edit**:
   - `quant_rtn` (INT8 RTN, per‑channel, clamp/group size)

- **Guards** (policy-tiered; “GuardChain” = ordered guard pipeline):

  1. **Invariants** (pre/post: shapes/finite/tying)
  2. **Spectral** (per-family z-caps; monitor or gate per tier)
  3. **RMT** (ε-band on outliers; monitor or gate per tier)
  4. **Variance (VE)** (predictive paired ΔlogNLL gate; tiered sidedness)

- **Evaluation Report (schema v1, PM‑only)**: Primary Metric (ppl or
  accuracy) with paired statistics, structural deltas, spectral/RMT stats, VE
  provenance, seeds/hashes, pairing metrics, and **policy digest**. Canonical
  artifact: `reports/.../evaluation.report.json`.

**Scope (what InvarLock does / does not do):**

- InvarLock certifies **regression risk from weight edits** (e.g., quantization or
  pruning) relative to a fixed baseline under a specific configuration.
- It focuses on **paired primary metrics** (ppl/accuracy) plus structural and
  guard telemetry (invariants, spectral, RMT, variance) for those edits.
- It **does not** claim to solve content‑safety problems (toxicity, bias,
  jailbreaks) or alignment in general, and it does not evaluate arbitrary
  training changes or new datasets.
- It is calibrated and tested on Linux/macOS environments using the HF/PyTorch
  stack described in the docs; native Windows is not supported.
- For the detailed assurance case and threat model, see
  `docs/assurance/00-safety-case.md` and `docs/security/threat-model.md`.

Minimal excerpt (redacted):

```json
{
  "schema_version": "v1",
  "run_id": "...",
  "validation": {
    "primary_metric_acceptable": true,
    "guard_overhead_acceptable": true
  },
  "primary_metric": {
    "kind": "ppl_causal",
    "preview": 12.3,
    "final": 12.1,
    "ratio_vs_baseline": 0.98,
    "display_ci": [0.97, 0.99]
  },
  "structure": {"layers_modified": 0, "params_changed": 0},
  "spectral": {"caps_applied": 0},
  "rmt": {"stable": true},
  "auto": {"tier": "balanced"}
}
```

---

## 🛡️ Guard Order & Balanced Defaults

**Canonical order**: `["invariants", "spectral", "rmt", "variance", "invariants"]`

**Balanced profile (example)**

```yaml
guards:
  spectral:
    mode: monitor
    sigma_quantile: 0.95
    deadband: 0.10
    scope: all
    max_caps: 5
    max_spectral_norm: null         # disable absolute clamp; rely on calibrated κ_f
    multiple_testing: { method: bh, alpha: 0.05, m: 4 }
    family_caps: { ffn: 2.5, attn: 2.8, embed: 3.0, other: 3.0 }   # z-caps (FPR-derived)
  rmt:
    mode: monitor
    epsilon_by_family: { ffn: 0.10, attn: 0.08, embed: 0.12, other: 0.12 }
  variance:
    tap: "post mlp.c_proj (pre-residual)"
    targets: "edited_modules_only"
    discovery:
      deadband: 0.02
      min_abs_adjust: 0.012
      max_scale_step: 0.03
    gating:
      sided: "one-sided"                     # improvement-only
      min_effect_lognll: 9e-4                # pilot-derived power threshold
```

> **Conservative** raises z-caps/ε/deadband/min-effect and uses **two-sided** VE; **Aggressive** relaxes accordingly.

---

> 🔍 For development and CI commands (pytest, mkdocs, generators), see CONTRIBUTING.md.

---

## ✂️ Edits & Plugins

- **Quant RTN** (built‑in): INT8 RTN, per‑channel, group size, percentile clamp
- **Compare & Evaluate (BYOE, recommended)**: Bring your baseline + subject checkpoints and evaluate with InvarLock
- **Plugins (optional)**: Adapters and guards via entry points. Adapters extend
  model loading/inference (e.g., GPTQ/AWQ formats); plugins do not add edit
  algorithms beyond RTN. List components with:

  ```bash
  invarlock plugins --help        # summary
  invarlock plugins guards        # guard plugins
  invarlock plugins edits         # edit plugins
  invarlock plugins adapters      # adapters and backend hints
  ```

---

## 🔁 Certification Criteria (balanced profile)

Key checks enforced by balanced policy (summary):

- **Pairing invariants**: preview = final counts; `match=1.00`, `overlap=0.00` (fail-fast in CI/Release)
- **PM ratio gate** (ppl or accuracy): upper CI ≤ **1.10**
- **Drift**: 0.95–1.05 (paired log-space)
- **Spectral/RMT**: within tier FPR/ε band
- **Catastrophe rollback**: automatic revert if PPL > **2.0×**
- **Guard overhead**: a bare-vs-guarded comparison records `validation.guard_overhead_acceptable=true` when ≤ 1 % PPL overhead


---

## 🧾 Minimal Config (balanced GPT-2, CI profile)

```yaml
model:
  id: "<set-your-model-id>"   # e.g., gpt2
  adapter: "hf_causal"
  device: "cpu"
dataset:
  provider: "wikitext2"
  split: "validation"
  seq_len: 512
  stride: 512
  preview_n: 64
  final_n: 64
  seed: 42
edit:
  # Optional: built-in quant demo. Omit for Compare & Evaluate/BYOE.
  name: quant_rtn
  plan:
    bitwidth: 8
    per_channel: true
    scope: attn
eval:
  metric:
    kind: ppl_causal
  loss:
    type: causal
guards:
  order: [invariants, spectral, rmt, variance, invariants]
  spectral: { mode: monitor }
  rmt: { mode: monitor }
  variance:
    tap: "post mlp.c_proj (pre-residual)"
    targets: "edited_modules_only"
    discovery: { deadband: 0.02, min_abs_adjust: 0.012, max_scale_step: 0.03 }
    gating: { sided: one-sided, min_effect_lognll: 9e-4 }
auto:
  enabled: true
  tier: balanced
  probes: 0
output:
  dir: runs
  save_model: false
  save_report: true
```

---

## 🩺 Doctor (preflight)

Run preflight checks before a run to catch misconfigurations early:

```bash
invarlock doctor --config configs/presets/causal_lm/wikitext2_512.yaml --json
```

Text mode emits lines prefixed with `ERROR:`, `WARNING:`, or `NOTE:` and stable
codes like `[INVARLOCK:D001]`. JSON mode includes `summary`, `policy`,
`findings[]`, `resolution`, and `format_version`.

---

## 🏗️ Source Layout (Single Distribution)

```text
invarlock/
├─ src/
│  ├─ invarlock/                 # core + unified namespace
│  │  ├─ core/               # runner, registry, contracts, events, ABI
│  │  ├─ cli/                # console app + command wrappers (unified import path)
│  │  ├─ adapters/           # model adapters (HF causal/MLM/seq2seq/onnx)
│  │  ├─ edits/              # quant_rtn
│  │  ├─ guards/             # invariants, spectral, rmt, variance
│  │  ├─ eval/               # evaluation metrics and helpers
│  │  ├─ reporting/          # report assembly, report generation/validation
│  │  ├─ assurance/          # assurance surface aggregating report helpers
│  │  ├─ plugins/            # built-in example plugins
│  │  └─ observability/      # monitoring/metrics/tracing wrappers
├─ configs/                  # presets (repo‑only; clone to use)
├─ docs/                     # user guides, reference, assurance notes
├─ scripts/                  # automation / QA helpers
└─ tests/                    # unit/integration/property tests

Note: The package exposes a single import namespace (`invarlock.*`). Presets/scripts are repo resources and not packaged in wheels.
```

---

## 📚 Documentation

- User Guide: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/getting-started.md>
- Quickstart: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/quickstart.md>
- Compare & Evaluate (BYOE): <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/compare-and-evaluate.md>
- Reading a report: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/reading-report.md>
- Assurance (proof notes): <https://github.com/invarlock/invarlock/tree/main/docs/assurance>
  - eval math, spectral FPR, RMT ε, VE gate power, determinism
- Config Schema: <https://github.com/invarlock/invarlock/blob/main/docs/reference/config-schema.md>
- Guard Reference: <https://github.com/invarlock/invarlock/blob/main/docs/reference/guards.md>

---

## ⚡ Quick CPU Demos (dev)

For tiny, CPU‑only demos that produce readable PASS banners in dev, enable
tiny‑relax and run the matrix script (repo clone required). This mode relaxes
primary‑metric token floors and is intended for smoke testing only (not release
evidence):

```bash
export INVARLOCK_TINY_RELAX=1 INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 \
       TRANSFORMERS_NO_TORCHVISION=1 TOKENIZERS_PARALLELISM=false
RUN=1 NET=1 bash scripts/run_tiny_all_matrix.sh
```

Add `INCLUDE_MEASURED_CLS=1` to include a measured classification step (requires warmed HF caches/network).

---

## 🧪 Determinism & Provenance

- Seeds: `{python, numpy, torch}` recorded in evaluation reports
- Dataset/tokenizer hashes recorded
- Paired non-overlapping windows (fail-fast if counts mismatch or pairing < 1.0)
- Cert math checks: `ppl_ratio.point == exp(mean ΔlogNLL)` and CI from the **same** paired Δ array

---

## 🤝 Contributing

```bash
make dev-install     # editable + dev tools (pytest, ruff, mypy, mkdocs, etc.)
make test            # run tests
make lint            # ruff + mypy
make format          # ruff format/fix
make docs            # build docs (mkdocs)
make verify          # tests, lint, format, markdownlint
```

Please see `CONTRIBUTING.md` for guidelines and `Makefile` for more targets.

---

## 📄 License

Apache-2.0 — see `LICENSE`.

---

### Notes

- PPL levels depend on `seq_len` (e.g., 768-token windows typically reduce PPL vs shorter contexts).
