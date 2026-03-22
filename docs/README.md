# InvarLock Documentation

InvarLock is edit‑agnostic (BYOE). A small built‑in quantization demo
(`quant_rtn`, 8‑bit) exists for CI/quickstart. See
[Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md).

Welcome to the documentation hub for InvarLock (Edit‑agnostic robustness reports for weight edits).
The material below is organized so new users can ramp quickly while practitioners
find detailed reference, design rationales, and assurance notes.

---

## Start Here

1. **[Getting Started](user-guide/getting-started.md)** – environment setup and the first evaluation loop.
2. **[Quickstart](user-guide/quickstart.md)** – CLI highlights for common workflows.
3. **[Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md)** – baseline ↔ subject with guardchain.
4. **[Primary Metric Smoke](user-guide/primary-metric-smoke.md)** – tiny examples for ppl/accuracy kinds.

### Quick Examples

```bash
# Core-only install (no torch/transformers): CLI + config tools
pip install invarlock

# HF/torch stack for adapter-based flows (evaluate/run)
pip install "invarlock[hf]"

# Compare & evaluate (BYOE checkpoints)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline <BASELINE_MODEL> \
  --subject  <SUBJECT_MODEL> \
  --adapter  auto
```

Tip: enable Hub downloads per command when fetching models/datasets:
`INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate ...`

Security-default note: `evaluate`, `run`, and `calibrate` use the runtime
container by default. Use `--allow-host-execution` only for trusted local
workflows that intentionally bypass that boundary.

---

## Documentation Map

### User Guide

- [Getting Started](user-guide/getting-started.md)
- [Quickstart](user-guide/quickstart.md)
- [Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md)
- [Primary Metric Smoke](user-guide/primary-metric-smoke.md)
- [Configuration Gallery](user-guide/config-gallery.md)
- [Example Reports](user-guide/example-reports.md)
- [Reading a report](user-guide/reading-report.md)
- [Troubleshooting](user-guide/troubleshooting.md) — Error codes and common fixes
- [Plugins](user-guide/plugins.md) — Extending adapters and guards
- [Bring Your Own Data](user-guide/bring-your-own-data.md) — Custom datasets
- [Proof Packs](user-guide/proof-packs.md) — Validation suite bundles
- [Proof Packs Internals](user-guide/proof-packs-internals.md) — Suite architecture and preset derivation flow

### Reference

- [Reference Index](reference/index.md)
- [CLI Reference](reference/cli.md)
- [Tier Policy Tuning CLI (Calibration)](reference/calibration.md) — `invarlock calibrate` for tier policy sweeps
- [Configuration Schema](reference/config-schema.md)
- [Guards](reference/guards.md)
- [Model Adapters](reference/model-adapters.md)
- [Model Family Catalog](reference/model-family-catalog.md)
- [reports](reference/reports.md) — Schema, telemetry, and HTML export
- [Tier Policy Catalog (runtime tiers.yaml)](reference/tier-policy-catalog.md)
- [Datasets](reference/datasets.md)
- [Artifact Layout](reference/artifacts.md)
- [Observability](reference/observability.md)
- [API Guide](reference/api-guide.md)
- [Programmatic Quickstart](reference/programmatic-quickstart.md)
- [Environment Variables](reference/env-vars.md)

<!-- Runbooks removed in minimal public footprint -->

<!-- Design docs removed in minimal public footprint -->

### Assurance

- [Assurance Case](assurance/00-assurance-case.md)
- [Evaluation Math Derivation](assurance/01-eval-math-derivation.md)
- [Coverage & Pairing Plan](assurance/02-coverage-and-pairing.md)
- [BCa Bootstrap (Paired Δlog)](assurance/03-bca-bootstrap.md)
- [Guard Contracts & Primer](assurance/04-guard-contracts.md)
- [Spectral False-Positive Control](assurance/05-spectral-fpr-derivation.md)
- [RMT ε-Rule](assurance/06-rmt-epsilon-rule.md)
- [VE Predictive Gate](assurance/07-ve-gate-power.md)
- [Determinism Contracts](assurance/08-determinism-contracts.md)
- [Tier v1.0 Calibration](assurance/09-tier-v1-calibration.md)
- [Guard Overhead Method](assurance/10-guard-overhead-method.md)
- [Policy Provenance & Digest](assurance/11-policy-provenance.md)
- [Device Drift Bands](assurance/12-device-drift-bands.md)
- [GPU/MPS-First Guards (Decision Memo)](assurance/13-gpu-mps-first-guards.md)

Note: Every assurance claim is backed by automated tests and cross-referenced in
the docs. See Guard Contracts → Coverage Reference
(assurance/04-guard-contracts.md) for the test index.

Calibration CSVs and proof reports referenced in these notes are produced by
local or CI runs (typically under `runs/null_sweeps/**` and
`reports/calibration/**`) and are not committed to the repository. Attach them
to change proposals or releases when you update calibration.

<!-- Developer docs removed in minimal public footprint. See project root CHANGELOG.md. -->

### Security

- [Threat Model](security/threat-model.md) — Assets and adversaries
- [Security Architecture](security/architecture.md) — Components and defaults
- [Best Practices](security/best-practices.md) — Operational recommendations
- [Release Verification](security/release-verification.md) — Offline verification of published release bundles
- [pip-audit Allowlist](security/pip-audit-allowlist.md)

### Governance

- [Contribution Guidelines](https://github.com/invarlock/invarlock/blob/main/CONTRIBUTING.md)

---

## Core Concepts

1. **Configure** – describe model, dataset, edit, and guard policies in YAML.
2. **Execute** – run `invarlock run` or `evaluate` under a CI or release profile;
   model-loading commands use the runtime container by default.
3. **Validate** – generate reports via `invarlock report` and run `invarlock verify`;
   attested outputs include `runtime.manifest.json` next to
   `evaluation.report.json`.
4. **Iterate** – compare runs, adjust edit plans, and reissue reports until gates pass.

The guard suite (invariants, spectral, variance, and RMT) keeps edits inside
configured acceptance envelopes even when aggressive compression is attempted.

---

## Live Example Verification

- Runnable documentation surfaces can be verified locally with
  `python scripts/verify_live_examples.py` or `make docs-live`.
- This live check executes concrete Markdown CLI snippets through the current
  checkout and smoke-runs notebooks under `notebooks/`.
- Artifacts land under `tmp/live_examples/`, including per-command JSONL
  results, notebook stdout/stderr logs, and a machine-readable `summary.json`.
- Placeholder/template snippets still need to stay parseable, but only concrete
  runnable examples should be treated as copy-paste-ready.
- This verifier is currently local-only and is not enforced in GitHub Actions.

---

## Building Docs Offline vs Online

- Offline (default): mkdocs builds without contacting the Internet. Mermaid
  diagrams are disabled by default to keep builds fully local.
  - Command: `mkdocs build` or run `make docs` without `--strict`.
- Online (enable networked assets explicitly): enable Mermaid diagrams (via CDN)
  and keep strict checks.
  - Command: `INVARLOCK_DOCS_MERMAID=1 mkdocs build --strict`

Notes

- The configuration references CDNs (MathJax/Polyfill) via `extra_javascript` in
  the generated HTML. These are not fetched at build time; they load when you
  view the HTML in a browser with network access.
- The mermaid2 plugin pings the CDN; we gate it behind the
  `INVARLOCK_DOCS_MERMAID` environment variable to avoid network dependencies by
  default.

---

## Support Matrix

| Surface | Preset shipped | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| GPT-2 causal LM | Yes | Yes | Yes | Yes |
| BERT / RoBERTa MLM | Yes | Yes | Yes | Yes |
| Mistral 7B causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| Qwen2 7B causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| Qwen3 causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| QwQ-32B reasoning causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| DeepSeek-R1-Distill-Qwen causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| Phi-4 causal LM (text-only eval) | Yes | Yes | Yes | No, repo-shipped pilot config only |
| TinyLlama 1.1B causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| OLMo 2 causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| Qwen3.5 causal LM | Yes | Yes | Yes | No, repo-shipped pilot config only |
| Seq2Seq / local pairs | Yes | Yes | No | No |

Published assurance basis currently covers GPT-2 and BERT profiles. Repo-shipped
presets and pilot calibration configs for additional experimental families,
including Mistral 7B, Qwen2 7B, Qwen3, QwQ-32B, DeepSeek-R1-Distill-Qwen,
Phi-4 text-only, TinyLlama 1.1B, OLMo 2, and Qwen3.5, do not become part of the published
assurance basis until supporting artifacts are attached. Access-gated vendor
checkpoints are intentionally excluded from the shipped support matrix and
preset inventory, and ungated families that have not yet closed into clean
pilot lanes remain in the model family backlog rather than the support matrix.

Machine-readable support metadata lives in `contracts/support_matrix.json`. It is
the canonical source of truth for normalized support tiers
(`published_basis`, `supported_experimental`, `community_experimental`) and for
published-basis evidence references.

Maintained shipped-model evidence automation lives in
`scripts/model_evidence_sweep.py`, with tmux-based remote launch support in
`scripts/run_model_evidence_remote.py` and a nightly/manual runner workflow in
`.github/workflows/model-evidence-sweep.yml`.

For the broader inventory of declared support, implemented-but-not-public
coverage, usage-only checkpoint families, and recommended additions, see
[Model Family Catalog](reference/model-family-catalog.md).

---

## Common Workflows

### Research

```bash
pip install "invarlock[adapters,guards,eval]"
invarlock doctor
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
```

### Development

```bash
invarlock run -c configs/presets/causal_lm/wikitext2_512.yaml --profile ci --tier balanced
invarlock plugins adapters
python scripts/verify_ci_matrix.sh
```

### Production Evaluation

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline /path/to/baseline \
  --subject  /path/to/edited \
  --adapter auto \
  --profile release \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
invarlock verify reports/eval/evaluation.report.json
# expects reports/eval/runtime.manifest.json next to the report
```

---

## Configuration Snapshot

```yaml
model:
  id: gpt2
  adapter: hf_causal
  device: auto
dataset:
  provider: wikitext2
  seq_len: 768
  stride: 768
  preview_n: 240
  final_n: 240
  seed: 42
edit:
  # No edit by default (Compare & evaluate/BYOE recommended), or use built-in quant demo:
  # edit:
  #   name: quant_rtn
  #   plan:
  #     bitwidth: 8
  #     per_channel: true
guards:
  spectral:
    kappa: 3.2
  variance:
    tier: balanced
eval:
  pairing:
    enforce: true
output:
  dir: runs/
```

---

<!-- Quick CPU Demos section removed in minimal public footprint -->

```bash
NET=1 INCLUDE_MEASURED_CLS=1 RUN=0 bash scripts/run_tiny_all_matrix.sh
```

Run with `RUN=1` to execute the matrix.

---

**Quick Links**
[Getting Started](user-guide/getting-started.md) ·
[CLI Reference](reference/cli.md) ·
[Primary Metric Smoke](user-guide/primary-metric-smoke.md) ·
[Example Reports](user-guide/example-reports.md) ·
[Contributing](https://github.com/invarlock/invarlock/blob/main/CONTRIBUTING.md)
