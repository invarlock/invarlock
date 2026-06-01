# InvarLock Documentation

InvarLock provides auditable release gates for edited model checkpoints. It
validates baseline-vs-subject comparisons, not a specific edit toolchain. A
small built-in RTN dequantized weight-edit simulation (`quant_rtn`, 8-bit)
exists for advanced smoke and demo workflows; production workflows are
bring-your-own-edited-checkpoint (BYOE). See [Compare & evaluate
(BYOE)](user-guide/compare-and-evaluate.md) and the [Public Evidence
Walkthrough](user-guide/public-evidence-walkthrough.md).

Welcome to the documentation hub for InvarLock (auditable release gates for
edited model checkpoints).
The material below is organized so new users can ramp quickly while practitioners
find detailed reference, design rationales, and assurance notes.
It is aimed at checkpoint editors, CI and assurance owners, and researchers
running paired evaluation on text workflows plus the included image-text path.

---

## Start Here

1. **[Getting Started](user-guide/getting-started.md)** – environment setup and the first `evaluate` → `verify` → `report html` loop.
2. **[Quickstart](user-guide/quickstart.md)** – CLI highlights for common workflows.
3. **[Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md)** – baseline ↔ subject paired evaluation with the guard chain.
4. **[Primary Metric Smoke](user-guide/primary-metric-smoke.md)** – tiny examples for ppl/accuracy kinds.

### Choose Your Path

- **Wheel user / reviewer**: start with [Quickstart](user-guide/quickstart.md) if you already have an `evaluation.report.json` bundle and want to verify, explain, or render it.
- **Evaluator**: start with [Getting Started](user-guide/getting-started.md) if you need to run `invarlock evaluate` and produce a fresh evaluation bundle.
- **Repo maintainer**: use the same user guides first, then reach for repo-only smokes, `configs/`, and local runtime-image flows after the core path is green.

### Quick Example

```bash
pip install "invarlock[hf]"

# Compare & evaluate (BYOE checkpoints)
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline <BASELINE_MODEL> \
  --subject  <SUBJECT_MODEL> \
  --baseline-adapter auto \
  --subject-adapter  auto \
  --profile  ci
```

Tip: enable Hub downloads per command when fetching models/datasets:
`invarlock evaluate --allow-network ...`

Security-default note: `evaluate` uses the runtime container by default. Use
`--execution-mode host` only for host-side workflows that intentionally bypass that
boundary. Advanced runtime-heavy workflows live under `invarlock advanced`.

---

## Documentation Map

### User Guide

- [Getting Started](user-guide/getting-started.md)
- [Quickstart](user-guide/quickstart.md)
- [Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md)
- [Primary Metric Smoke](user-guide/primary-metric-smoke.md)
- [Live Examples](user-guide/live-examples.md)
- [Public Evidence Walkthrough](user-guide/public-evidence-walkthrough.md)
- [Configuration Gallery](user-guide/config-gallery.md)
- [Example Reports](user-guide/example-reports.md)
- [Reading a report](user-guide/reading-report.md)
- [Troubleshooting](user-guide/troubleshooting.md) — Error codes and common fixes
- [Plugins](user-guide/plugins.md) — Extending adapters and guards
- [Bring Your Own Data](user-guide/bring-your-own-data.md) — Custom datasets
- [Evidence Packs](user-guide/evidence-packs.md) — Validation suite bundles
- [Evidence Packs Internals](user-guide/evidence-packs-internals.md) — Suite architecture and preset derivation flow

### Reference

- [Reference Index](reference/index.md)
- [CLI Reference](reference/cli.md)
- [Public Contracts](reference/contracts.md)
- [Tier Policy Tuning CLI (Calibration)](reference/calibration.md) — `invarlock advanced calibrate` for tier policy sweeps
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

Maintainer-only runbooks may exist locally and are intentionally omitted from
this public docs index.

<!-- Design docs intentionally omitted from this public docs index. -->

### Assurance

- [Assurance Case](assurance/00-assurance-case.md)
- [Trust Model](assurance/14-trust-model.md)
- [Strict Assurance Checklist](assurance/15-strict-assurance-checklist.md)
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
- [GPU/MPS-First Guard Measurement Contracts](assurance/13-gpu-mps-first-guards.md)
- [Guard Validation Smoke](assurance/16-guard-validation-smoke.md)
- [Empirical Guard Evidence](assurance/17-empirical-guard-evidence.md)

Note: Every assurance claim is backed by automated tests and cross-referenced in
the docs. See Guard Contracts → Coverage Reference
(assurance/04-guard-contracts.md) for the test index.

Calibration CSVs and proof reports referenced in these notes are produced by
local or CI runs (typically under `runs/null_sweeps/**` and
`reports/calibration/**`) and are not committed to the repository. Attach them
to change proposals or releases when you update calibration.

<!-- Developer docs intentionally omitted from this public docs index. See project root CHANGELOG.md. -->

### Security

- [Threat Model](security/threat-model.md) — Assets and adversaries
- [Security Architecture](security/architecture.md) — Components and defaults
- [Best Practices](security/best-practices.md) — Operational recommendations
- [Release Verification](security/release-verification.md) — Verification of published package artifacts and source tags
- [Runtime Provenance Guide](security/runtime-provenance-guide.md) — Manifest requirements for strict assurance
- [pip-audit Allowlist](security/pip-audit-allowlist.md)

### Governance

- [Contribution Guidelines](https://github.com/invarlock/invarlock/blob/v0.9.0/CONTRIBUTING.md)

---

## Core Concepts

1. **Configure** – describe model, dataset, edit, and guard policies in YAML.
2. **Execute** – run `invarlock evaluate` under a CI or release profile;
   model-loading commands use the runtime container by default unless you pass
   `--execution-mode host`.
3. **Validate** – run `invarlock verify` and render HTML via `invarlock report html`;
   container-backed outputs include `runtime.manifest.json` next to
   `evaluation.report.json`.
   Directory inputs to `invarlock report` are only accepted when they contain
   canonical `report.json` or `evaluation.report.json`.
4. **Iterate** – compare runs, adjust edit plans, and reissue reports until gates pass.

The guard suite (invariants, spectral, variance, and RMT) keeps edits inside
configured acceptance envelopes even when aggressive compression is attempted.

---

## Live Example Verification

- Curated CI-safe live examples are gated by `make docs-live-fast` and cover
  `README.md`, `docs/user-guide/getting-started.md`,
  `docs/user-guide/quickstart.md`,
  `notebooks/invarlock_python_api.ipynb`, and
  `notebooks/invarlock_policy_tiers.ipynb`.
- Runnable documentation surfaces can be verified locally with
  `make docs-live-fast`, `python scripts/docs/verify_live_examples.py`, or
  `make docs-live`.
- The curated fast lane replays concrete Markdown CLI snippets in host
  mode with seeded demo evidence, then smoke-runs the curated notebook subset.
- For heavyweight notebook cells that would otherwise trigger model downloads or
  full evaluations, the curated lane reuses seeded demo reports and keeps the
  later contract-reading and verification steps live.
- `make docs-live` remains the broader local lane that replays runnable
  Markdown examples and smoke-runs notebooks under `notebooks/`, using the same
  host seeded-demo approach for heavyweight model-loading steps.
- Artifacts land under `tmp/live_examples/`, including per-command JSONL
  results, notebook stdout/stderr logs, and a machine-readable `summary.json`.
- Placeholder/template snippets must remain parseable, but only concrete
  runnable examples should be treated as copy-paste-ready.
- GitHub Actions enforce the curated deterministic subset; the full verifier
  remains a local or long-gate lane.

---

## Building Docs Offline vs Online

- Offline (default): mkdocs builds without contacting the Internet. Mermaid
  diagrams are disabled by default to keep builds fully local.
  - Command: `make docs` or `mkdocs build --strict`.
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

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| GPT-2 causal LM | Yes | Yes | Yes | Yes |
| BERT / RoBERTa MLM | Yes | Yes | Yes | Yes |
| Mistral 7B causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Ministral 3 causal LM (text-only eval) | Yes | Yes | Yes | No, repo-included pilot config only |
| Qwen2 7B causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Qwen2.5 7B causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Qwen2.5 14B causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Qwen3 causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| DeepSeek-R1-Distill-Qwen causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Phi-4 causal LM (text-only eval) | Yes | Yes | Yes | No, repo-included pilot config only |
| Gemma 4 E2B causal LM (text-only eval) | Yes | Yes | Yes | No, repo-included pilot config only |
| TinyLlama 1.1B causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| OLMo 2 causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Qwen3.5 causal LM | Yes | Yes | Yes | No, repo-included pilot config only |
| Seq2Seq / local pairs | Yes | Yes | No | No |

Published assurance basis covers GPT-2 and BERT profiles. Repo-included
presets and pilot calibration configs for additional experimental families,
including Mistral 7B, Ministral 3 text-only, Qwen2 7B, Qwen2.5 7B, Qwen2.5 14B, Qwen3,
DeepSeek-R1-Distill-Qwen, Phi-4 text-only, Gemma 4 E2B text-only, TinyLlama
1.1B, OLMo 2, and Qwen3.5, do not become part of the published
assurance basis until supporting artifacts are attached. Access-gated vendor
checkpoints are intentionally excluded from the included support matrix and
preset inventory, and ungated families without clean pilot lanes remain in the
model family backlog rather than the support matrix.

`published_basis` remains the narrow public evidence floor, while
`supported_experimental` means the repo ships the preset, calibration config,
targeted tests, and smoke/evidence path for the lane without claiming a
published-basis fixture set.

Image-text evaluation uses the built-in
`hf_multimodal` adapter and the `vision_text` provider. Public support remains
text-only for the Gemma 4 lane, and audio evaluation is deferred.

Machine-readable support metadata lives in `contracts/support_matrix.json`. It is
the canonical source of truth for normalized support tiers
(`published_basis`, `supported_experimental`, `community_experimental`) and for
published-basis evidence references.

Model evidence automation lives in
`scripts/model_evidence/model_evidence_sweep.py`, with tmux-based remote launch support in
`scripts/model_evidence/run_model_evidence_remote.py` and a nightly/manual runner workflow in
`.github/workflows/model-evidence-sweep.yml`.
Repo-prepared-but-not-yet-promoted lanes are tracked in
`contracts/model_family_catalog.json`.
For the new Gemma 4 text lane, the repo-maintained local smoke is the included
manifest dry-run (`scripts/model_evidence/model_evidence_sweep.py --slug gemma4_e2b --dry-run`).
The image-text path also includes an offline demo preset at
`configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml` plus
`tests/fixtures/vision_text/demo_manifest.jsonl` for provider/config validation;
live multimodal model execution requires an installed HF stack and model
weights.

For the broader inventory of declared support, implemented-but-not-public
coverage, usage-only checkpoint families, and recommended additions, see
[Model Family Catalog](reference/model-family-catalog.md).

---

## Common Workflows

### Research

```bash
pip install "invarlock[adapters,guards,eval]"
invarlock doctor
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject /path/to/edited \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --preset configs/presets/causal_lm/wikitext2_512.yaml
```

### Development

```bash
invarlock advanced plugins adapters
invarlock advanced calibrate --help
make ci-matrix
```

### Production Evaluation

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
  --baseline /path/to/baseline \
  --subject  /path/to/edited \
  --baseline-adapter auto --subject-adapter auto \
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

<!-- Quick CPU demos are intentionally omitted from this public docs index. -->

```bash
bash scripts/smoke/run_tiny_all_matrix.sh
```

Run with `RUN=1 NET=1` to execute the matrix and allow downloads.

---

**Quick Links**
[Getting Started](user-guide/getting-started.md) ·
[CLI Reference](reference/cli.md) ·
[Primary Metric Smoke](user-guide/primary-metric-smoke.md) ·
[Example Reports](user-guide/example-reports.md) ·
[Contributing](https://github.com/invarlock/invarlock/blob/v0.9.0/CONTRIBUTING.md)
