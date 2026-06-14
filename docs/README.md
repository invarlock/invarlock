# InvarLock Documentation

InvarLock provides auditable strict verification for edited model checkpoints. It
validates baseline-vs-subject comparisons, not a specific edit toolchain. A
small built-in RTN dequantized weight-edit simulation (`quant_rtn`, 8-bit)
exists for advanced smoke and demo workflows; production workflows are
bring-your-own-edited-checkpoint (BYOE). See [Compare & evaluate
(BYOE)](user-guide/compare-and-evaluate.md) and the [Public Evidence
Walkthrough](user-guide/public-evidence-walkthrough.md).

Welcome to the documentation hub for InvarLock (auditable strict verification for
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
- [Integration Examples](user-guide/integrations.md)
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
- [Tier Policy v1 Calibration](assurance/09-tier-v1-calibration.md)
- [Guard Overhead Method](assurance/10-guard-overhead-method.md)
- [Policy Provenance & Digest](assurance/11-policy-provenance.md)
- [Device Drift Bands](assurance/12-device-drift-bands.md)
- [GPU/MPS-First Guard Measurement Contracts](assurance/13-gpu-mps-first-guards.md)
- [Guard Validation Smoke](assurance/16-guard-validation-smoke.md)
- [Empirical Guard Evidence](assurance/17-empirical-guard-evidence.md)

Note: Every assurance claim is backed by automated tests and cross-referenced in
the docs. See Guard Contracts → Coverage Reference
(assurance/04-guard-contracts.md) for the test index.

Calibration CSVs and evidence reports referenced in these notes are produced by
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

- [Contribution Guidelines](https://github.com/invarlock/invarlock/blob/v0.10.0/CONTRIBUTING.md)

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
  diagrams are disabled by default to keep builds fully local. The generated
  HTML references MathJax so formulas render in browsers with network access;
  MathJax is not fetched during the build.
  - Command: `make docs` or `mkdocs build --strict`.
- Online (enable networked assets explicitly): enable Mermaid diagrams (via CDN)
  and keep strict checks.
  - Command: `INVARLOCK_DOCS_MERMAID=1 mkdocs build --strict`

Notes

- The configuration references MathJax via `extra_javascript` in the generated
  HTML. This is required for Arithmatex formulas to render on the published
  docs site.
- The mermaid2 plugin pings the CDN; we gate it behind the
  `INVARLOCK_DOCS_MERMAID` environment variable to avoid network dependencies by
  default.

---

## Support Matrix

### Baseline Fixture Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| GPT-2 causal LM | Yes | Yes | Yes | Yes |
| BERT / RoBERTa MLM | Yes | Yes | Yes | Yes |

### Published Small/Local Decoder Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| TinyLlama 1.1B causal LM | Yes | Yes | Yes | Yes |
| SmolLM3 3B causal LM | Yes | Yes | Yes | Yes |
| Gemma 4 E2B causal LM (text-only eval) | Yes | Yes | Yes | Yes |
| Ministral 3 3B causal LM (text-only eval) | Yes | Yes | Yes | Yes |
| Phi-4 mini causal LM | Yes | Yes | Yes | Yes |
| Granite 4.1 3B causal LM | Yes | Yes | Yes | Yes |

### Published 7B-9B Decoder Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| Mistral 7B causal LM | Yes | Yes | Yes | Yes |
| Ministral 3 8B causal LM (text-only eval) | Yes | Yes | Yes | Yes |
| Qwen2 7B causal LM | Yes | Yes | Yes | Yes |
| Qwen2.5 7B causal LM | Yes | Yes | Yes | Yes |
| Qwen3 causal LM | Yes | Yes | Yes | Yes |
| Qwen3.5 causal LM | Yes | Yes | Yes | Yes |
| DeepSeek-R1-Distill-Qwen causal LM | Yes | Yes | Yes | Yes |
| DeepSeek-R1-0528-Qwen3 8B causal LM | Yes | Yes | Yes | Yes |
| OLMo 2 7B causal LM | Yes | Yes | Yes | Yes |
| OpenLLaMA 7B causal LM | Yes | Yes | Yes | Yes |
| Falcon 7B causal LM | Yes | Yes | Yes | Yes |
| Granite 4.1 8B causal LM | Yes | Yes | Yes | Yes |

### Published 13B-14B And Reasoning Decoder Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| Ministral 3 14B causal LM (text-only eval) | Yes | Yes | Yes | Yes |
| OLMo 2 13B causal LM | Yes | Yes | Yes | Yes |
| Qwen2.5 14B causal LM | Yes | Yes | Yes | Yes |
| DeepSeek-R1-Distill-Qwen 14B causal LM | Yes | Yes | Yes | Yes |
| Phi-4 causal LM (text-only eval) | Yes | Yes | Yes | Yes |

### Multimodal Published Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| Qwen3.5 2B image-text LM | Yes | Yes | Yes | Yes |
| Qwen3.5 4B image-text LM | Yes | Yes | Yes | Yes |
| Gemma 4 E2B image-text LM | Yes | Yes | Yes | Yes |
| Gemma 4 E4B image-text LM | Yes | Yes | Yes | Yes |
| Gemma 4 12B any-to-any LM | Yes | Yes | Yes | Yes |
| Gemma 4 26B-A4B MoE image-text LM | Yes | Yes | Yes | Yes |

### Large/MoE Published Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| OLMoE 1B-active/7B-total causal LM | Yes | Yes | Yes | Yes |
| Mixtral 8x7B MoE causal LM | Yes | Yes | Yes | Yes |
| Qwen3 30B-A3B MoE causal LM | Yes | Yes | Yes | Yes |

### Seq2Seq Published Evidence

| Surface | Preset included | Adapter available | Pilot calibration config present | Published assurance basis |
| ------- | -------------- | ----------------- | -------------------------------- | ------------------------- |
| FLAN-T5 base seq2seq LM | Yes | Yes | Yes | Yes |

Published assurance basis covers GPT-2, BERT, Mistral 7B, Ministral 3 3B,
Ministral 3 8B, Ministral 3 14B, TinyLlama 1.1B, Gemma 4 E2B text-only,
Gemma 4 E2B image-text, Gemma 4 E4B image-text, Granite 4.1 3B,
Granite 4.1 8B, OLMo 2 7B, OLMo 2 13B, Qwen2 7B, OpenLLaMA 7B,
Falcon 7B, Qwen2.5 7B, Qwen2.5 14B, Qwen3 8B, Qwen3.5 9B,
Qwen3.5 2B image-text, DeepSeek-R1-Distill-Qwen 7B,
DeepSeek-R1-0528-Qwen3 8B,
DeepSeek-R1-Distill-Qwen 14B, Phi-4 text-only, Qwen3.5 4B image-text,
Gemma 4 12B image-text, Gemma 4 26B-A4B image-text MoE, OLMoE
1B-active/7B-total MoE, Mixtral 8x7B MoE, Qwen3 30B-A3B MoE, and FLAN-T5 base
seq2seq profiles.
Repo-included presets and pilot calibration configs for prepared practical-pick
lanes do not become part of the published assurance basis until supporting
artifacts are attached. OLMoE is the smaller MoE published-basis validation
lane; Mixtral 8x7B and Qwen3 30B-A3B are larger no-op preservation
bases. Gemma 4 26B-A4B is a multimodal MoE image-text preservation basis; it is
not audio, exhaustive expert-bank, or MoE routing-quality evidence. The Qwen3
30B-A3B fixture requires all-8 80GB-GPU sharding and uses scoped
attention/router/shared-expert guard scans; it is not an exhaustive expert-bank
or MoE routing-quality claim.
The empirical guard manifest includes no-op published-basis summaries for the
modern promoted families. They are null-behavior evidence and calibration
inputs, but they do not re-derive the packaged spectral/RMT/variance tier
constants; transferred attention caps remain budgeted sentinels until a
family-specific null sweep supports an FPR interpretation.
Mistral 7B additionally ships the current real guard-value scenario package:
`public_evidence/published_basis/mistral_7b/guard_value_demo/` records PM-pass,
baseline-relative spectral, RMT, and variance/VE evidence from clean
confirmation reruns.
Practical-pick families without tuned edit params or public evidence fixtures
are tracked as `community_experimental` rows, even when a repo pilot preset and
calibration config are already present. Access-gated vendor checkpoints are
intentionally excluded from the included preset inventory.
The Phi-4 public fixture is text-only and skips guard-overhead measurement by
preset policy; strict release verification accepts that declared skip.
The FLAN-T5 base public fixture uses pinned CNN/DailyMail validation data via
`hf_seq2seq`; strict release verification accepts one advisory guard warning
while the hard policy gates pass.
The Qwen3.5 4B image-text lane now includes a public VQAv2 preset, null-sweep
config, strict public report, runtime manifest, and signed evidence pack after
the structured JSON-answer prompt fix.

`published_basis` remains the narrow public evidence floor, while
`supported_experimental` means the repo ships the preset, calibration config,
targeted tests, smoke/evidence path, and tuned edit-param coverage for the lane
without claiming a published-basis fixture set. `community_experimental` rows
are candidate inventory entries; some already have repo pilot presets and
calibration configs, but still need the remaining promotion artifacts before
they become release-supported lanes.

Image-text evaluation uses the built-in
`hf_multimodal` adapter and the `vision_text` provider. Install
`invarlock[multimodal]` for this path; Gemma 4 unified checkpoints require
`transformers>=5.12.0` and `torchvision>=0.26.0`. Gemma 4 E2B has separate
text-only and image-text public bases; Qwen3.5 2B, Qwen3.5 4B, Gemma 4 E4B,
Gemma 4 12B, and Gemma 4 26B-A4B also have public image-text bases. Audio
evaluation is deferred. Public
image-text basis promotion requires
measured accuracy on a pinned public dataset above the repo floor; preservation
passing alone is not sufficient.

Machine-readable support metadata lives in `contracts/support_matrix.json`. It is
the canonical source of truth for normalized support tiers
(`published_basis`, `supported_experimental`, `community_experimental`) and for
published-basis evidence references. Model lifecycle decisions live in
`contracts/model_classification.json`: that file records whether a lane or
family is published, backlog, blocked, smoke-only, usage-only, or out of scope,
and centralizes blocked named checkpoints for future license/access changes.

Model evidence automation lives in
`scripts/model_evidence/model_evidence_sweep.py`, with tmux-based remote launch support in
`scripts/model_evidence/run_model_evidence_remote.py` and a nightly/manual runner workflow in
`.github/workflows/model-evidence-sweep.yml`.
For large MoE lanes that do not fit comfortably on one GPU, the remote helper
supports grouped CUDA visibility, for example
`--gpu-group 0,1,2,3` to launch one sweep shard with all four GPUs exposed
instead of one shard per GPU.
Repo-prepared-but-not-yet-promoted lanes are tracked in
`contracts/model_family_catalog.json`; promotion eligibility and blockers are
tracked in `contracts/model_classification.json`.
For the Gemma 4 text lane, the repo-maintained local smoke is the included
manifest dry-run (`scripts/model_evidence/model_evidence_sweep.py --suite repo-mentioned-gpu --slug gemma4_e2b_public --dry-run`).
The image-text path also includes an offline demo preset at
`configs/presets/multimodal/gemma4_e2b_vision_text_256.yaml` and a Gemma
4 12B pilot at `configs/presets/multimodal/gemma4_12b_vision_text_256.yaml` plus
`tests/fixtures/vision_text/demo_manifest.jsonl` for provider/config validation.
Gemma 4 E2B, Gemma 4 E4B, and Gemma 4 12B use
`configs/presets/multimodal/gemma4_e2b_public_vqav2_256.yaml`,
`configs/presets/multimodal/gemma4_e4b_public_vqav2_256.yaml`, and
`configs/presets/multimodal/gemma4_12b_public_vqav2_256.yaml` and
matching `configs/calibration/null_sweep_gemma4_*.yaml` files together with the
model-evidence materializer for pinned public VQAv2 sample-validation data. Their
public fixtures live under `public_evidence/published_basis/gemma4_e2b_image_text/`,
`public_evidence/published_basis/gemma4_e4b/`, and
`public_evidence/published_basis/gemma4_12b/`, and the local smoke manifest
remains provider/config validation only. Gemma 4 26B-A4B uses the analogous
`configs/presets/multimodal/gemma4_26b_a4b_public_vqav2_256.yaml` and
`configs/calibration/null_sweep_gemma4_26b_a4b.yaml` path; its public fixture
lives under `public_evidence/published_basis/gemma4_26b_a4b/`. The Qwen3.5 2B
and Qwen3.5 4B public image-text bases use the same pinned VQAv2
materialization pattern through
`configs/presets/multimodal/*_public_vqav2_256.yaml` and matching
`configs/calibration/null_sweep_*.yaml` files.

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
[Contributing](https://github.com/invarlock/invarlock/blob/v0.10.0/CONTRIBUTING.md)
