# InvarLock Documentation

InvarLock is a standalone verification layer for baseline-versus-subject
checkpoint comparisons. Production subjects come from your quantization, pruning, adapter,
fine-tuning, or other external edit workflow; the main path is
bring-your-own-edited-checkpoint (BYOE).

In these docs, a **strict pass** means the verifier received the complete raw
baseline report, an independently maintained acceptance policy pack, and an
independently pinned runtime-image digest, then accepted the report schema,
pairing, recomputed
metric, required guard evidence, and runtime-manifest binding. It is not
execution attestation, artifact provenance, or a general model-safety
certification. See the [Trust Model](assurance/14-trust-model.md).

A generated strict report is not yet a strict pass: its report-local gates may
show `PASS`, but `assurance.verdict` remains `pending_verifier` until
`invarlock verify` exits `0` with those independent inputs. A policy pack or
digest copied from the submitted bundle is not an independent trust anchor. See
[policy-pack build and
verification](reference/contracts.md#policy-packs) and the
[runtime provenance guide](security/runtime-provenance-guide.md).

---

## Start Here

1. **[Getting Started](user-guide/getting-started.md)** – the first `evaluate → verify → report html` loop.
2. **[Compare & Evaluate (BYOE)](user-guide/compare-and-evaluate.md)** – use a checkpoint produced by an external edit workflow.
3. **[Reading a Report](user-guide/reading-report.md)** – interpret PASS/FAIL, evidence maturity, warnings, and provenance.
4. **[Alternatives Comparison](reference/alternatives-comparison.md)** – decide when NeMo Evaluator, MLflow, lm-evaluation-harness, or another tool is the better fit.

### Choose Your Path

- **Report reader**: start with [Reading a Report](user-guide/reading-report.md).
- **Checkpoint evaluator**: start with [Getting Started](user-guide/getting-started.md).
- **CI owner**: continue from [Quickstart](user-guide/quickstart.md) to the [CLI Reference](reference/cli.md).
- **Toolchain designer**: use [Alternatives Comparison](reference/alternatives-comparison.md) before choosing workflow components.

### Quick Example

Strict evaluation requires a running Docker or Podman engine. Confirm the
engine separately (`docker info` or `podman info`) and run `invarlock doctor`
for Python, dependency, and accelerator diagnostics.

```bash
pip install "invarlock[hf]"
invarlock doctor

BASELINE_CHECKPOINT=/path/to/original-checkpoint
EDITED_SUBJECT_CHECKPOINT=/path/to/checkpoint-produced-by-your-edit-pipeline

# The subject must be the actual output of an external edit pipeline.
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline "$BASELINE_CHECKPOINT" \
  --subject "$EDITED_SUBJECT_CHECKPOINT" \
  --baseline-adapter auto \
  --subject-adapter  auto \
  --profile ci \
  --assurance strict \
  --verbose \
  --report-out reports/eval
```

Continue with the strict verifier command in [Getting
Started](user-guide/getting-started.md#verify-and-render). `evaluate` uses the
runtime container by default; network access is explicit with `--allow-network`.
`--verbose` prints the retained `Baseline report: ...` path, also recorded at
`provenance.baseline.report_path`. The template above is not a claim that a
strict black-box run has passed.

---

## Documentation Map

### User Guide

- [Getting Started](user-guide/getting-started.md)
- [Quickstart](user-guide/quickstart.md)
- [Compare & evaluate (BYOE)](user-guide/compare-and-evaluate.md)
- [Reading a report](user-guide/reading-report.md) — PASS meaning, evidence maturity, warnings, and provenance
- [Failure Examples](user-guide/failure-examples.md)
- [Evidence Packs](user-guide/evidence-packs.md) — Portable validation bundles
- [Troubleshooting](user-guide/troubleshooting.md) — Error codes and common fixes
- [Knowledge & self-edit workflows](user-guide/knowledge-and-self-edit-workflows.md)
- [Primary Metric Smoke](user-guide/primary-metric-smoke.md)
- [Live Examples](user-guide/live-examples.md)
- [Integration Examples](user-guide/integrations.md)
- [Public Evidence Walkthrough](user-guide/public-evidence-walkthrough.md)
- [Configuration Gallery](user-guide/config-gallery.md)
- [Example Reports](user-guide/example-reports.md)
- [Plugins](user-guide/plugins.md) — Extending adapters and guards
- [Bring Your Own Data](user-guide/bring-your-own-data.md) — Custom datasets

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
- [Alternatives and Workflow Fit](reference/alternatives-comparison.md)

### Assurance

- [Assurance Case](assurance/00-assurance-case.md)
- [Trust Model](assurance/14-trust-model.md)
- [Strict Assurance Checklist](assurance/15-strict-assurance-checklist.md)
- [Evaluation Math Derivation](assurance/01-eval-math-derivation.md)
- [Coverage & Pairing Plan](assurance/02-coverage-and-pairing.md)
- [BCa Bootstrap (Paired Δlog)](assurance/03-bca-bootstrap.md)
- [Guard Contracts & Primer](assurance/04-guard-contracts.md)
- [Spectral Selection Arithmetic and Assumptions](assurance/05-spectral-fpr-derivation.md)
- [RMT ε-Rule](assurance/06-rmt-epsilon-rule.md)
- [VE Predictive Gate](assurance/07-ve-gate-power.md)
- [Determinism Contracts](assurance/08-determinism-contracts.md)
- [Tier Policy v1 Calibration](assurance/09-tier-v1-calibration.md)
- [Guard Metric Impact Method](assurance/10-guard-metric-impact-method.md)
- [Policy Provenance & Digest](assurance/11-policy-provenance.md)
- [Device Drift Bands](assurance/12-device-drift-bands.md)
- [GPU/MPS-First Guard Measurement Contracts](assurance/13-gpu-mps-first-guards.md)
- [Guard Validation Smoke](assurance/16-guard-validation-smoke.md)
- [Diagnostic Empirical Guard Artifact Inventory](assurance/17-empirical-guard-evidence.md)

Automated tests cover the implementation contracts cross-referenced from the
assurance notes. Empirical performance and calibration claims are backed by
separately reviewed run artifacts and independently supplied trust anchors.

Calibration CSVs and evidence reports are produced by evaluation runs,
typically under `runs/null_sweeps/**` and `reports/calibration/**`. Publish the
supporting artifacts with any release that changes policy defaults. Current
public artifacts are listed by `public_evidence/published_basis_index.json`.

### Security

- [Threat Model](security/threat-model.md) — Assets and adversaries
- [Security Architecture](security/architecture.md) — Components and defaults
- [Best Practices](security/best-practices.md) — Operational recommendations
- [Release Verification](security/release-verification.md) — Verification of published package artifacts and source tags
- [Runtime Provenance Guide](security/runtime-provenance-guide.md) — Manifest requirements for strict assurance
- [pip-audit Allowlist](security/pip-audit-allowlist.md)

### Governance

- [Contribution Guidelines](https://github.com/invarlock/invarlock/blob/v0.12.1/CONTRIBUTING.md)

---

## Core Concepts

1. **Configure** – describe model, dataset, edit, and guard policies in YAML.
2. **Execute** – run `invarlock evaluate` under a CI or release profile;
   model-loading commands use the runtime container by default unless you pass
   `--execution-mode host`.
3. **Validate** – run `invarlock verify` with the complete raw baseline, a
   independently maintained policy pack, and an independently maintained
   `--expected-runtime-image-digest` for strict assurance, then render HTML via
   `invarlock report html`;
   container-backed outputs include `runtime.manifest.json` next to
   `evaluation.report.json`.
   Directory inputs to `invarlock report` are only accepted when they contain
   canonical `report.json` or `evaluation.report.json`.
4. **Iterate** – compare runs, adjust edit plans, and reissue reports until gates pass.

The guard suite (invariants, spectral, variance, and RMT) evaluates available
evidence against configured acceptance envelopes. A pass is scoped to those
measurements and policies; it is not a general safety or downstream-quality guarantee.

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

InvarLock maintains 39 evaluation lanes across causal, masked-language, seq2seq, and image-text workflows. Each lane has a checked adapter, preset, input definition, execution policy, and required artifact set.

The evidence column reports current artifacts only. A lane changes to **Available** after its current run and verification artifacts are published.

| Surface | Lane ID | Adapter | Evidence |
| --- | --- | --- | --- |
| BERT / RoBERTa MLM | `bert-mlm-hf` | `hf_mlm` | **Evidence not yet created** |
| DeepSeek-R1-0528-Qwen3 8B causal LM | `deepseek-r1-0528-qwen3-8b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| DeepSeek-R1-Distill-Qwen 14B causal LM | `deepseek-r1-distill-qwen-14b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| DeepSeek-R1-Distill-Qwen causal LM | `deepseek-r1-distill-qwen-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Falcon 7B causal LM | `falcon-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| FLAN-T5 base seq2seq LM | `flan-t5-base-seq2seq-hf` | `hf_seq2seq` | **Evidence not yet created** |
| Gemma 4 12B any-to-any LM | `gemma4-12b-any-to-any-hf` | `hf_multimodal` | **Evidence not yet created** |
| Gemma 4 26B-A4B MoE image-text LM | `gemma4-26b-a4b-moe-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| Gemma 4 31B image-text LM | `gemma4-31b-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| Gemma 4 E2B image-text LM | `gemma4-e2b-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| Gemma 4 E2B causal LM (text-only eval) | `gemma4-e2b-text-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Gemma 4 E4B image-text LM | `gemma4-e4b-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| GPT-OSS 20B causal LM | `gpt-oss-20b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| GPT-2 causal LM | `gpt2-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Granite 4.1 3B causal LM | `granite-4-1-3b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Granite 4.1 8B causal LM | `granite-4-1-8b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Ministral 3 14B causal LM (text-only eval) | `ministral-3-14b-text-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Ministral 3 3B causal LM (text-only eval) | `ministral-3-3b-text-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Ministral 3 8B causal LM (text-only eval) | `ministral-3-8b-text-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Mistral 7B causal LM | `mistral-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Mixtral 8x7B MoE causal LM | `mixtral-8x7b-moe-causal-hf` | `hf_causal` | **Evidence not yet created** |
| OLMo 2 13B causal LM | `olmo-2-13b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| OLMo 2 7B causal LM | `olmo-2-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| OLMoE 1B-active/7B-total causal LM | `olmoe-1b-7b-0924-causal-hf` | `hf_causal` | **Evidence not yet created** |
| OpenLLaMA 7B causal LM | `open-llama-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Phi-4 mini causal LM | `phi-4-mini-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Phi-4 causal LM (text-only eval) | `phi-4-text-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen2.5 14B causal LM | `qwen2-5-14b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen2.5 7B causal LM | `qwen2-5-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen2 7B causal LM | `qwen2-7b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen3 30B-A3B MoE causal LM | `qwen3-30b-a3b-moe-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen3.5 27B image-text LM (scoped) | `qwen3-5-27b-image-text-scoped-hf` | `hf_multimodal` | **Evidence not yet created** |
| Qwen3.5 2B image-text LM | `qwen3-5-2b-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| Qwen3.5 4B image-text LM | `qwen3-5-4b-image-text-hf` | `hf_multimodal` | **Evidence not yet created** |
| Qwen3.5 causal LM | `qwen3-5-causal-hf` | `hf_causal` | **Evidence not yet created** |
| Qwen3.6 27B image-text LM (scoped) | `qwen3-6-27b-image-text-scoped-hf` | `hf_multimodal` | **Evidence not yet created** |
| Qwen3 causal LM | `qwen3-causal-hf` | `hf_causal` | **Evidence not yet created** |
| SmolLM3 3B causal LM | `smollm3-3b-causal-hf` | `hf_causal` | **Evidence not yet created** |
| TinyLlama 1.1B causal LM | `tinyllama-1-1b-causal-hf` | `hf_causal` | **Evidence not yet created** |

Machine-readable definitions live in `contracts/evidence_catalog_v1.json` and `contracts/support_matrix.json`. Model and adapter implementation details live in the [Model Family Catalog](reference/model-family-catalog.md).

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
invarlock verify \
  --profile release \
  --assurance strict \
  --baseline /path/to/retained/baseline/report.json \
  --policy-pack /path/to/acceptance/policy-pack.json \
  --expected-runtime-image-digest "$EXPECTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
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
[Contributing](https://github.com/invarlock/invarlock/blob/v0.12.1/CONTRIBUTING.md)
