# InvarLock Documentation (0.3.0)

The OSS core is edit‑agnostic (BYOE). A small built‑in quantization demo
(`quant_rtn`, 8‑bit) exists for CI/quickstart. See
[Compare & Certify (BYOE)](user-guide/compare-and-certify.md).

Version 0.3.0

Welcome to the documentation hub for InvarLock (Edit‑agnostic robustness certificates for weight edits).
The material below is organized so new users can ramp quickly while practitioners
find detailed reference, design rationales, and assurance notes.

---

## Start Here

1. **[Getting Started](user-guide/getting-started.md)** – environment setup and the first certification loop.
2. **[Quickstart](user-guide/quickstart.md)** – CLI highlights for common workflows.
3. **[Compare & Certify (BYOE)](user-guide/compare-and-certify.md)** – baseline ↔ subject with guardchain.
4. **[Primary Metric Smoke](user-guide/primary-metric-smoke.md)** – tiny examples for ppl/accuracy kinds.

### Quick Examples

```bash
# Core-only install (no torch/transformers): CLI + config tools
pip install invarlock

# HF/torch stack for adapter-based flows (certify/run)
pip install "invarlock[hf]"

# Compare & Certify (BYOE checkpoints)
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline <BASELINE_MODEL> \
  --subject  <SUBJECT_MODEL> \
  --adapter  auto
```

Tip: enable Hub downloads per command when fetching models/datasets:
`INVARLOCK_ALLOW_NETWORK=1 invarlock certify ...`

---

## Documentation Map

### User Guide

- [Getting Started](user-guide/getting-started.md)
- [Quickstart](user-guide/quickstart.md)
- [Compare & Certify (BYOE)](user-guide/compare-and-certify.md)
- [Primary Metric Smoke](user-guide/primary-metric-smoke.md)
- [Configuration Gallery](user-guide/config-gallery.md)
- [Example Reports](user-guide/example-reports.md)
- [Reading a Certificate](user-guide/reading-certificate.md)

### Reference

- [CLI Reference](reference/cli.md)
- [Configuration Schema](reference/config-schema.md)
- [Guards](reference/guards.md)
- [Model Adapters](reference/model-adapters.md)
- [Exporting Certificates (HTML)](reference/exporting-certificates-html.md)
- [Certificate Schema (v1)](reference/certificate-schema.md)
- [Certificate Telemetry](reference/certificate_telemetry.md)
- [Datasets](reference/datasets.md)
- [Artifact Layout](reference/artifacts.md)

<!-- Runbooks removed in minimal OSS footprint -->

<!-- Design docs removed in minimal OSS footprint -->

### Assurance

- [Safety Case](assurance/00-safety-case.md)
- [Evaluation Math](assurance/01-eval-math-proof.md)
- [Coverage & Pairing Plan](assurance/02-coverage-and-pairing.md)
- [BCa Bootstrap (Paired Δlog)](assurance/03-bca-bootstrap.md)
- [Guard Contracts & Primer](assurance/04-guard-contracts.md)
- [Spectral False-Positive Control](assurance/05-spectral-fpr-derivation.md)
- [RMT ε-Rule](assurance/06-rmt-epsilon-rule.md)
- [VE Predictive Gate](assurance/07-ve-gate-power.md)
- [Determinism Contracts](assurance/08-determinism-contracts.md)
 - [Tier v1.0 Calibration](assurance/09-tier-v1-calibration.md)

Note: Every safety claim is backed by automated tests and cross-referenced in
the docs. See Guard Contracts → Coverage Reference
(assurance/04-guard-contracts.md) for the test index.

Calibration CSVs and proof certs referenced in these notes are produced by
local or CI runs (typically under `runs/null_sweeps/**` and
`reports/calibration/**`) and are not committed to the repository. Attach them
to change proposals or releases when you update calibration.

<!-- Developer docs removed in minimal OSS footprint. See project root CHANGELOG.md. -->

### Security

- [pip-audit Allowlist](security/pip-audit-allowlist.md)

### Governance

- [Contribution Guidelines](https://github.com/invarlock/invarlock/blob/main/CONTRIBUTING.md)

---

## Core Concepts

1. **Configure** – describe model, dataset, edit, and guard policies in YAML.
2. **Execute** – run `invarlock run` under a CI or release profile with pairing
   enforced.
3. **Validate** – generate certificates via `invarlock report` and run `invarlock verify` for policy compliance.
4. **Iterate** – compare runs, adjust edit plans, and reissue certificates until gates pass.

The guard suite (invariants, spectral, variance, and RMT) ensures edits stay
inside safety envelopes even when aggressive compression is attempted.

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

| Component | Support                                                               |
| --------- | --------------------------------------------------------------------- |
| Python    | 3.12+                                                                 |
| Devices   | CUDA, MPS (Apple Silicon), CPU                                        |
| Models    | GPT‑2 Small/Medium adapters                                           |
| Edits     | RTN quantization (demo built-in); others via Compare & Certify (BYOE) |
| Datasets  | WikiText‑2 (paired 200/200 windows), synthetic samples                |

---

## Common Workflows

### Research

```bash
pip install "invarlock[adapters,guards,eval]"
invarlock doctor
INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline gpt2 \
  --subject /path/to/edited \
  --adapter auto \
  --profile ci \
  --preset configs/tasks/causal_lm/ci_cpu.yaml
```

### Development

```bash
invarlock run -c configs/edits/quant_rtn/8bit_attn.yaml --profile ci --tier balanced
invarlock plugins adapters
python scripts/verify_ci_matrix.sh
```

### Production Certification

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
  --baseline /path/to/baseline \
  --subject  /path/to/edited \
  --adapter auto \
  --profile release \
  --preset configs/tasks/causal_lm/release_cpu.yaml
invarlock verify reports/cert/evaluation.cert.json
```

---

## Configuration Snapshot

```yaml
model:
  id: gpt2
  adapter: hf_gpt2
  device: auto
dataset:
  provider: wikitext2
  seq_len: 768
  stride: 768
  preview_n: 200
  final_n: 200
  seed: 42
edit:
  # No edit by default (Compare & Certify/BYOE recommended), or use built-in quant demo:
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

<!-- Quick CPU Demos section removed in minimal OSS footprint -->

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
