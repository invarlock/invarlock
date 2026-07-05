# Integration Examples

## Purpose

The source tree now has a first-party scaffold for optional integration
examples under `examples/integrations/`. These examples are for checkpoint-edit
toolchains that want to attach InvarLock regression evidence after producing an
edited subject checkpoint.

The integration surface lives in the source tree and keeps optional target
dependencies out of the runtime package and core install.

## Current Scope

| Surface | Status |
| --- | --- |
| Shared evidence wording | Present under `examples/integrations/_shared/evidence-scope.md`. |
| Expected artifact checklist | Present under `examples/integrations/_shared/expected-artifacts.md`. |
| Shared compare wrapper | Present under `examples/integrations/_shared/run_invarlock_compare.sh`. |
| Public end-to-end handoff | Present under `examples/integrations/public_e2e/`. |
| CI and registry evidence | Present under `examples/integrations/ci_registry/`. |
| Target-specific examples | Added one target at a time after backend compatibility is validated. |

Browse the integration scaffold in the repository:
<https://github.com/invarlock/invarlock/tree/staging/next/examples/integrations>

## Target Example Readiness

Target examples should declare one of these labels in their README:

| Label | Meaning |
| --- | --- |
| `runnable` | Commands are expected to generate `evaluation.report.json`, `verify.json`, and `evaluation.html` in the documented environment. |
| `reference-pattern` | Commands show how to attach existing reports to CI or registry surfaces without producing a new subject checkpoint. |
| `exploratory-host` | Commands run with `--execution-mode host --assurance off` for local dependency setup and backend investigation. |
| `compatibility-investigation` | The external artifact cannot yet be loaded or verified through the documented InvarLock path; the README records the blocker. |

## Preflight

Before adding or publishing a target example:

```bash
invarlock doctor
invarlock advanced plugins list --json
```

For optional backend targets, also check the Python module:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('gptqmodel') is not None)"
```

Replace `gptqmodel` with the backend module for the target example.

## Shared Workflow

For a baseline and subject that are already loadable through an InvarLock
adapter:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-smoke \
  --allow-network
```

The generated local output should include:

| Artifact | Role |
| --- | --- |
| `evaluation.report.json` | Canonical verifier input. |
| `verify.json` | Machine-readable verifier result. |
| `evaluation.html` | Human-readable report. |
| `runtime.manifest.json` | Runtime provenance for strict container-backed runs. |
| `run_command.txt` | Wrapper invocation plus evaluate, verify, and render commands. |

Generated reports, models, runs, HTML, and artifacts under
`examples/integrations/**` are ignored by git.

## CI and Registry Handoff

For a source-only public walkthrough, run the script under
`examples/integrations/public_e2e/`. It copies the checked-in external-edit
public report into a local output directory, verifies that local copy, renders
HTML, exports MLflow tags, writes a model-card block, writes a release-review
packet, and writes a CI summary Markdown file.

Use `invarlock report export` when the report already exists and you need to
attach the evidence to systems users already operate:

```bash
invarlock report export \
  --evaluation-report reports/eval/evaluation.report.json \
  --format mlflow-tags \
  --policy-profile ci \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/mlflow-tags.json

invarlock report export \
  --evaluation-report reports/eval/evaluation.report.json \
  --format model-card-md \
  --report-url https://example.test/evaluation.report.json \
  --evidence-url https://example.test/evidence.zip \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/model-card-invarlock.md

invarlock report export \
  --evaluation-report reports/eval/evaluation.report.json \
  --format release-review-md \
  --policy-profile release \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/release-review.md
```

For GitHub Actions, the repository ships a composite action at
`.github/actions/invarlock-report-gate`. It verifies the evaluation report,
renders HTML, writes MLflow tags, writes a release-review packet, appends a PR
summary, and uploads the evidence as a workflow artifact.

Important action inputs include `report`, `profile`, `assurance`,
`runtime-provenance`, `warning-policy`, `verify-output`, `html-output`,
`mlflow-tags-output`, and `review-output`. The generated exports consume the
same `verify-output` JSON so their pass/fail status reflects the verifier
result item for the same evaluation report when the action has run
verification.

This integration layer is intentionally small. It exports InvarLock evidence
for CI and downstream handoff into model cards, registries, MLflow,
Hugging Face Hub, Databricks, SageMaker, Vertex, or an organization's approval
workflow.

## Public Evidence Anchors

Use the shipped public evidence as the credibility floor while target examples
are being built:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/gpt2/evidence_pack/reports/report-001/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/lora_merge_byoe/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/fine_tune_byoe/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evidence_pack/reports/report-001/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evidence_pack/reports/report-001/evaluation.report.json
```

See the [Public Evidence Walkthrough](public-evidence-walkthrough.md) for the
full artifact taxonomy.
