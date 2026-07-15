# Integration Examples

## Purpose

Optional integration examples live under `examples/integrations/`. They show
checkpoint-edit toolchains how to attach InvarLock regression evidence after
producing an edited subject checkpoint.

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
<https://github.com/invarlock/invarlock/tree/main/examples/integrations>

## Target Example Readiness

Target examples should declare one of these labels in their README:

| Label | Meaning |
| --- | --- |
| `runnable` | Commands are expected to generate `evaluation.report.json`, `verify.json`, and `evaluation.html` in the documented environment. |
| `reference-pattern` | Commands show how to attach an existing current report to CI or registry surfaces without producing a new subject checkpoint. |
| `exploratory-host` | Commands run with `--execution-mode host --assurance off` for local dependency setup and backend investigation; their output is diagnostic, not strict or release evidence. |
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

For a source-only public handoff walkthrough, run the script under
`examples/integrations/public_e2e/`. It requires caller-supplied current
evidence and strict inputs, then copies that report into a local output
directory, verifies the copy, renders HTML, exports MLflow tags, writes a
model-card block, writes a release-review packet, and writes a CI summary
Markdown file.

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
same `verify-output` JSON only after a strict receipt-to-report-byte binding
check. The current verify receipt is unsigned, so the export records
`receipt_bound_untrusted` plus the claimed verifier outcome rather than a
verified pass/fail status. Treat the action's actual `invarlock verify` exit
code and retained report inputs as the verifier result.

This integration layer is intentionally small. It exports InvarLock evidence
for CI and downstream handoff into model cards, registries, MLflow,
Hugging Face Hub, Databricks, SageMaker, Vertex, or an organization's approval
workflow.

## Evidence Catalog

The maintained catalog provides stable lane identities for CI and downstream
handoff. `contracts/support_matrix.json` records the current evidence status;
completed artifacts are listed by `public_evidence/catalog_evidence_index.json`.

See the [Evidence Catalog](public-evidence-walkthrough.md) for the artifact
shape and review workflow.
