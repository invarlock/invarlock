# Public End-To-End Evidence Handoff

Status: `reference-pattern`

Script: runnable against checked-in public evidence.

This example turns an existing external-edit evidence run into the artifacts a
release or registry workflow usually needs:

- machine-readable `invarlock verify --json` output
- rendered `evaluation.html`
- MLflow tag export JSON
- Hugging Face model-card evidence Markdown
- release-review Markdown
- CI summary Markdown

The source evidence is
`public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/`. That run uses
`sshleifer/tiny-gpt2` as the baseline and a locally materialized subject
checkpoint produced by `external_edit_recipe.py`. The subject checkpoint is not
vendored in the repository; `checkpoint_refs.json` and
`external_edit_summary.json` record the external edit type and subject file
hashes.

## Run

From the repository root:

```bash
examples/integrations/public_e2e/run_public_e2e_release_review.sh --force
```

By default, generated files are written under:

```text
examples/integrations/public_e2e/reports/tiny-gpt2-external-magnitude-prune/
```

That path is ignored by git. To write elsewhere:

```bash
examples/integrations/public_e2e/run_public_e2e_release_review.sh \
  --output-dir /tmp/invarlock-public-e2e
```

## Output

| Artifact | Role |
| --- | --- |
| `evaluation.report.json` | Local copy of the canonical verifier input. |
| `runtime.manifest.json` | Runtime provenance sidecar copied beside the report for strict verification. |
| `checkpoint_refs.json` | Baseline, subject, and evidence-pack references. |
| `external_edit_summary.json` | External edit metadata and subject file hashes. |
| `invarlock-verify.json` | Machine-readable verifier result for this local report copy. |
| `evaluation.html` | Human-readable report. |
| `mlflow-tags.json` | Dependency-free registry tag payload. |
| `model-card-invarlock.md` | Copy-pasteable model-card evidence block. |
| `release-review.md` | Release-review packet for maintainers and auditors. |
| `ci-summary.md` | Markdown summary matching the GitHub Actions summary shape. |
| `run_command.txt` | Wrapper invocation for this handoff run. |
| `source_run_command.txt` | Original evaluation command copied from the source evidence, when present. |
| `run_summary.txt` | Paths and success status for this local run. |

## CI Usage

For GitHub Actions, use the repo-local composite action when this source tree is
checked out:

```yaml
- uses: ./.github/actions/invarlock-report-gate
  with:
    report: public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json
    profile: release
    assurance: strict
    runtime-provenance: container
```

The `uses: ./.github/actions/invarlock-report-gate` line is repo-local. Outside
this source tree, copy/vendor `.github/actions/invarlock-report-gate/` into the
target repository until a tagged remote action reference is published.

## Scope

This example is an evidence handoff. It does not regenerate the subject
checkpoint, push to MLflow, update Hugging Face Hub, or approve a deployment.
The canonical evidence remains the checked-in report, runtime manifest,
checkpoint references, external edit summary, and signed evidence pack under
`public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/`.
