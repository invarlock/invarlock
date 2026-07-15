# Public End-To-End Evidence Handoff

Status: `reference-pattern`

Script: requires caller-supplied current evidence.

This example turns an existing external-edit evidence run into the artifacts a
release or registry workflow usually needs:

- machine-readable `invarlock verify --json` output
- rendered `evaluation.html`
- MLflow tag export JSON
- Hugging Face model-card evidence Markdown
- release-review Markdown
- CI summary Markdown

The script accepts a current evaluation report, its complete retained raw
baseline, an independently maintained acceptance policy pack, and an
independently obtained runtime-image digest.

## Run

From the repository root:

```bash
examples/integrations/public_e2e/run_public_e2e_release_review.sh \
  --report /path/to/evaluation.report.json \
  --baseline /path/to/raw-baseline-report.json \
  --policy-pack /path/to/acceptance-policy-pack.json \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  --force
```

The default mode is strict verification with the `ci` profile. All four trust
inputs are required. Do not reconstruct the baseline from the subject, generate
the policy pack as part of the submitted bundle, or copy the trusted digest from
the submitted runtime manifest. A caller may explicitly select
`--assurance off --profile dev` for non-assurance inspection, but that result is
not a release acceptance.

By default, generated files are written under:

```text
examples/integrations/public_e2e/reports/release-review/
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
| `runtime.manifest.json` | Runtime provenance sidecar copied beside the report for verification. |
| `baseline.report.json` | Strict runs only: independently supplied raw baseline copied for replay. |
| `acceptance-policy-pack.json` | Strict runs only: independently supplied policy copied for replay. |
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
    report: path/to/evaluation.report.json
    baseline: path/to/raw-baseline-report.json
    policy-pack: path/to/acceptance-policy-pack.json
    profile: ci
    assurance: strict
    runtime-provenance: container
    expected-runtime-image-digest: ${{ env.TRUSTED_RUNTIME_IMAGE_DIGEST }}
```

The `baseline`, `policy-pack`, and expected image digest must come from sources
independent of the submitted report.

The `uses: ./.github/actions/invarlock-report-gate` line is repo-local. Outside
this source tree, copy/vendor `.github/actions/invarlock-report-gate/` into the
target repository until a tagged remote action reference is published.

## Scope

This example is an evidence handoff. It does not regenerate the subject
checkpoint, push to MLflow, update Hugging Face Hub, or approve a deployment.
The canonical evidence is the caller-supplied report, sibling runtime manifest,
retained raw baseline, acceptance policy pack, and independently pinned runtime
image identity. Optional checkpoint references and external-edit summaries are
copied when present.
