# CI and Registry Evidence

Status: `reference-pattern`

This example shows how to attach existing InvarLock evidence to CI, MLflow, and
Hugging Face Hub surfaces without adding a new model registry.

## GitHub Actions

Use the first-party composite action to verify an existing report, render HTML,
generate a review packet, and upload the evidence bundle. This minimal example
uses an existing public evidence report checked into the repository:

```yaml
jobs:
  invarlock:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install -e .
      - uses: ./.github/actions/invarlock-report-gate
        with:
          report: public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json
          profile: release
          assurance: strict
          runtime-provenance: container
```

The `uses: ./.github/actions/invarlock-report-gate` line is a repo-local action
reference. For repositories outside this source tree, copy/vendor
`.github/actions/invarlock-report-gate/` into the target repository and keep the
local reference, or switch to a tagged remote action reference after one is
published. A versioned remote action reference is not part of the public
contract yet.

The action writes:

| Artifact | Purpose |
| --- | --- |
| `evaluation.html` | Human-readable report. |
| `release-review.md` | Release packet for release checks. |
| `mlflow-tags.json` | Registry tag export. |
| `invarlock-verify.json` | Machine-readable verify result. |

## MLflow Tags

Generate a dependency-free MLflow tag export:

```bash
invarlock verify --json \
  public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json \
  --profile release \
  --assurance strict \
  > reports/eval/invarlock-verify.json

invarlock report export \
  --evaluation-report public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json \
  --format mlflow-tags \
  --policy-profile release \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/mlflow-tags.json
```

Apply it from an MLflow-enabled environment:

```python
import json
import mlflow

payload = json.load(open("reports/eval/mlflow-tags.json", encoding="utf-8"))
for key, value in payload["tags"].items():
    mlflow.set_tag(key, value)
mlflow.log_artifact(
    payload["artifact"]["path"],
    artifact_path=payload["artifact"]["artifact_path"],
)
```

The export sets tags for status, report SHA-256, policy profile, baseline, and
subject.

## Hugging Face Model Card Block

Generate a copy-pasteable evidence block:

```bash
invarlock report export \
  --evaluation-report public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json \
  --format model-card-md \
  --report-url https://example.test/evaluation.report.json \
  --evidence-url https://example.test/evidence.zip \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/model-card-invarlock.md
```

Paste the block into the model card near other evaluation evidence. The block
summarizes regression evidence only; it is not deployment approval.

## Release Review Packet

Generate a release-review packet for release maintainers and auditors:

```bash
invarlock report export \
  --evaluation-report public_evidence/real_runs/tiny_gpt2_external_magnitude_prune/evaluation.report.json \
  --format release-review-md \
  --policy-profile release \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/release-review.md
```

The packet includes the baseline and subject identities, pass/fail status,
report hash, policy profile, gate checklist, and evidence checklist.
