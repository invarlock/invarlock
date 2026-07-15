# CI and Registry Evidence

Status: `reference-pattern`

This example shows how to attach current InvarLock evidence to CI, MLflow, and
Hugging Face Hub surfaces without adding a new model registry. It requires a
current report and its independently supplied strict inputs.

## GitHub Actions

Use the first-party composite action to verify a current report, render HTML,
generate a review packet, and upload the evidence bundle:

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
          report: path/to/current/evaluation.report.json
          baseline: path/to/raw-baseline-report.json
          policy-pack: path/to/acceptance-policy-pack.json
          profile: release
          assurance: strict
          runtime-provenance: container
          expected-runtime-image-digest: sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST
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

Generate a dependency-free MLflow tag export from a successful current strict
verification:

```bash
invarlock verify --json \
  path/to/current/evaluation.report.json \
  --baseline path/to/raw-baseline-report.json \
  --policy-pack path/to/acceptance-policy-pack.json \
  --profile release \
  --assurance strict \
  --runtime-provenance container \
  --expected-runtime-image-digest sha256:REPLACE_WITH_REVIEWED_64_HEX_DIGEST \
  > reports/eval/invarlock-verify.json

invarlock report export \
  --evaluation-report path/to/current/evaluation.report.json \
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
subject. It is a derived handoff record, not an independent release acceptance;
retain the successful verifier result and its strict inputs with the export.

## Hugging Face Model Card Block

Generate a copy-pasteable evidence block:

```bash
invarlock report export \
  --evaluation-report path/to/current/evaluation.report.json \
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
  --evaluation-report path/to/current/evaluation.report.json \
  --format release-review-md \
  --policy-profile release \
  --verify-result reports/eval/invarlock-verify.json \
  --output reports/eval/release-review.md
```

The packet includes the baseline and subject identities, pass/fail status,
report hash, policy profile, gate checklist, and evidence checklist.
