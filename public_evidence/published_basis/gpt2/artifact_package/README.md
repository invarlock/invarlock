# GPT-2 Published-Basis Artifact Package

This package is the repo-safe public artifact index for the GPT-2 published
basis. It binds the baseline checkpoint reference, subject checkpoint reference,
strict-pass report, runtime manifest, signed evidence pack, and verifier
commands in one place.

The package is weight-free by design. Checkpoint materialization is external:
readers can verify the shipped evidence without downloading weights, and
maintainers can rebuild the checkpoints from
`../evidence_pack_recipe.json` when producing a fresh evidence drop.

## Contents

| File | Role |
| --- | --- |
| `artifact_package.json` | Machine-readable package manifest and verifier commands. |
| `checkpoint_refs.json` | Baseline and subject checkpoint references for the BYOE lane. |
| `../evaluation.report.json` | Canonical strict verification report. |
| `../runtime.manifest.json` | Container runtime provenance bound to the report. |
| `../evidence_pack/` | Signed, checksum-bound evidence pack. |

## Verify

```bash
uv run invarlock verify \
  public_evidence/published_basis/gpt2/evaluation.report.json \
  --profile release \
  --assurance strict

uv run invarlock advanced evidence-pack verify \
  public_evidence/published_basis/gpt2/evidence_pack \
  --strict \
  --profile release \
  --report-assurance strict \
  --expected-fingerprint sha256:0668414f854d1e75cc7514302a2c974aae1141e2acba225afd55acf3e35eacfb
```

Expected result: both commands pass. The evidence-pack verifier reports
`authenticity=pinned` when JSON output is requested.
