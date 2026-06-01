# Expected Artifacts

Runnable integration examples should generate a small, reviewable artifact set
under an ignored local output directory such as `reports/<target>/`.

| Artifact | Required | Role |
| --- | --- | --- |
| `evaluation.report.json` | Yes | Canonical verifier input for the baseline-vs-subject comparison. |
| `verify.json` | Yes | Machine-readable verifier output from `invarlock verify --json`. |
| `evaluation.html` | Yes | Human-readable report rendered from the evaluation report. |
| `runtime.manifest.json` | Strict mode | Runtime provenance emitted by the container-backed evaluation path. |
| `checkpoint_refs.json` | Target dependent | Baseline/subject provenance for externally materialized checkpoints. |
| `run_command.txt` | Recommended | Wrapper invocation and concrete evaluate, verify, and render commands. |
| `evidence_pack/` | Optional | Signed, checksum-bound bundle for distributable review workflows. |

Generated artifacts should stay out of source control unless a future public
evidence update intentionally promotes a small fixture into `public_evidence/`.

## Verification Commands

```bash
invarlock verify --json reports/<target>/evaluation.report.json \
  > reports/<target>/verify.json

invarlock report html \
  -i reports/<target>/evaluation.report.json \
  -o reports/<target>/evaluation.html
```

For release-grade strict review, use:

```bash
invarlock verify --profile release --assurance strict \
  reports/<target>/evaluation.report.json
```
