# Expected Artifacts

Runnable integration examples should generate a small, reviewable artifact set
under a local output directory outside version-controlled source files, such as
`reports/<target>/<artifact-lane>/` when multiple lanes are being compared.

| Artifact | Required | Role |
| --- | --- | --- |
| `evaluation.report.json` | Yes | Canonical verifier input for the baseline-vs-subject comparison. |
| `verify.json` | Yes | Machine-readable verifier output from `invarlock verify --json`. |
| `evaluation.html` | Yes | Human-readable report rendered from the evaluation report. |
| `runtime.manifest.json` | Strict mode | Runtime provenance emitted by the container-backed evaluation path. |
| `backend_inventory.json` | Quantized adapters | Backend, adapter, smoke, and quantized-module inventory emitted by InvarLock report persistence when adapter provenance is available. |
| `checkpoint_refs.json` | Target dependent | Baseline/subject provenance for materialized checkpoints or runtime adapter subjects. |
| `lane_artifact.json` | Recommended | Canonical artifact-lane label and effective execution, assurance, runtime provenance, and device settings. |
| `run_command.txt` | Recommended | Wrapper invocation and concrete evaluate, verify, and render commands. |
| `run_summary.txt` | Recommended | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `evidence_pack/` | Optional | Signed, checksum-bound bundle for distributable review workflows. |

Generated artifacts should stay out of source control unless a future public
evidence update intentionally promotes a small fixture into `public_evidence/`.

## Verification Commands

```bash
invarlock verify --json reports/<target>/<artifact-lane>/evaluation.report.json \
  > reports/<target>/<artifact-lane>/verify.json

invarlock report html \
  -i reports/<target>/<artifact-lane>/evaluation.report.json \
  -o reports/<target>/<artifact-lane>/evaluation.html
```

For release-profile strict verification, use:

```bash
invarlock verify --profile release --assurance strict \
  --runtime-provenance container \
  reports/<target>/cuda-container-strict/evaluation.report.json
```
