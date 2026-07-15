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
| `baseline.report.json` | Strict acceptance | Independently retained raw baseline used for verifier replay. |
| `acceptance-policy-pack.json` | Strict acceptance | Independently maintained policy thresholds used for verifier replay. |
| `backend_inventory.json` | Quantized adapters | Backend, adapter, smoke, and quantized-module inventory emitted by InvarLock report persistence when adapter provenance is available. |
| `runtime_quantization_proof.json` | Strict module-backed quantized adapters | Strict JSON observation of a live loaded model's recognized backend-specific runtime types. It is not a packed-storage or checkpoint-artifact proof. |
| `checkpoint_refs.json` | Target dependent | Baseline/subject provenance for materialized checkpoints or runtime adapter subjects. |
| `external_edit_summary.json` | Target dependent | Edit or quantization materialization metadata and file hashes for examples that create a subject checkpoint before evaluation. |
| `training_receipt.json` | Training targets | Immutable training profile, data, baseline, adapter (when applicable), subject, and delta evidence recomputed before evaluation. |
| `training_binding.json` | Training targets | Post-evaluation proof that the subject tree and copied receipt still match the verified training artifact. |
| `training_evidence_proof.json` | Training targets | Receipt-bound artifact replay and reload evidence for the evaluated training subject. |
| `training_profile_snapshot.json` | Training targets | Immutable reviewed training profile and explicit validation scope used for the lane. |
| `adapter_runtime_summary.json` | Target dependent | Runtime adapter metadata, quantization settings, save-boundary notes, and file hashes for adapter-loaded subject paths. |
| `fixture_summary.json` | Target dependent | Local fixture parameters and file hashes copied into the lane output by examples that generate fixture data. |
| `lane_artifact.json` | Recommended | Canonical artifact-lane label and effective execution, assurance, runtime provenance, and device settings. |
| `run_command.txt` | Recommended | Wrapper invocation and concrete evaluate, verify, and render commands. |
| `run_summary.txt` | Recommended | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |
| `evidence_pack/` | Optional | Signed, checksum-bound bundle for distributable review workflows. |

Generated artifacts should stay out of source control unless a future public
evidence update intentionally adds a verified fixture to `public_evidence/`.

## Verification Commands

```bash
invarlock verify --json --profile dev --assurance off \
  reports/<target>/<artifact-lane>/evaluation.report.json \
  > reports/<target>/<artifact-lane>/verify.json

invarlock report html \
  -i reports/<target>/<artifact-lane>/evaluation.report.json \
  -o reports/<target>/<artifact-lane>/evaluation.html
```

For release-profile strict verification, use:

```bash
invarlock verify --profile release --assurance strict \
  --baseline /path/to/retained/baseline/report.json \
  --policy-pack /path/to/acceptance-policy-pack.json \
  --runtime-provenance container \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/<target>/<artifact-lane>/evaluation.report.json
```

Supply the baseline report and policy pack from independent storage. Set
`TRUSTED_RUNTIME_IMAGE_DIGEST` from release policy or another
channel independent of the report bundle being verified.

For the primary CUDA/container strict lane, `<artifact-lane>` is
`cuda-container-strict`.
