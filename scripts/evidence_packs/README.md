# Evidence-pack helper scripts

The public repository provides evidence formats, local evaluation, artifact
validation, and evidence-pack verification.

The maintained shell entry point is:

```bash
scripts/evidence_packs/verify_pack.sh --pack /path/to/pack --strict
```

The package-native advanced commands are the primary inspection and
verification interface:

```bash
invarlock advanced evidence-pack inspect /path/to/pack --json
invarlock advanced evidence-pack verify /path/to/pack --strict --json
invarlock advanced evidence-pack verify-set --help
```

The `python/` directory contains local artifact generators and pure validators
used by tests and package contracts. Editing validators replay declared
checkpoint transformations from files supplied by the caller.

## Produce one catalog lane

`scripts/model_evidence/run_catalog_lane.py` is the repo-owned production front
door for one maintained lane. It reads the catalog and independently resolved
model and dataset revisions, prepares the catalog preset, runs the paired
evaluation, strictly verifies the report, assembles and signs the evidence
pack, strictly verifies the finished pack, and then exposes the pack in a
caller-selected staging directory.

Run it from the declared runtime container with a read-only source checkout and
caller-controlled inputs:

```bash
python scripts/model_evidence/run_catalog_lane.py \
  --lane gpt2-causal-hf \
  --resolved-inputs "$RESOLVED_INPUTS" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --signing-key "$EVIDENCE_SIGNING_KEY" \
  --runtime-image "$RUNTIME_IMAGE" \
  --runtime-image-digest "$RUNTIME_IMAGE_DIGEST" \
  --source-commit "$SOURCE_COMMIT" \
  --source-bundle-sha256 "$SOURCE_BUNDLE_SHA256" \
  --out "$STAGING_ROOT/gpt2-causal-hf" \
  --device cuda \
  --allow-network
```

The container launcher sets `INVARLOCK_CONTAINER_EXECUTION=1`. The command
requires an output path outside `public_evidence/`, refuses to overwrite an
existing result, and emits both a staged pack and a sibling strict-verification
receipt. The signing key is read for the signature and is not copied into the
pack.

The workflow has three small parts:

| Surface | Responsibility |
| --- | --- |
| Repository command | Produce and verify exactly one catalog-bound staged pack. |
| Execution environment | Launch the pinned image, allocate a GPU, provide caches, and mount resolved inputs, policy, and key material. |
| Repository update | Copy a verified staged pack into the current public evidence set and update that lane's index/status entry. |

Running many lanes is repetition around this single-lane command. A platform or
ordinary job runner may provide concurrency, retries, host selection, and
key handling without changing the evidence format or production recipe.
Hardware canaries use this same command; the selected execution host changes,
while lane preparation, evaluation, verification, signing, and staging do not.

## Tests

The shell verifier and its portable runtime helper are covered by:

```bash
scripts/evidence_packs/tests/run.sh
```

Package-level evidence-pack behavior is covered by the Python test suite.
