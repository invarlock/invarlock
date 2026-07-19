# Runnable examples

The primary [`run/`](run/) example creates two distinct tiny Hugging Face
checkpoints and executes a meaningful CPU release-regression decision through
the complete OCI-backed `evaluate -> verify -> report` path. It is the shortest
checked-in demonstration of real provider scoring and independent anchors.

The files at this directory root form a second, offline import-mode transaction.
They use small synthetic provider records so the evidence, signature,
trust-anchor, policy, and report contracts run without downloading a model,
contacting a service, or using a GPU. Those fixtures test the import contract;
they are not measurements of real models.

## Run the primary Hugging Face journey

Follow [`run/README.md`](run/README.md) to build and smoke-test the CPU runtime,
generate distinct tiny checkpoints, execute the two isolated workers, verify
the resulting pack with independently derived artifact/schedule/runtime/policy
anchors, and render the report.

## What is included

```text
examples/
├── request.yaml
├── rejected-request.yaml
├── generate_keys.py
├── run_trust_boundary_demo.py
├── inputs/
│   └── schedule.json
├── policy/
│   └── acceptance.json
├── trusted-inputs/
│   └── input-digests.json
└── import/
    ├── paired-records.json
    ├── rejected-paired-records.json
    ├── baseline/
    │   ├── model-artifact.identity.json
    │   ├── runtime-provider.receipt.json
    │   ├── runtime-scoring.observation.json
    │   ├── runtime.manifest.json
    │   ├── report.json
    │   └── run.yaml
    ├── subject/
    │   └── ...the same six provider files
    └── rejected-subject/
        └── ...the same six provider files with one wrong answer
```

The imported runtime identities are deliberately obvious fixture digests:

- baseline: `sha256:1111111111111111111111111111111111111111111111111111111111111111`
- subject: `sha256:2222222222222222222222222222222222222222222222222222222222222222`

They are authenticated test coordinates, not references to available images.

## Run the offline import journey

From a source checkout, install that checkout, enter this directory, and create
two separate keys:

```bash
python -m pip install -e .
cd examples
python generate_keys.py --output-dir .keys
```

For a published release, install the matching `invarlock` version instead.

Produce the signed bundle:

```bash
invarlock evaluate request.yaml \
  --signing-key .keys/evidence-signer.pem
```

Read the evidence-signer fingerprint from the independently generated anchor,
then verify with every required trust input and write a signed receipt:

```bash
EVIDENCE_SIGNER_FINGERPRINT="$(tr -d '\n' < .keys/evidence-signer.fingerprint)"
BASELINE_ARTIFACT_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["baseline_artifact"])')"
SUBJECT_ARTIFACT_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["subject_artifact"])')"
SCHEDULE_DIGEST="$(python -c 'import json; print(json.load(open("trusted-inputs/input-digests.json"))["canonical_schedule"])')"

invarlock verify artifacts/evidence/ \
  --policy policy/acceptance.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT_DIGEST" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT_DIGEST" \
  --expected-schedule "$SCHEDULE_DIGEST" \
  --expected-baseline-runtime sha256:1111111111111111111111111111111111111111111111111111111111111111 \
  --expected-subject-runtime sha256:2222222222222222222222222222222222222222222222222222222222222222 \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt verification.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity local-example-verifier
```

Render the signer-authenticated report in the terminal and as HTML:

```bash
invarlock report artifacts/evidence/
invarlock report artifacts/evidence/ --html evidence.html --explain
```

Expected decision: both exact-match means are `1` and their point delta is `0`
percentage points. The 50 paired records produce a verifier-replayed confidence
interval of about `[-7.13, 7.13]` percentage points. The policy requires at
least 50 records, an interval no wider than 20 percentage points, and a lower
bound of at least `-10`; all three requirements pass. The report verifies the
bundle's evidence signature but does not replace the independent signed
verification receipt.

## Run the trust-boundary demonstration

The demonstration runner creates isolated evaluation and verifier workspaces.
The evidence signer authenticates one accepted comparison and one comparison
with a report-local policy failure. Only the immutable evidence directories
move into the verifier submission area. The verifier uses a separate private
key and separately provisioned artifact, schedule, policy, runtime, and signer
anchors, then records:

- a signed acceptance receipt for the parity comparison;
- a signed rejection receipt for the one-record (`-2` percentage-point)
  regression whose lower confidence bound falls below the policy floor; and
- a signed integrity rejection after changing one report byte.

From the repository root:

```bash
make trust-boundary-demo
```

The target refreshes its ignored disposable workspace under
`examples/artifacts/trust-boundary-demo` before each run. Direct script
invocations remain no-clobber and require a new `--workspace`. The failure
scenario is useful for demonstrating decision semantics, not for accepted
public evidence.

The output destinations are no-clobber. For another tutorial run, work in a
fresh copy of this directory or choose a new evidence and receipt destination.
Do not delete or rewrite evidence that has already been distributed.

For the contracts behind the example, read the
[getting-started guide](../docs/user-guide/getting-started.md),
[schedule and policy guide](../docs/user-guide/schedule-and-policy.md), and
[evidence and verification guide](../docs/user-guide/evidence-and-verification.md).
