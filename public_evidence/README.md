# Public evidence

This directory is the public index for evidence created with the canonical
InvarLock transaction. Each published entry contains an
`invarlock/evidence-pack-v1` bundle and its independently signed verification
receipt.

Every current entry uses a 400-record paired schedule selected from pinned
public datasets. Text comparisons use a domain-balanced MMLU-Pro suite;
vision-text comparisons use a subject-balanced MMMU-Pro Vision suite. Both
suites are exactly balanced across answer choices A-J. The checked-in
[qualification-suite manifest](../docs/reference/qualification-suites.manifest.json) records the
source revisions, licenses, exclusions, deterministic selection algorithm,
distributions, selected IDs, and schedule digests. See the
[publication guide](../docs/user-guide/public-evidence.md#published-evidence)
for the complete inventory and interpretation boundary.

When evidence is added, the source repository may carry the complete pack or a
hash-bound external release asset. Installed wheels contain only the compact
index. The signed bundle includes the policy snapshot used for its evaluation.
The verifier's independently obtained copy of that policy, expected artifact-
identity and canonical-schedule digests, expected runtime digests,
evidence-signer and verifier fingerprints, and private signing key are trust
inputs. Those verifier-owned inputs remain outside the submitted bundle and
arrive through independent channels; verification compares them with the
corresponding signed evidence. GGUF evidence adds an independently approved
normalized-request digest to those inputs.

Verification requires an external policy, both expected artifact-identity
digests, the expected canonical-schedule digest, both expected runtime digests,
the expected evidence-signer fingerprint, and a verifier identity and signing
key. When either side uses `llama_cpp`, it also requires the independently
approved normalized-request digest.
See the [verification guide](../docs/user-guide/evidence-and-verification.md)
for the trust model. A complete command is:

```bash
invarlock verify PATH/TO/EVIDENCE \
  --policy policy.json \
  --expected-baseline-artifact "$BASELINE_ARTIFACT_DIGEST" \
  --expected-subject-artifact "$SUBJECT_ARTIFACT_DIGEST" \
  --expected-schedule "$SCHEDULE_DIGEST" \
  --expected-baseline-runtime "$BASELINE_RUNTIME_DIGEST" \
  --expected-subject-runtime "$SUBJECT_RUNTIME_DIGEST" \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity verifier.example
```

When either evidence side uses `llama_cpp`, also pass the independent request
anchor as `--expected-request-digest "$REQUEST_DIGEST"` before `--receipt`.

Render its signer-authenticated comparison report with:

```bash
invarlock report PATH/TO/EVIDENCE
```

Publication review must validate the pack and its receipt independently. A
report authenticates the signed bundle; it does not authorize the verifier or
replace the signed verification receipt. Systems relying on a receipt should
use the Python verification facade described in the
[API guide](../docs/reference/api-guide.md#verify-a-signed-receipt) with an
independently recorded verifier identity and fingerprint.

The complete local-entry, external release-asset, disclosure-review, and
empty-state procedures are in the
[public evidence guide](../docs/user-guide/public-evidence.md). After changing
the source index or adding a local entry, synchronize and audit the compact
package copy:

```bash
make public-evidence-sync
make public-evidence-audit
```

The synchronizer derives local entries from `evidence.meta.json`. An
external-only entry is instead written to this directory's source index with
an HTTPS locator and canonical archive digest, then copied to the package by
the same synchronization command.
