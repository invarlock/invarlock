# Public evidence

This directory is the public index for evidence created with the canonical
InvarLock transaction. Each published entry must contain an `evidence-pack-v1`
bundle and its independently signed verification receipt.

No current evidence has been created yet. The index remains useful in this
state because it distinguishes an empty evidence set from a missing or damaged
publication surface.

When evidence is added, the source repository may carry the complete pack or a
hash-bound external release asset. Installed wheels contain only the compact
index. Policies, expected artifact-identity and canonical-schedule digests,
expected runtime digests, evidence-signer fingerprints, verifier fingerprints,
and private keys are trust inputs; they must remain outside the signed bundle
and arrive through independent channels.

Verification requires an external policy, both expected artifact-identity
digests, the expected canonical-schedule digest, both expected runtime digests,
the expected evidence-signer fingerprint, and a verifier identity and signing
key.
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
