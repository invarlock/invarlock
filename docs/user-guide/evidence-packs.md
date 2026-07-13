# Evidence packs

An evidence pack is a portable, integrity-checked directory that binds an
evaluation report to its runtime manifest, independent baseline, policy,
source provenance, resolved inputs, and final verdict. A signed manifest makes
file replacement detectable; it does not by itself establish that the signer
is trusted or that the reported evaluation happened as claimed.

## Public interface

The public package defines the pack format and provides inspection,
verification, and exact-set verification for catalog-bound artifacts.

The package-native interface is under the advanced namespace:

```bash
invarlock advanced evidence-pack --help
invarlock advanced evidence-pack inspect /path/to/pack --json
invarlock advanced evidence-pack verify /path/to/pack --strict --json
```

The `scripts/evidence_packs/verify_pack.sh` wrapper provides the same
verification boundary to shell-based workflows.

## What strict verification means

Strict verification checks the signed manifest, checksums, allowed file set,
provenance references, bundled reports, runtime-image trust anchor, and the
external policy material required by the selected assurance mode. The signer
must also be anchored independently with an expected fingerprint or trust
store.

A PASS final verdict participates in a complete verification alongside the
exact reports, required materials, signed content, and independent trust
inputs.

Example:

```bash
invarlock advanced evidence-pack verify /path/to/pack \
  --strict \
  --expected-fingerprint "$TRUSTED_SIGNER_FINGERPRINT" \
  --report-assurance strict \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  --json
```

`--skip-verify` returns the distinct integrity-diagnostic status. Assurance
automation uses the normal verification path shown above.

## Catalog-bound verification

The checked public evidence catalog fixes the lane, model ID, adapter, dataset
coordinates, preset, execution policy, and required artifact roles for one
evaluation. Each pack records authenticated model and dataset revisions plus
source provenance for comparison with independently supplied trust anchors.

The v1 execution policy is closed and authenticated with the catalog entry. It
pins the profile bytes, tier, strict container mode, no-op edit, and evaluation
window counts. Every v1 entry uses the pinned release profile at 400 preview and
400 final windows across data modalities. Verification derives the expected
report policy from that entry.

Use command help for the exact required arguments in the installed version:

```bash
invarlock advanced evidence-catalog --help
invarlock advanced evidence-pack verify --help
invarlock advanced evidence-pack verify-set --help
```

Verification must receive the catalog digest from an independent channel:

```bash
invarlock advanced evidence-pack verify /path/to/catalog-pack \
  --strict \
  --expected-catalog-digest "$EXPECTED_CATALOG_DIGEST" \
  --expected-fingerprint "$TRUSTED_SIGNER_FINGERPRINT" \
  --report-assurance strict \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  --json
```

For the complete replacement set, `verify-set` additionally requires the
intended source commit and source-bundle digest. Its receipt records those
anchors and each pack's authenticated values, so a same-signer pack from an
older source tree cannot satisfy the set:

```bash
invarlock advanced evidence-pack verify-set \
  --catalog contracts/evidence_catalog_v1.json \
  --expected-catalog-digest "$EXPECTED_CATALOG_DIGEST" \
  --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --expected-source-bundle-digest "$EXPECTED_SOURCE_BUNDLE_DIGEST" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  --expected-fingerprint "$TRUSTED_SIGNER_FINGERPRINT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --pack /path/to/pack \
  --receipt /path/to/set-receipt.json \
  --json
```

## Pack contents

The authoritative manifest enumerates every authenticated file. A
catalog-bound pack includes these logical roles:

| Role | Purpose |
| --- | --- |
| Evaluation report | Metrics, guard evidence, subject identity, and report-local gates |
| Runtime manifest | Execution configuration and runtime-image binding |
| Final verdict | Verdict bound to the exact report set |
| Independent baseline | Raw baseline measurements used by strict verification |
| Policy pack | External acceptance policy matched during verification |
| Source provenance | Signer-asserted source commit and source-bundle digest, set-wide checked against independent anchors |
| Resolved inputs | Signer-asserted immutable model and dataset coordinates |
| Runtime config and preset | Exact evaluated configuration and its deterministic input |
| Catalog | Entry identity and required artifact roles |

Additional required material, such as a vision dataset materialization, is
declared by the selected catalog entry. Strict mode enforces the manifest's
exact allowed file set.

## Exit status

The evidence-pack command uses separate nonzero statuses for usage, missing
inputs, format errors, signature failures, integrity failures, report
verification failures, and integrity-only diagnostics. Automation should use
the process status and machine-readable `ok` field rather than matching human
output.

## Related references

- [Public contracts](../reference/contracts.md)
- [Runtime provenance](../security/runtime-provenance-guide.md)
- [Reports](../reference/reports.md)
- [Trust model](../assurance/14-trust-model.md)
