# Captured Results

!!! tip "User guide"
    **Outcome:** Publish and independently verify captured evaluation evidence.
    **Audience:** Evaluator and release-integration engineers.
    **Prerequisites:** Paired records, a policy, and the core `invarlock` wheel.

Use the core `evaluate`, `verify`, and `report` commands when an evaluator has
already produced paired records. The request binds the baseline, subject,
policy, and output location without introducing a separate command namespace.

## Prepare the request

`invarlock evaluate --init demo --example classification` creates synthetic
paired runs, policy, and `demo/request.yaml`. `extraction` and `judge` are the
other starter choices. Replace these illustrative inputs and thresholds with
reviewed records before drawing conclusions about a model.

```yaml
format_version: invarlock/evaluation-request-v2
execution:
  mode: captured
comparison:
  baseline:
    path: baseline.json
    adapter: invarlock
  subject:
    path: subject.json
    adapter: invarlock
  policy: policy.json
output:
  evidence: evidence
```

Request paths resolve beneath the request's parent directory. CLI
`--baseline-run`, `--subject-run`, and `--output` overrides resolve from the
caller's working directory and must remain inside that request root. They change
locations, not adapters or identity pins. A source's `expected_run_digest` still
has to match after an override. Runtime/container and installed-scorer controls
do not apply to captured requests.

## Signed handoff

The evaluation operator supplies its Ed25519 key through `--signing-key` or
`INVARLOCK_SIGNING_KEY`. A recipient independently reviews the complete runs and
policy, derives their pins through `invarlock.engine.run_digest`,
`normalize_captured_request`, and `captured_request_digest`, and obtains the
evidence-signer fingerprint through an authorized channel. Do not copy anchors out of
the submitted pack or treat the operator's preflight output as recipient approval.

Use a recipient-owned `trust/trust-inputs.json` outside the pack. Replace all
digest placeholders with the independently approved values:

```json
{
  "format": "invarlock/trust-inputs-v2",
  "kind": "captured",
  "policy": {"path": "policy.json"},
  "anchors": {
    "baseline_run_digest": "sha256:...",
    "subject_run_digest": "sha256:...",
    "request_digest": "sha256:...",
    "evidence_signer_fingerprint": "sha256:..."
  },
  "verifier": {
    "identity": "recipient-verifier",
    "signing_key_path": "verifier.pem"
  }
}
```

The policy and verifier key resolve relative to the profile. The recipient owns
that key separately from the evaluation operator. `evaluate --keygen keys` creates an
Ed25519 demonstration key pair; do not share the private key between roles.
Profile paths reject traversal and symlinks. Explicit trust flags cannot be
combined with `--trust-profile`, and environment anchors cannot override it.

```bash
invarlock evaluate request.yaml --signing-key evidence-signer.pem --json
invarlock verify evidence/ --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json --json
invarlock report evidence/ --html report.html --markdown report.md \
  --junit results.xml --json
```

The equivalent explicit verification options are `--policy`,
`--expected-baseline-run`, `--expected-subject-run`, `--expected-request-digest`,
`--expected-signer`, `--verifier-identity`, `--verifier-signing-key`, and
`--receipt`. All are required. Native artifact/schedule/runtime anchors are not
substitutes. The profile's canonical digest is bound into the external signed
receipt. Receipt authentication uses
`invarlock.engine.verify_signed_verification_receipt` with the same independent
run/request/policy/signer pins and expected verifier identity/fingerprint.
Its `ok` authenticates the receipt, including an authentic rejection; separately
require `statement.verdict.ok` for a passing technical result.

## Preflight and local CI

Before publication, `evaluate --preflight --json` checks paired inputs, reviewed case
membership, and the planned work allowance without scoring or writing evidence.
Both evaluation and recipient verification accept `--max-bootstrap-draws`; each
caller owns its allowance, and increasing the evaluation allowance does not raise
the recipient's allowance. The default is 102,400,000 draws across the complete
policy, including overlapping slices.
Preflight emits `invarlock/evaluation-preflight-v3`, with `kind: captured`,
`requested_authentication`, run/policy/request digests, record/scope counts, and
required versus allowed bootstrap draws. It emits no decision, assurance, or
receipt fields. For key-free pin derivation, use the pure SDK normalization and
digest helpers; signed preflight validates the supplied signing key.

`evaluate --unsigned` publishes an explicitly unsigned local pack. It conflicts
with an explicit signing key and never falls back to signing implicitly.
`verify` rejects unsigned packs with exit `6`. If the manifest was safely examined,
it may issue a signed rejection receipt, never a positive verification receipt.
Local rendering
preserves the unsigned assurance label rather than presenting a verified result.

Successful evaluation exits `0` when it publishes evidence, even for an adverse
policy decision. For local CI, add `--fail-on-policy` to return `7` after publishing
a regression or insufficient-evidence result. Reports still describe the recorded
decision; they do not independently verify unsigned results. Preserve the gate
status while generating JUnit output:

```sh
evaluation_status=0
invarlock evaluate request.yaml --unsigned --fail-on-policy || evaluation_status=$?
case "$evaluation_status" in
  0|7) ;;
  *) exit "$evaluation_status" ;;
esac
invarlock report evidence/ --junit results.xml || exit "$?"
exit "$evaluation_status"
```

Preflight cannot be combined with `--fail-on-policy`, because it does not compute
a policy decision. A failed publication must not trigger reporting of an older
evidence directory.

`--fail-on-policy` is CLI-only: `pass` exits `0`, `regression`,
`insufficient_evidence`, or native `fail` exits `7`, and an unknown/unavailable
decision exits `2`. The successful publication JSON is retained on gate exits;
SDK evaluation continues to return its publication result. Input or local budget
refusal exits `2` without a new pack or receipt. A signed captured verification
that completes with an adverse policy decision exits `7` with its rejection
receipt. Structural/contract rejection exits `4`; authenticated binding or source
integrity rejection exits `6`. Every nonzero verification
status rejects. Retry budget refusal only after approving an adequate local
allowance, not by weakening policy.

## Outputs and assurance

Captured evaluation emits `invarlock/evaluation-result-v2`, with `kind`, `ok`,
`evidence`, `comparison_id`, baseline/subject run digests, `policy_digest`,
`request_digest`, `pack_manifest_digest`, `authentication`, `decision`, and
`policy_verdict`. Publication `ok: true` is not a claim of independent verification.
Captured verification emits `invarlock/evidence-pack-verify-v2`; its signed
`invarlock/evidence-verification-receipt-v3` has
`verification_scope: captured_comparison`, replay status and scoring assurance.
Only `invarlock/evidence-pack-v2` directories are captured evidence packs.

`report --json` emits `invarlock/evidence-report-v2` with `kind: captured`, `ok`,
`pack_manifest_digest`, `requested_outputs`, `written_outputs`, `failed_output`,
and `errors`. Output maps use `html`, `markdown`, and `junit` keys; an omitted
destination is not written. Destinations are checked together for collisions
before writing. If a later write fails, earlier outputs remain and are listed in
`written_outputs`. JUnit represents regression as failure and insufficient
evidence as error. See [reports](../reference/reports.md) and
[capacity](../reference/evaluation-capacity.md).

The captured record format is intentionally separate from native runtime
receipts. It authenticates the bytes and policy decision supplied by the
submission; it does not establish model quality, representativeness, or runtime
truth beyond those inputs. Receipt `scoring_assurance` is an ordered array of
`name`, `slice`, `kind`, and `scoring_assurance` entries, each `recomputed` or
`recorded`, according to policy, or `null` after integrity rejection. A comparison
may contain both kinds of assurance. Recorded judgments and measurements are
attributed inputs, not rerun inference. Captured packs and receipts, even when
signed and passing, are rejected by native-only acceptance attestations, ModelKit
acceptance, and deployment-approval consumers.
