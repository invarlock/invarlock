# Evidence and verification

`invarlock evaluate` publishes one closed evidence directory governed by
[`contracts/evidence_pack.schema.json`](https://github.com/invarlock/invarlock/blob/main/contracts/evidence_pack.schema.json).
The bundle is the portable machine record; a signed verification receipt is the
independent acceptance record.

!!! tip "User guide"

    **Outcome:** Verify an immutable evidence bundle against independent
    anchors, validate its signed receipt, and render a human-readable report.

    **Audience:** Verifier operators, decision owners, and downstream receipt
    verifiers deciding whether to rely on one evidence transaction.

    **Prerequisites:** A complete evidence directory, independently obtained
    policy, runtime digests and evidence-signer fingerprint, plus an independently
    selected verifier key and identity when issuing a receipt.

## Artifact lifecycle at a glance

```text
closed request
    |
    | evaluate + evidence-signing key
    v
immutable evidence/  ----------------------+
    |                                       |
    | verify + independent anchors          | report
    v                                       v
verification.receipt.json               evidence.html
```

The evidence directory is the only signed evaluation artifact. The receipt is a
separate verifier statement. HTML is a reproducible view. Keep all three roles
distinct even when one automation job creates them in sequence.

## Canonical evidence layout

A completed bundle contains fixed roles similar to:

```text
evidence/
├── manifest.json
├── checksums.sha256
├── manifest.signature.json
├── request.json
├── inputs/
│   ├── baseline.json
│   ├── subject.json
│   ├── baseline_runtime.json
│   ├── subject_runtime.json
│   ├── dataset.json
│   └── policy.json
├── schedule/
│   └── runtime-behavioral-schedule.json
├── providers/
│   ├── baseline/...
│   └── subject/...
├── runs/
│   ├── baseline/report.json
│   └── subject/report.json
├── records/
│   └── paired-records.json
└── reports/
    └── evaluation.report.json
```

The schema pins the allowed paths. The signed manifest binds the normalized
request, every input identity, schedule, policy, provider material, paired
records, canonical report, and checksum inventory. Unexpected, missing,
non-canonical, changed, or symbolic-link-backed files fail verification.

For the exact file-by-file contract, see
[Evidence artifacts](../reference/artifacts.md). A normal user should not
construct or repair this directory by hand; `evaluate` creates and signs it.

## Evidence signature

The evidence signature proves that one Ed25519 key signed the manifest and that
the manifest binds the bundle bytes. It does not establish that the signer was
authorized, that a runtime actually executed as described, or that submitted
measurements are true. Those questions require independent trust anchors,
provider controls, and review.

The evidence-signer fingerprint is:

```text
sha256(EVIDENCE_SIGNER_PUBLIC_KEY_RAW_ED25519_BYTES)
```

It is not the hash of a PEM or DER file. Obtain the expected fingerprint from a
trusted key registry or authenticated handoff, never from the submitted bundle.

## Independent verification inputs

The verifier supplies all acceptance authority from outside the evidence:

1. exact policy bytes;
2. expected baseline artifact-identity digest;
3. expected subject artifact-identity digest;
4. expected canonical schedule digest;
5. expected baseline runtime digest;
6. expected subject runtime digest;
7. expected evidence-signer fingerprint;
8. a new receipt destination;
9. an independent Ed25519 verifier key; and
10. a stable verifier identity.

Example:

```bash
invarlock verify artifacts/evidence/ \
  --policy policy/acceptance.json \
  --expected-baseline-artifact sha256:a42368dfc967f48cebd2e05805e0d6f3467be592257c1e4a41608f405872557e \
  --expected-subject-artifact sha256:d5581534ea53bc92c68c48aa69ea2f9bd316d300074f46cdbdc3cdd3d67d3e37 \
  --expected-schedule sha256:ac60be37db4b0b6d8cc607822ca0a9d7bfe42c42b976015c4aace9226073b2ea \
  --expected-baseline-runtime sha256:1111111111111111111111111111111111111111111111111111111111111111 \
  --expected-subject-runtime sha256:2222222222222222222222222222222222222222222222222222222222222222 \
  --expected-signer "$EVIDENCE_SIGNER_FINGERPRINT" \
  --receipt verification.receipt.json \
  --verifier-signing-key .keys/verifier.pem \
  --verifier-identity local-example-verifier
```

Equivalent environment variables exist for the policy, artifact, schedule,
runtime, evidence-signer, verifier-key, and verifier-identity inputs. Supplying
explicit arguments makes a recorded invocation easier to review. `--receipt`
is kept explicit because its output is transaction-specific.

For routine use, place the same independent values in a closed
`invarlock/trust-inputs-v1` profile and replace the explicit trust options with
`--trust-profile trust/trust-inputs.json`. Environment variables cannot
override a profile, and the signed receipt records its canonical digest.

### Obtain each anchor independently

| Verify input | Good source | Circular source to avoid |
| --- | --- | --- |
| Policy file | Authorized configuration or immutable policy distribution | `inputs/policy.json` copied from the submitted pack |
| Artifact-identity digests | Approved build or artifact-registry record over the canonical provider identity bytes | Baseline or subject digest read from the submitted manifest |
| Canonical schedule digest | Approved evaluation-plan record over the exact canonical schedule bytes | Dataset digest read from the submitted manifest |
| Runtime digests | Pinned image build or deployment record | Runtime manifest inside the pack |
| Evidence-signer fingerprint | Authorized signer registry or authenticated handoff | Public key embedded in the pack signature |
| Verifier key and identity | Verifier-owned configuration | Values proposed by the evidence signer |

Independent does not necessarily mean a different storage product. It means
the submitter cannot rewrite both the evidence and the value intended to check
it as one transaction.

## What verification replays

The verifier:

- snapshots the evidence without trusting manifest-controlled paths;
- validates the closed manifest and signature structures;
- pins the evidence-signing public key to the external fingerprint;
- verifies checksums, inventory, canonical JSON, and signature bytes;
- compares both artifact identities and the canonical schedule to external
  digests;
- compares exact external policy bytes to the bound policy digest;
- compares both provider runtime identities to external runtime digests;
- validates artifact, provider, observation, schedule, and report cross-bindings;
- re-derives every built-in paired score from typed observation facts or,
  when selected, replays an explicitly authorized deterministic scorer twice;
- recomputes the metric, its paired interval, threshold comparison, optional
  count/width qualification, and report; and
- signs the complete result and caller-supplied anchors with the verifier key.

Multiple verifiers may independently obtain the same policy bytes bound into
the pack and issue receipts under their own verifier identities and keys. If a
verifier's policy digest does not equal the policy bound into this evidence
version, verification rejects the bundle instead of silently reinterpreting it.
Applying different policy bytes requires a new evidence transaction.

## Read the verification result

Use `--json` in automation. A successful result includes the core verifier
fields plus the receipt path and verifier identity. Check the structured
meaning rather than a human substring:

```json
{
  "ok": true,
  "integrity_ok": true,
  "reports_verified": true,
  "verification_scope": "paired_comparison",
  "assurance_status": "verified",
  "policy_verdict": "pass",
  "authenticity": "pinned",
  "signed_receipt": "verification.receipt.json",
  "verifier_identity": "local-example-verifier"
}
```

The real output contains additional format, anchor, warning, error, and signer
details. Treat the example as a field guide, not a replacement schema.

| Exit/result | Interpretation | Action |
| --- | --- | --- |
| Exit `0`, `ok: true`, integrity true, policy pass, signer pinned | Scoped verification passed | Validate and retain the signed receipt. |
| Nonzero, integrity true, policy fail | Evidence replayed, but the finite-schedule acceptance rule failed | Retain rejection when useful; do not call it an accepted result. |
| Nonzero, integrity false | Bundle or semantic binding could not be trusted | Quarantine the submission and diagnose the earliest integrity error. |
| Receipt exists but command failed | The verifier may have signed a rejection | Validate the receipt and inspect its signed verdict; existence alone is not success. |
| Command failed before receipt exists | Verification did not reach a signable result | Fix the structural or input problem and use a new receipt path. |

## Signed receipt

The receipt is canonical JSON containing:

- the evidence manifest digest;
- external policy digest;
- baseline and subject runtime anchors;
- expected evidence-signer fingerprint;
- verifier identity and verifier-key fingerprint;
- integrity and policy verdicts; and
- an Ed25519 signature plus the verifier public key.

The receipt destination must be outside the evidence directory and must not
already exist. It is written read-only. A failed verification can still produce
a valid receipt that records the rejection; the command then exits nonzero.
Validate the verifier fingerprint independently before relying on the receipt.

### Validate a received receipt

Receipt verifiers can verify the verifier signature and all caller-owned
bindings without adding a fourth CLI command:

```python
from pathlib import Path

from invarlock.engine import verify_signed_verification_receipt

verification = verify_signed_verification_receipt(
    Path("verification.receipt.json"),
    Path("evidence"),
    policy_path=Path("trusted/acceptance.json"),
    expected_runtime_digests={
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    },
    expected_pack_signer_fingerprint="sha256:" + "3" * 64,
    expected_verifier_identity="release-verifier",
    expected_verifier_fingerprint="sha256:" + "4" * 64,
)

if not verification.ok:
    raise SystemExit("; ".join(verification.errors))
```

The function binds the receipt to the supplied pack manifest, policy bytes,
runtime digests, evidence-signer fingerprint, verifier identity, and verifier
fingerprint. It does not discover authorization from the receipt itself.

## Report semantics

```bash
invarlock report artifacts/evidence/
invarlock report artifacts/evidence/ --html evidence.html --explain
```

`report` authenticates the bundle's internal integrity and embedded evidence
signature before rendering. It shows:

- comparison identifier, built-in metric or scorer identity, and record count;
- baseline and subject means;
- exact-match delta, normalized-NLL ratio, or scorer-extension delta;
- the metric-specific paired interval;
- paired exact-match regressions, improvements, exact McNemar probability, and
  Newcombe 95% interval when exact match is selected;
- sample qualification showing the required and observed paired count and
  interval width when the policy enables those coupled controls;
- a token-weighted perplexity interpretation when normalized-NLL tokenizer and
  token-count facts are comparable;
- the scorer binding and deterministic per-side replay when a scorer extension
  is selected;
- policy limit and the verdict formed from the conservative bound plus any
  configured count and precision requirements;
- policy digest; and
- evidence-signer fingerprint.

The report verifies the embedded signature without pinning that signer to an
independent allowlist, and it does not validate a verification receipt. Treat
it as a human view of signed evidence, not as the independent acceptance result.

Use the report to answer presentation questions:

- Which artifacts, built-in metric or scorer, schedule size, and evidence
  signer does this bundle name?
- What baseline and subject means were recorded?
- What comparison value and threshold produced the bound verdict?
- Which policy digest is bound into the report?

Use verification and its receipt to answer the acceptance question: did an
independent verifier replay this exact manifest under separately recorded
anchors?

## Store and transfer

Keep the immutable records together:

```text
review/
├── evidence/                    # signed canonical evidence bundle
├── verification.receipt.json   # verifier-signed decision and anchors
└── evidence.html               # optional, regenerable view
```

Transfer the evidence and public anchors, never either private key. Preserve the
directory structure and file bytes. Archive transport hashes if your storage
system supplies them, but do not substitute a transport archive hash for
InvarLock's internal verification.

### Handoff checklist

The evaluation handoff contains:

- the complete immutable evidence directory;
- the comparison or release identifier needed by the decision process; and
- no private signing key.

The verifier obtains elsewhere:

- the policy file;
- expected baseline and subject runtime digests;
- expected evidence-signer fingerprint;
- its own identity and private key; and
- a new receipt destination.

The receipt verifier obtains elsewhere:

- expected verifier identity and fingerprint; and
- the same policy, runtime, and evidence-signer anchors named by the receipt.

After transport, rerun receipt validation against the received evidence and
anchors. Do not rely solely on a successful check performed before transfer.

## Failure and recovery rules

- **Never patch a signed bundle.** Recover the original immutable copy or
  produce a new transaction.
- **Never reuse a receipt path.** Signed acceptance and rejection records are
  no-clobber.
- **Never substitute embedded anchors.** Fix the external configuration or
  reject the submission.
- **Never apply a new policy to an old report.** A policy change requires new
  evidence because the policy identity and canonical report are bound.
- **Never hide a failed attempt by overwriting it.** Follow the review process
  for retention, classification, and a new comparison identifier.
- **Never put HTML or receipts inside evidence.** Extra files make the closed
  inventory fail.

## Acceptance checklist

Before relying on a result, confirm:

- the command exited with the expected status;
- a signed receipt exists outside the bundle;
- the receipt verifier identity and fingerprint match an independent registry;
- policy and both runtime anchors came from the decision authority;
- the evidence-signer fingerprint was pinned independently;
- no verification error was waived;
- the metric and threshold match the intended claim; and
- the report's scope is not being generalized beyond this schedule, artifacts,
  runtimes, and policy.
