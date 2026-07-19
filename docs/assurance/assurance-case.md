# Assurance case

!!! abstract "Assurance note"
    **In plain language:** A pass records that one subject met one policy
    against one baseline on the fixed schedule. Broader quality, safety, and
    deployment decisions use complementary evidence.

    **Question:** What decision does a strictly verified InvarLock transaction
    support, and which broader questions use complementary evidence?

    **Decision use:** Use this page to decide whether a signed receipt and its
    operational assumptions are sufficient for a particular acceptance case.

    **Evidence:** Authenticated inputs, paired provider records, deterministic
    replay, the signed evidence pack, independent anchors, and the
    verifier-signed receipt.

InvarLock supports one narrow assurance claim: a named subject satisfied a
caller-approved acceptance policy relative to a named baseline on one authenticated,
ordered evaluation schedule, under the runtime identities recorded for that
transaction.

The claim covers the submitted records. Questions about other inputs, model
safety, model intent, and deployment fitness remain explicit complementary
review domains.

## Top-level claim

Let:

| Symbol | Meaning |
| --- | --- |
| $E$ | Submitted `invarlock/evidence-pack-v1` directory |
| $M$ | Canonical manifest in $E$ |
| $\pi_v$ | Policy bytes independently supplied by the verifier |
| $A_B, A_S$ | Independently supplied baseline and subject artifact-identity digests |
| $d_{\mathcal S}$ | Independently supplied canonical schedule digest |
| $R_B, R_S$ | Independently supplied baseline and subject runtime digests |
| $K_E$ | Independently authorized evidence-signer public-key fingerprint |
| $K_V, I_V$ | Verifier key fingerprint and stable verifier identity |
| $D(E,A_v)$ | Deterministic replay result for evidence $E$ under verifier anchors $A_v=(\pi_v,A_B,A_S,d_{\mathcal S},R_B,R_S,K_E)$ |

The machine acceptance predicate is:

$$
\operatorname{Accept}(E,A_v)
\iff
\operatorname{Integrity}(E)
\land
\operatorname{EvidenceSignerAuth}(E,K_E)
\land
\operatorname{BindingsReplay}(E,A_B,A_S,d_{\mathcal S},R_B,R_S)
\land
\operatorname{PolicyDigest}(E)=H(\pi_v)
\land
D(E,A_v)=\texttt{pass}.
$$

A signed receipt records this predicate's result, the digest of $M$, $A_v$,
$K_V$, and $I_V$. The predicate is deliberately narrower than the prose claim:
relying on it also requires the operational assumptions below.

## Assurance argument

```text
fixed inputs + paired provider records
                    |
                    v
 deterministic comparison + interval replay
                    |
                    v
       canonical signed evidence bundle
                    |
                    v
 independent anchors + verifier-signed receipt
```

The argument has six parts. A claim is supported only when every applicable
row succeeds.

| Claim | Evidence | Enforcement |
| --- | --- | --- |
| The comparison names closed inputs. | The normalized request and input identities bind the baseline, subject, schedule, policy, both runtime digests, and an exact scorer binding when selected. | The request loader rejects unknown fields, unsafe paths, ambiguous YAML, missing inputs, unsupported metrics, and incomplete scorer bindings. The publisher checks the request against provider identities, material digests, and the independently authorized scorer registry. |
| Both sides evaluated the same ordered records. | The canonical schedule contains one task, unique record IDs, authenticated ordered input parts, expected outputs, and dataset coordinates. Each provider observation binds the schedule digest and repeats the ordered record IDs and input digests. | Pair derivation rejects missing, failed, duplicated, reordered, task-mismatched, or input-mismatched records. See [Pairing and replay](pairing-and-replay.md). |
| The decision follows the declared arithmetic. | The bundle contains runtime facts, verifier-derived paired scores, the canonical comparison report, its metric-specific paired interval, optional sample qualification, the exact policy digest, and scorer replay results when selected. | Verification reconstructs every built-in score or replays an explicitly authorized scorer twice, then reconstructs each paired statistic or interval replicate, comparison, optional count/width checks, and verdict from the bound records and independently supplied policy. See [Decision semantics](decision-semantics.md). |
| The published evidence has not changed. | `manifest.json`, `checksums.sha256`, the complete file inventory, and the Ed25519 evidence signature bind the bundle. | Strict verification rejects checksum gaps, extra files, unsafe paths, JSON that is not canonical, signature failure, or an evidence-signer fingerprint different from the caller's anchor. |
| Acceptance uses an independent trust decision. | The verifier supplies the policy bytes, expected artifact identities, canonical schedule, runtime digests, evidence-signer fingerprint, verifier identity, and verifier key outside the bundle. | `invarlock verify` replays the bundle under those anchors and writes a signed receipt outside the immutable evidence directory. |
| People see the same decision as machines. | Console and HTML output are rendered from the canonical comparison report. | `invarlock report` authenticates bundle integrity and the evidence signature before rendering; it does not alter evidence or create an acceptance verdict. |

The public contracts are summarized in
[Public contracts](../reference/contracts.md). The principal implementation
paths are the
[evaluation transaction](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evaluation_transaction.py),
[evidence contract](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evidence_pack_contract.py),
and
[independent verifier](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evidence_pack_verification.py).

## Assumptions and defeaters

| Assumption | Why it matters | Defeater |
| --- | --- | --- |
| Evidence signer and verifier fingerprints were authorized through independent identity processes. | A valid signature authenticates a key, not an organization or role by itself. | Fingerprint copied from submitted evidence, unknown key owner, revoked or compromised key. |
| Verifier anchors are controlled outside the evidence submission path. | Otherwise the evidence signer can choose the expected policy, artifacts, schedule, runtimes, and signer. | Anchors parsed from the bundle or supplied by the same actor without independent review. |
| Schedule and policy were selected before subject results were inspected. | Post-result selection can preserve perfect internal consistency while biasing the result. | Cherry-picked records, changed expected outputs, favorable run selection, or threshold tuning. |
| Provider measurements are credible enough for the decision context. | Replay checks consistency, not physical execution truth. | Compromised evaluation environment or provider, missing external attestation where one is required. |
| Selected scorer code was reviewed and independently authorized. | A scorer extension executes code that determines per-record values, although the core retains aggregation and policy arithmetic. | Scorer authorization copied from the request, substituted installed package, malicious scorer code, or an extension that uses a network, external model, human judgment, or LLM judge. |
| The baseline is an appropriate reference. | The result is relative to that baseline. | Backdoored, obsolete, misidentified, or otherwise unsuitable baseline. |
| The finite schedule matches the intended decision scope. | No population inference or representativeness test is performed. | Unsupported generalization to users, tasks, languages, or conditions not scheduled. |

A verifier can return `ok: true` while an operational assumption is false.
That is not a verifier bug: the assumption lies outside the bytes the verifier
can observe. Decision owners must record assumption failures as nonacceptance or as a
separate exception, never by weakening replay.

## Policy provenance

The request and evidence bind exact policy bytes, and verification requires an
independently supplied byte-identical copy. This gives policy **identity** and
prevents a submitted bundle from silently substituting a different threshold.

The engine does not establish who authored, inspected, or approved the policy.
Maintain that provenance outside the pack with, at minimum:

- policy digest and version or change identifier;
- author and approver identities;
- metric and threshold rationale;
- effective and retirement dates;
- applicable artifact, schedule, and runtime scope; and
- exception and rollback procedure.

A policy edit changes its digest and requires a new evidence pack. The current
contract does not re-evaluate one existing pack under different policy bytes.

## What a passing result establishes

A passing signed verification receipt establishes that:

- the verifier authenticated the exact `invarlock/evidence-pack-v1` manifest named by
  the receipt;
- the bundle was signed by the evidence-signing key whose fingerprint the verifier
  independently expected;
- the bundle inventory, checksums, input bindings, runtime-integration bindings,
  schedule, paired records, runtime manifests, comparison arithmetic, paired
  interval, and optional sample qualification replayed without error;
- the independently supplied policy, artifact-identity, schedule, and runtime
  digests matched the bound identities; and
- the replayed finite-schedule decision passed that policy using the required
  conservative interval bound and any configured count and width controls.

The receipt also identifies the verifier key and verifier identity that made
that statement. It does not silently upgrade the statement beyond those
facts.

## Result states

Use three review states rather than collapsing every outcome into pass/fail:

| State | Meaning | Action |
| --- | --- | --- |
| Verified pass | Integrity and replay succeeded; policy verdict is `pass`; required operational assumptions were explicitly approved. | May support the scoped decision recorded by the decision owner. |
| Verified policy fail | Integrity and replay succeeded; policy verdict is `fail`. | Preserve the signed failure receipt; do not publish or describe as accepted. |
| Not verified | Format, inventory, signature, anchor, runtime, pairing, or replay failed. | Treat as no accepted evidence. Diagnose without bypassing the failed check. |

An incompatible or not-yet-evaluated lane belongs in the third category. It is
not a negative model result and must not be converted to one.

## Decision scope and complementary evidence

The signed receipt supports the recorded finite-schedule decision: it binds the
named artifacts, exact ordered schedule, policy, per-side runtimes, signer, and
verifier to deterministic replay. Broader reliance calls for evidence matched
to the additional question:

| Additional decision question | Complementary evidence |
| --- | --- |
| Did the declared runtime execute the workload? | Independently controlled rerun or measured-execution attestation bound to the workload and evidence digest |
| Does the result generalize beyond the scheduled records? | Sampling design fixed in advance, representativeness analysis, and a separately defined population-inference protocol |
| Is the baseline an appropriate reference? | Baseline provenance, suitability review, and domain-specific validation |
| Is the subject suitable for deployment? | Safety, security, privacy, robustness, and operational review for the intended use |
| Were submitted measurements truthful? | Execution controls or attestation independent of the evidence-signing environment |

Signatures authenticate the signed bytes and identities. The operational
assumptions above and complementary evidence determine how far a decision can
rely on those authenticated facts. See
[Reproducibility and provenance](reproducibility.md) and the
[Threat model](../security/threat-model.md) for the corresponding controls.

## Evidence ownership

`evaluate` owns evidence creation, `verify` owns the scoped acceptance result,
and `report` owns presentation. Keep those transactions and their keys separate:

```bash
invarlock evaluate request.yaml
invarlock verify evidence/ \
  --policy policy/acceptance.json \
  --expected-baseline-artifact sha256:BASELINE_ARTIFACT_DIGEST \
  --expected-subject-artifact sha256:SUBJECT_ARTIFACT_DIGEST \
  --expected-schedule sha256:SCHEDULE_DIGEST \
  --expected-baseline-runtime sha256:BASELINE_RUNTIME_DIGEST \
  --expected-subject-runtime sha256:SUBJECT_RUNTIME_DIGEST \
  --expected-signer sha256:EVIDENCE_SIGNER_KEY_FINGERPRINT \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity release-verifier
invarlock report evidence/ --html evidence.html --explain
```

Use the [acceptance checklist](acceptance-checklist.md) before relying on the
receipt.

## Standards context

InvarLock uses:

- SHA-256 as specified by the
  [NIST Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4) for material and
  inventory digests;
- Ed25519 as specified by
  [RFC 8032](https://www.rfc-editor.org/rfc/rfc8032) for evidence-signer and verifier
  signatures; and
- JSON as specified by
  [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259), constrained by
  InvarLock's versioned canonical file profiles.

These references define primitives and encodings. They do not extend the
assurance claim beyond the implemented checks and documented assumptions.
