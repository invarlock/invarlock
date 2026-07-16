# Pairing and replay

!!! abstract "Assurance note"
    **In plain language:** Baseline and subject results are comparable only
    when they cover the same records in the same order with the same inputs.
    Any mismatch stops verification instead of being averaged away.

    **Question:** How does verification establish that baseline and subject
    results describe the same ordered evaluation records?

    **Decision use:** Use this page to diagnose pairing failures and to judge
    whether imported or executed observations support the same replay claim.

    **Evidence:** The canonical schedule, ordered provider observations,
    record and input digests, derived paired records, and exact replay checks.

Pairing is the central comparison invariant. The baseline and subject must
produce one successful observation for every scheduled record, in exactly the
same order, with the same record identity and input digest.

## Notation

Let the canonical schedule be the ordered sequence:

$$
\mathcal{S}=\left((r_i, q_i, h_i, y_i)\right)_{i=1}^{n},
$$

where:

| Symbol | Meaning |
| --- | --- |
| $r_i$ | Unique logical record ID |
| $q_i$ | Exact ordered input-part array |
| $h_i$ | Lowercase SHA-256 of the canonical JSON bytes of $q_i$ |
| $y_i$ | Expected output |
| $\mathcal{O}_B$, $\mathcal{O}_S$ | Ordered baseline and subject observations |
| $C_0(z)$ | InvarLock's contract-defined canonical JSON bytes for $z$, without a final line feed |
| $H(z)$ | SHA-256 of bytes $z$ |
| $d_{\mathcal S}$ | Canonical schedule digest, $H(C_0(\mathcal S))$ |

The notation $C_0$ is the no-final-line-feed form used for canonical schedules
and scoring-record arrays: UTF-8, sorted keys, compact separators, finite
numbers, and preserved array order. Other public documents use the same JSON
profile with the newline behavior defined by their individual contract. This
profile is deterministic but is not a claim of conformance to the separate
JSON Canonicalization Scheme in RFC 8785.

## Canonical schedule contract

The `invarlock/runtime-behavioral-schedule-v1` document contains:

- dataset provider, name, configuration, immutable revision when required,
  and split;
- an ordered, nonempty array of at most 10,000 records;
- a unique, safe logical `record_id` for each record;
- an ordered array of authenticated text or content input parts and its
  lowercase SHA-256 digest; and
- nonempty `expected_output` used by all supported metrics.

The loader rejects unknown fields, unsafe logical names, duplicate record IDs,
empty material, hosted-dataset revisions that are not canonical, and an input
digest that does not match the canonical ordered input parts. Text parts carry
inline text and its digest; content parts carry a path-free content ID, media
type, byte length, and digest. It serializes the validated schedule and
recomputes `schedule_sha256`.

The schedule digest serves as the dataset material identity in
`evidence-pack-v1`. Dataset coordinates explain the declared source; the
digest identifies the exact ordered schedule evaluated. Coordinates alone are
not a material identity.

## Provider-observation contract

Each side emits an `invarlock/runtime-scoring-observation-v1` document binding:

- provider name and artifact-identity digest;
- the canonical schedule digest;
- ordered records with `record_id` and `input_sha256`;
- successful output or scoring facts; and
- `aggregate_source_sha256`, the digest of the complete canonical record
  array.

For side $X \in \{B,S\}$, the verifier requires:

$$
\operatorname{IDs}(\mathcal O_X)
= (r_1,\ldots,r_n),
$$

and:

$$
\forall i,\quad
\operatorname{inputDigest}(\mathcal O_{X,i}) = h_i
\quad\land\quad
\operatorname{status}(\mathcal O_{X,i})=\texttt{ok}.
$$

It also recomputes:

$$
d_{\mathcal O_X}=H\!\left(C_0(\operatorname{records}(\mathcal O_X))\right)
$$

and requires equality with the observation's `aggregate_source_sha256`.

Provider receipts and runtime manifests cross-bind artifact identity,
observation, provider implementation, execution settings, runtime image, and
device facts. The runtime-side report binds the observation digest, schedule,
provider, artifact, and record count. These links prevent a record or sidecar
from being transplanted without invalidating another authenticated binding.

## Pair derivation

For schedule position $i$, the accepted pairing relation is:

$$
r_i=r_{B,i}=r_{S,i}
\quad\land\quad
h_i=h_{B,i}=h_{S,i}
\quad\land\quad
z_{B,i}=z_{S,i}=\texttt{ok}.
$$

Equality is positional and sequence-wide. Matching sets or matching counts are
insufficient. Reordering, omission, duplication, an input-part change, an
error record, or a schedule-digest change fails the transaction.

The resulting `invarlock/paired-records-v1` file stores, for each schedule
record:

- `record_id` and `input_sha256`;
- verifier-derived baseline and subject scores; and
- a digest of each complete source observation record.

Normalized-NLL pairs may also bind a verifier-derived perplexity interpretation
when authenticated tokenizer digests and equal positive per-pair token counts
make the token-weighted value comparable. That interpretation has no acceptance
authority.

Verification derives the entire document again and requires exact equality
with the stored canonical file. It then rebuilds the comparison report from
those derived pairs, including the selected paired interval.

## Worked mismatch examples

Given schedule IDs:

```text
[sample-001, sample-002, sample-003]
```

these observations all fail closed:

| Observation IDs or facts | Reason |
| --- | --- |
| `[sample-001, sample-003, sample-002]` | Same set, wrong order |
| `[sample-001, sample-002]` | Missing scheduled record |
| `[sample-001, sample-002, sample-003, sample-003]` | Duplicate and extra record |
| Correct IDs, changed digest for `sample-002` | Input bytes do not match schedule |
| Correct IDs, `sample-002` has `status: error` | Partial evidence is not scored |
| Correct records, wrong `schedule_sha256` | Observation is bound to another schedule |

The verifier never sorts, intersects, truncates, or drops records to manufacture
a paired subset.

## Run and import equivalence

Run mode asks installed providers to authenticate and score both artifacts.
Import mode accepts complete provider evidence created elsewhere. Both modes
converge on the same pair derivation, canonical report, and
`evidence-pack-v1` publication path.

Import mode is not a relaxed path. Imported paired records must equal the pairs
derived from the imported schedule and provider observations. All provider,
artifact, runtime, order, digest, and score checks still apply.

See the [evaluation request guide](../user-guide/evaluation-request.md) for the
two request shapes and the
[provider contracts](../reference/contracts.md#provider-contracts) for their
versioned documents.

## Replay guarantees and limits

Replay detects:

- baseline or subject artifact identity different from the caller-owned
  expected digest;
- a canonical schedule different from the caller-owned expected digest;
- schedule or input changes;
- baseline/subject order drift;
- provider aggregate values unsupported by record facts;
- artifact, provider, runtime, observation, or report substitution;
- changes to paired scores, decision arithmetic, or the policy-relevant paired
  interval bound; and
- use of policy bytes different from the independently supplied policy.

Replay does not prove that provider facts came from genuine model execution. An
evidence signer able to fabricate a complete, mutually consistent set of measurements can
sign that set. External measured-execution attestation or an independently
controlled rerun is needed for a stronger claim.

## Selection assumptions

Perfect pairing does not make a schedule representative. The assurance
argument assumes that an accountable decision owner:

1. defines the target behavior and sampling frame;
2. fixes the dataset revision, split, records, expected outputs, and policy
   before subject results are visible;
3. records or reviews the schedule digest before evaluation;
4. does not remove difficult records after observing failures; and
5. creates a new evidence pack for any schedule or policy change.

Violation of an assumption need not produce a verifier error. It is a defeater
for broader interpretation of the otherwise valid finite-schedule result.

The canonical loader and replay are implemented in
[behavioral schedule](https://github.com/invarlock/invarlock/blob/main/src/invarlock/core/runtime_provider/behavioral_schedule.py)
and
[evidence pack contract](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evidence_pack_contract.py).

## References

- [RFC 8259: The JavaScript Object Notation Data Interchange
  Format](https://www.rfc-editor.org/rfc/rfc8259), Internet Standard, 2017.
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785),
  2020. InvarLock uses its own versioned canonical profile and does not claim
  RFC 8785 conformance.
- National Institute of Standards and Technology,
  [Secure Hash Standard](https://doi.org/10.6028/NIST.FIPS.180-4),
  2015.
