# Deterministic and judge evidence sets

An evidence set combines one captured deterministic comparison and one required
bounded judge comparison over the same complete baseline and subject runs. Both
component policies must pass before a recipient accepts the combined result.

!!! info "Reference"

    - **Surface:** Evidence-set index, recipient policy, verification and reporting
    - **Stability:** Additive closed formats; existing component formats retain their meanings
    - **Use this page when:** Requiring deterministic and repeated-judge checks on the same frozen answers

## Component boundary

Produce each component with its existing `evaluate` request. Captured metrics
must be recomputed by `exact_match`, `normalized_match`, `numeric_tolerance`,
`json_fields`, `json_exact` or `token_f1`; recorded scores are not accepted as
the deterministic component. The judge policy must declare `required`.

Both components must bind identical complete run digests, original case-set
digest and subject artifact. Matching case IDs alone is insufficient. Inputs,
references, metadata and answers must match through those complete bindings.
Judge repetitions do not create additional deterministic records.

An index contains exactly the two named members. For existing child directories:

```python
from pathlib import Path
from invarlock.evidence_sets.contracts import write_evidence_set_index

write_evidence_set_index(
    Path("evidence"), deterministic="deterministic", judge="judge"
)
```

This writes an unsigned `evidence-set.json` transport index with the following
shape. Digest placeholders must be replaced by the helper's actual values:

```json
{
  "format": "invarlock/evidence-set-v1",
  "members": {
    "deterministic": {
      "kind": "captured",
      "path": "deterministic",
      "statement_sha256": "sha256:..."
    },
    "judge": {
      "kind": "judge",
      "path": "judge",
      "statement_sha256": "sha256:..."
    }
  }
}
```

The statement pins hash exact child manifest/envelope bytes. The index supplies
no signer authority or acceptance policy. Nested sets, overlapping member paths,
traversal, symlinks and conflicting format markers are rejected.

## Independent recipient policy

The recipient retains its composition policy and both component trust profiles
outside the entire evidence set. Relative profile paths resolve under the
composition policy's parent. The captured profile keeps its existing independent
policy, anchors, verifier identity and signing key. The judge profile keeps its
existing signer, plan, measurement, result and intended-subject pins.

```json
{
  "format": "invarlock/evidence-set-recipient-policy-v1",
  "scope": "same-answer-component-conjunction-v1",
  "index_sha256": "sha256:...",
  "shared_inputs": {
    "baseline_run_sha256": "sha256:...",
    "subject_run_sha256": "sha256:...",
    "case_set_sha256": "sha256:...",
    "subject_artifact_sha256": "sha256:..."
  },
  "members": {
    "deterministic": {
      "kind": "captured",
      "trust_profile": "captured.json",
      "trust_profile_sha256": "sha256:...",
      "role": "required"
    },
    "judge": {
      "kind": "judge",
      "trust_profile": "judge.json",
      "trust_profile_sha256": "sha256:...",
      "role": "required"
    }
  },
  "decision_rule": "all-required-components-pass",
  "statistical_scope": "component-methods-no-joint-confidence"
}
```

All composition digests use the `sha256:` prefix. Index and trust-profile pins
hash exact file bytes; original-run and case-set pins use their canonical public
contract functions. Approve component policies and shared input identities
independently before approving the transport index. Copying a submitted
index's identities does not establish that approval.

The runnable source example at `examples/judge-with-deterministic/demo.py`
constructs independent input expectations before publishing its synthetic
components. Its generated recipient policy shows every required field.

## Verification and report commands

```bash
invarlock verify evidence --trust-profile recipient/composition.json \
  --receipt verification.json --json
invarlock report evidence --html report.html --markdown report.md --junit report.xml
```

`verify` freshly authenticates and replays each component with its own recipient
profile, checks common input identity, then requires both accepted results.
Explicit component signer or policy flags cannot be combined with composition;
place these controls in the corresponding independent profiles.
`--max-bootstrap-draws` remains the recipient's captured replay work allowance.
No provider calls or scorer extensions execute.

Exit 0 means both required components passed and were accepted. Exit 7 means
both components verified but a required result was adverse or inconclusive.
Exit 4 means composition could not establish authentication, replay or required
input identity. Missing CLI trust inputs or invalid output destinations exit 2.

An optional result retains the complete captured signed receipt and a scoped
signed judge receipt as bounded JSON text inside the member results. Both member
receipts use the captured trust profile's independently loaded verifier identity
and key. The composite result remains local and unsigned; it cannot authorize
itself. All output destinations must be new and outside the evidence set. Reports
preserve both methods and show the shared original cases, component outcomes and
distinct assurance levels; reporting does not perform recipient acceptance.

The corresponding APIs are:

```python
from pathlib import Path
from invarlock.evidence_sets.verification import (
    verify_evidence_set,
    verify_stored_evidence_set_result,
)

result = verify_evidence_set(
    Path("evidence"),
    recipient_policy=Path("recipient/composition.json"),
    receipt=Path("verification.json"),
)
accepted = result.accepted

rechecked = verify_stored_evidence_set_result(
    Path("verification.json"),
    Path("evidence"),
    recipient_policy=Path("recipient/composition.json"),
)
```

Rechecking recomputes the result and requires exact equality. Stored flags or
component receipts are never substituted for current verification.

## Statistical scope

The combined policy is a conjunction of component decisions. It provides **no
joint confidence guarantee**. Captured v1 retains its marginal case-based
intervals. Judge evidence retains its declared independent-unit weighting and
family-adjusted bounds. Neither method's confidence level is promoted to a
confidence claim about the entire set.

No metric can rescue a failed required component. An advisory judge result
cannot be promoted into required acceptance. A common confidence family,
arbitrary member graphs and additional evidence families are outside this
profile. All existing standalone requests, evidence and receipts remain valid.

For the individual contracts, see [captured results](../user-guide/captured-results.md)
and [bounded judge measurements](judge-measurements.md).
