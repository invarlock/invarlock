# SPDX 3.0.1 AI observation

This CPU-only example preserves one SPDX 3.0.1 JSON-LD AI document as an
authenticated InvarLock observation. It demonstrates a narrow interoperability
boundary; it does not add an SPDX artifact type, policy input, verdict rule, or
core contract.

Run the deterministic fixture check from the repository root:

```bash
make example-spdx-ai-observation
```

The example-owned mapper:

1. reads the canonical SPDX bytes and rejects duplicate keys, non-finite JSON,
   reformatting, a final line feed, and an unbounded graph;
2. checks a documented subset of the SPDX 3.0.1 AI shape, including the pinned
   context, one rooted `ai_AIPackage`, one SHA-256 integrity method, a supplier,
   and declared and concluded license relationships;
3. matches the package SHA-256 to the content digest in a canonical InvarLock
   GGUF artifact identity; and
4. embeds the parsed document, its exact source-byte digest, field provenance,
   validation status, and cross-binding result in
   [`observation-payload.json`](observation-payload.json).

An evaluation request can attach those canonical wrapper bytes without any
special schema support:

```yaml
observations:
  - id: subject-spdx-ai
    kind: spdx.ai
    scope: subject
    path: observations/subject-spdx-ai.json
```

`invarlock evaluate` then wraps the payload in the existing
`invarlock/evidence-observation-v1` envelope. The envelope binds the observation
to the comparison, schedule, policy, and both artifact identities, and fixes
its authority to `observation`. Changing or transplanting that envelope is an
InvarLock integrity failure. It is distinct from the SPDX validation result
recorded inside the payload. Core treats the wrapper fields as opaque
observation content; it does not perform SPDX-aware semantic validation.

## Validation boundary

The committed wrapper records the example's limited deterministic checks as
`passed`. Official SPDX JSON Schema validation, OWL/SHACL semantic validation,
and full profile conformance are separately recorded as `not_evaluated`.
Passing this example is therefore not an SPDX conformance claim.

SPDX 3.0.1 says a conformant JSON-LD document requires both official JSON
Schema validation and semantic validation against the ontology with SHACL. Its
AI profile also requires exactly one declared-license and one concluded-license
relationship for each AI package. The fixture follows those visible
requirements, but deliberately does not convert its subset checks into a full
validation claim.

The exact specification source is pinned to SPDX 3.0.1 commit
`61a649da8ca27924ac1ca8d2a061cb228839b24c`. The source document's compact
canonical JSON has no final line feed; the InvarLock observation payload uses
InvarLock's canonical JSON encoding with a final line feed. The wrapper records
the source-byte digest before parsing so those two canonicalization domains are
not conflated.

## Trust boundary

The fixture does not include model bytes. Its cross-binding confirms agreement
between two declarations: the SPDX package hash and the supplied canonical
InvarLock GGUF identity. The mapper must receive the exact subject identity used
by the transaction. Recipients can compare
`artifact_cross_binding.invarlock_artifact_identity_digest` with
`bindings.artifact_digests.subject`; the example tests assert that equality. In
a signed transaction, runtime evidence authenticates the identity against the
actual artifact. The SPDX document remains supporting context and never
participates in the paired statistics or policy verdict.

References:

- [SPDX 3.0.1 serialization requirements](https://spdx.github.io/spdx-spec/v3.0.1/serializations/)
- [SPDX 3.0.1 AI profile](https://spdx.github.io/spdx-spec/v3.0.1/model/AI/AI/)
- [SPDX 3.0.1 AI package](https://spdx.github.io/spdx-spec/v3.0.1/model/AI/Classes/AIPackage/)
