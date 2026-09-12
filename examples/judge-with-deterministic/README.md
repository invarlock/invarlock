# Deterministic and judge evidence on the same answers

This offline example publishes two existing evidence formats over exactly the
same synthetic one-case baseline and subject runs. The deterministic component
recomputes exact match; the bounded judge component replays retained illustrative
ratings. There are no model calls. The judge result is deliberately inconclusive
because one independent unit cannot meet its policy.

From a source checkout with InvarLock installed, choose a new output directory:

```bash
python examples/judge-with-deterministic/demo.py --output /tmp/invarlock-composition
invarlock verify /tmp/invarlock-composition/evidence \
  --trust-profile /tmp/invarlock-composition/recipient/composition.json --json
invarlock report /tmp/invarlock-composition/evidence \
  --html /tmp/invarlock-composition/report.html \
  --markdown /tmp/invarlock-composition/report.md \
  --junit /tmp/invarlock-composition/report.xml
```

The demonstration script exits zero when both components verify, including the
expected inconclusive result. The separate `verify` command exits **7** because
recipient acceptance requires both policies to pass. Its JSON distinguishes
successful authentication/replay from acceptance. The report has separate metric
tabs and preserves each component's statistical method.

`verification.json` retains the complete captured signed receipt and judge local
receipt as JSON text within its member entries. The composite result itself is a
local calculation. Reuse it only through fresh `verify_stored_evidence_set_result`
recomputation; an `accepted` field does not establish authority.

The script creates separate evidence-signer and recipient keys for demonstration.
A real recipient must independently review the original runs and policies and
obtain the evidence-signer fingerprint through an authorized channel. Never create
production trust simply by copying a submitted pack's values. Keep the generated
`signer/` and `recipient/` directories outside shared evidence.

See [evidence sets](../../docs/reference/evidence-sets.md) for the closed index,
recipient policy and API. For the preceding generation stage, follow the
[answer capture example](../answer-capture/README.md), then freeze its answers
before repeated judging. Deterministic records remain original cases; judge
repetitions do not multiply their count. The conjunction provides **no joint
confidence guarantee** across the two methods.
