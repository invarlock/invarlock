# Deterministic and judge evidence on the same answers

This offline example publishes two existing evidence formats over exactly the
same synthetic one-case baseline and subject runs. The deterministic component
recomputes exact match; the bounded judge component replays retained illustrative
ratings. There are no model calls. The judge result is deliberately inconclusive
because one independent unit cannot meet its policy.

Use this example when a release decision requires both a deterministic check and
a rubric-based check on the same frozen answers. It keeps each method's evidence
and acceptance rule visible while requiring both to pass.

You need Python 3.12 or newer, the core InvarLock package and example script from
the same source revision, and a writable directory. No optional collector,
provider credentials, GPU or container engine is needed.

## Build and inspect the example

Run from the source checkout with InvarLock installed. Choose an output directory
that does not exist:

```bash
python examples/judge-with-deterministic/demo.py --output composition-demo
```

The script creates signed component evidence, demonstration trust inputs and
`verification.json`. Run recipient verification explicitly to observe the
expected policy rejection:

```bash
invarlock verify composition-demo/evidence \
  --trust-profile composition-demo/recipient/composition.json --json
```

That command intentionally exits 7. Run reporting separately so a shell that
stops on nonzero exits does not skip the report:

```bash
invarlock report composition-demo/evidence \
  --html composition-demo/report.html \
  --markdown composition-demo/report.md \
  --junit composition-demo/report.xml
```

The demonstration script exits zero when both components verify, including the
expected inconclusive result. The separate `verify` command exits **7** because
recipient acceptance requires both policies to pass. Its JSON distinguishes
successful authentication/replay from acceptance. The report has separate metric
tabs and preserves each component's statistical method.

## What the result proves

`verification.json` retains the complete captured signed receipt and judge local
receipt as JSON text within its member entries. The composite result itself is a
local calculation. Reuse it only through fresh `verify_stored_evidence_set_result`
recomputation; an `accepted` field does not establish authority.

Successful replay means the submitted component evidence is authentic and its
results can be reproduced. Acceptance additionally requires every required
component to meet its own policy. The one-case judge result prevents acceptance
here even when all signatures and deterministic calculations are valid.

The script creates separate evidence-signer and recipient keys for demonstration.
A real recipient must independently review the original runs and policies and
obtain the evidence-signer fingerprint through an authorized channel. Never create
production trust simply by copying a submitted pack's values. Keep the generated
`signer/` and `recipient/` directories outside shared evidence.

Keep the component evidence unchanged. To rerun the script, choose a new output
directory; to rerender existing evidence, choose new report paths. This synthetic
fixture demonstrates composition and rejection handling, not model quality or
production authorization.

See [evidence sets](../../docs/reference/evidence-sets.md) for the closed index,
recipient policy and API. For the preceding generation stage, follow the
[answer capture example](../answer-capture/README.md), then freeze its answers
before repeated judging. Deterministic records remain original cases; judge
repetitions do not multiply their count. The conjunction provides **no joint
confidence guarantee** across the two methods.
