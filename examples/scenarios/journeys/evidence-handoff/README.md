# Hand evidence to an independent verifier

## When to use this example

Use this runnable journey when an evaluation system sends an immutable evidence
pack to a separate release reviewer. It demonstrates the essential trust
boundary: evidence carries signed facts, while expected artifact, schedule,
runtime, policy, and signer identities arrive independently.

## Inputs you bring

The checked-in demonstration generates disposable evidence and verifier keys,
accepted and rejected comparisons, independent trust inputs, signed receipts,
and a deliberately altered copy of the accepted pack. A real handoff instead
uses organization-controlled keys and trust distribution.

## InvarLock transaction

The evaluation workspace signs and publishes evidence. Only immutable evidence
moves to the verifier workspace. The verifier accepts one policy pass, records
one authentic policy failure, and rejects the altered copy before rendering the
accepted report.

## What the result establishes

The journey proves observable CLI behavior for independent trust anchors,
signed receipt creation, policy-failure handling, report rendering, and
tamper rejection. It does not replay a static success summary.

## Interpretation boundary

The bundled records are synthetic and explain the transaction rather than
measure a real model. Real-model conclusions belong only to evidence created
from authenticated model runs or imports on an appropriate schedule.

## Run it

From the repository root, execute the maintained end-to-end target:

```console
make trust-boundary-demo
```

Inspect the accepted evidence, independent signed receipt, HTML report,
policy-failure receipt, and rejected altered copy beneath the disposable
`examples/artifacts/trust-boundary-demo` workspace. The runner recreates that
ignored workspace for each Make invocation.
