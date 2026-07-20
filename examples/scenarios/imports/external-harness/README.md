# Import paired results from an evaluation harness

## When to use this example

Use this recipe when an existing evaluation harness already executes the
baseline and subject. The bridge must preserve stable per-record identities,
inputs, expected outputs, model/runtime configuration, and raw results.
InvarLock imports those closed records and recomputes the selected score; it
does not trust a harness aggregate.

## Inputs you bring

- The exact harness configuration and immutable baseline and subject identities.
- One canonical ordered schedule with stable IDs.
- Per-record baseline and subject outputs or scorer inputs.
- Closed side manifests, runtime configuration, reports, observations,
  identities, and signed or authenticated receipts.
- A built-in metric or explicitly authorized scorer-extension descriptor and
  configuration.

## InvarLock transaction

Use execution mode `import`. Both side-result sets must reproduce the same
authenticated schedule and artifact identities. For task-specific scoring,
bind a verifier-replayable scorer extension; its descriptor, version,
configuration schema, configuration digest, and result digest become evidence
inputs.

## What the result establishes

A passing receipt establishes that the imported per-record material and scorer
binding were intact, replayable, paired, and accepted by the selected policy.

## Interpretation boundary

Harness totals, reward columns, and presentation summaries are context rather
than acceptance inputs unless the verifier can reproduce them from the bound
records. The bridge must fail when IDs are missing, duplicated, reordered, or
mapped to a different schedule.

## Run it

Map the harness output into the six closed files per side shown in the
[offline import example](../../../README.md). Place the paired records and
canonical schedule beneath the request root, then run:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem
```

Complete the independent `verify` and `report` handoff described in the
[scenario catalog](../../README.md#common-transaction). A bridge implementation
should include golden fixtures proving stable-ID mapping and verifier score
replay.
