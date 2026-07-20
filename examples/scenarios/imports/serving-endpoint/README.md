# Evaluate a serving endpoint change

## When to use this example

Use this recipe when a hosted inference service, model server, or gateway
changes model artifacts, runtime versions, generation settings, routing, or
deployment configuration. The endpoint client captures paired records for an
authenticated, bounded import transaction.

## Inputs you bring

- Stable prompts and expected outputs with caller-assigned record IDs.
- Baseline and subject endpoint configuration identities, including model,
  runtime, decoding settings, and observation window.
- Per-record outputs mapped to the same canonical schedule.
- Closed side files and receipts created by a provider-ABI bridge.
- A policy appropriate to the task and independently provisioned trust inputs.

Capture endpoint responses in a controlled evaluation job. Secrets, bearer
tokens, customer inputs, mutable endpoint URLs, and unrestricted response
metadata must not enter the evidence pack.

## InvarLock transaction

Use authenticated import mode. The endpoint bridge assigns the runtime and
artifact identities and emits the closed side results. Exact match works for
closed-answer tasks; other task outcomes require an authenticated,
verifier-replayable scorer extension.

## What the result establishes

A passing receipt establishes that the recorded candidate endpoint
configuration satisfied the paired policy during the authenticated observation
window.

## Interpretation boundary

The evidence does not predict future service availability or behavior after a
configuration changes. Latency, throughput, geographic routing, and service
health belong in separately authenticated observations or service controls.

## Run it

Normalize the endpoint job into the import layout documented by the
[offline example](../../../README.md). Review the result set for secrets and
unstable metadata, then execute without giving InvarLock network credentials:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem
```

Complete the independent `verify` and `report` handoff described in the
[scenario catalog](../../README.md#common-transaction).
