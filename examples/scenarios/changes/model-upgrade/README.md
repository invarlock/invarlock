# Evaluate a model upgrade

## When to use this example

Use this recipe when replacing a deployed checkpoint with another model,
revision, or size. Unlike a transformation-specific recipe, the two artifacts
may come from different releases or organizations. The release question must
still be stated through one paired task schedule and policy.

## Inputs you bring

- The immutable currently deployed checkpoint and proposed replacement.
- A common task dataset with stable IDs and expected outputs.
- A policy covering minimum records, paired effect, confidence interval, and
  acceptable regressions.
- Runtime images able to execute each side under authenticated settings.
- Independent artifact, schedule, runtime, policy, and signer anchors.

## InvarLock transaction

For closed-answer tasks, use `exact_match`; the verifier reports baseline-pass
to subject-fail regressions, improvements, paired effect, confidence interval,
and McNemar's exact test. Use normalized NLL for a comparable intrinsic
likelihood question, not as a general model ranking.

## What the result establishes

A passing receipt establishes that the exact replacement model satisfied the
selected paired release policy relative to the exact deployed baseline.

## Interpretation boundary

A model upgrade can improve some records and regress others. Inspect the paired
counts and interval rather than relying only on mean accuracy. The evidence is
specific to the selected task, schedule, runtimes, and policy.

## Run it

Author a run request with the selected metric and per-side provider settings.
If both models use the built-in Hugging Face provider, one runtime image may be
shared; otherwise provide digest-pinned per-side images:

```console
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --baseline-runtime-image "$BASELINE_IMAGE" \
  --baseline-runtime-image-digest "$BASELINE_DIGEST" \
  --subject-runtime-image "$SUBJECT_IMAGE" \
  --subject-runtime-image-digest "$SUBJECT_DIGEST" --preflight
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --baseline-runtime-image "$BASELINE_IMAGE" \
  --baseline-runtime-image-digest "$BASELINE_DIGEST" \
  --subject-runtime-image "$SUBJECT_IMAGE" \
  --subject-runtime-image-digest "$SUBJECT_DIGEST"
```

Complete the independent `verify` and `report` handoff described in the
[scenario catalog](../../README.md#common-transaction).
