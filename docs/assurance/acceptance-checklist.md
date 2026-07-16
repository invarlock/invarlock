# Acceptance checklist

!!! abstract "Assurance note"
    **In plain language:** A valid signature is only the start. A decision owner must
    also confirm the independent anchors, decision scope, key authorization,
    and known limitations before relying on the result.

    **Question:** Has a decision owner established every prerequisite needed to rely
    on one signed evidence decision?

    **Decision use:** Use this as the final human acceptance gate after strict
    verification, before an evidence pack supports a downstream decision.

    **Evidence:** Independently sourced anchors, a verified immutable pack, a
    signed verifier receipt, and the decision-owner-maintained decision record.

Use this checklist for one `evidence-pack-v1` decision. A checked item means
the decision owner established it independently; it must not mean merely that the
bundle contains a matching assertion.

**Hard stop:** one unchecked required item means the evidence has not completed
this acceptance gate. Record an exception outside the pack; do not modify the
pack, suppress a verifier error, or reinterpret a failed policy result.

## Before evaluation

- [ ] The baseline and subject are the artifacts intended for comparison.
- [ ] The schedule, expected outputs, built-in metric or scorer binding, and
      policy were fixed before subject results were inspected.
- [ ] The schedule represents the behavior relevant to this decision; known
      sampling limits are recorded.
- [ ] Dataset source, split, immutable revision, sampling frame, exclusions,
      and expected-output review are documented.
- [ ] Artifact revisions, provider resources, and runtime images are immutable
      and digest-pinned.
- [ ] Network, remote-code, plugin, device, and decoding settings were explicitly approved.
- [ ] If a scorer extension is selected, its implementation, ID, version,
      descriptor digest, and configuration digest were independently reviewed
      and authorized for both evaluation and verification.
- [ ] The evidence-signing key is authorized for evidence creation and is not the
      verifier key.
- [ ] Retry, stopping, and run-selection rules were fixed; failed or superseded
      attempts will remain reviewable.

## Verify

- [ ] Obtain the policy bytes, baseline and subject artifact-identity digests,
      canonical schedule digest, baseline and subject runtime digests, and
      expected evidence-signer fingerprint from independently maintained sources.
- [ ] Run `invarlock verify` with every required anchor, a distinct verifier
      key and identity, and a receipt path outside the evidence directory.
- [ ] Confirm exit code `0`, `ok: true`, `integrity_ok: true`,
      `policy_verdict: pass`, and evidence-signer authenticity `pinned` in JSON output.
- [ ] Confirm the receipt names the expected manifest digest, policy digest,
      baseline and subject artifact-identity digests, canonical schedule digest,
      runtime digests, evidence-signer fingerprint, verifier identity, and
      verifier fingerprint.
- [ ] Confirm evidence signer and verifier fingerprints are currently authorized and
      not subject to an unresolved compromise or revocation event.
- [ ] Preserve the immutable evidence directory and signed receipt together.

## Interpret

- [ ] Confirm the metric and threshold express the intended acceptance rule.
- [ ] For a scorer extension, confirm every replayed score is a
      higher-is-better value in `[0, 1]`, the core arithmetic mean and
      subject-minus-baseline percentage-point delta are used, and the policy
      pins the exact scorer and configuration digests.
- [ ] For exact match, interpret the result as an absolute percentage-point
      delta with exact string equality; review paired regressions,
      improvements, the exact McNemar probability, and the Newcombe interval.
- [ ] For normalized NLL, interpret the result as a ratio of arithmetic means
      of teacher-forced expected-continuation NLL-per-UTF-8-byte values, not as
      a general model-quality score.
- [ ] If a perplexity ratio is displayed, confirm it is verifier-derived from
      comparable tokenizer and token-count facts and treat it as interpretation
      without policy authority.
- [ ] For exact match, confirm the paired interval uses
      `newcombe_hybrid_score_paired_v1`, interval mass `0.95`, and the
      paired-binary-outcomes scope.
- [ ] For normalized NLL, confirm the paired interval uses
      `paired_percentile_bootstrap_sha256_v1`, interval mass `0.95`, `2048`
      replicates, and the authenticated-schedule scope.
- [ ] For a scorer extension, confirm the same paired-resampling method and
      lower-bound `delta_min_pp` rule are used, and that strict replay used the
      explicitly authorized scorer registry.
- [ ] Treat LLM judges and any network, human, external-model, SQL-execution,
      code-execution, or model-similarity result as observation-only evidence,
      not an acceptance scorer.
- [ ] Confirm the verdict uses the policy-relevant conservative bound: lower
      for exact-match delta and upper for normalized NLL.
- [ ] Treat the result as a finite-schedule decision. The paired interval
      describes schedule-composition sensitivity; it does not establish
      population coverage or representativeness.
- [ ] Review runtime, sampling, run-selection, baseline-trust, and execution-
      attestation limitations before approving downstream use.
- [ ] If the result lies on or near a policy boundary, independently rerun it
      or record why one run is sufficient for this decision.

## Present and retain

- [ ] Use `invarlock report evidence/ --html evidence.html --explain` for a
      human view; do not treat the rendered file as the signed acceptance
      record.
- [ ] Record any external attestation, independent rerun, exception, or decision owner
      decision beside the bundle without modifying it.
- [ ] Re-evaluate when any artifact, schedule, policy, provider, runtime, or
      execution setting changes.

## Minimal decision record

Keep this information beside the pack and receipt:

```text
evidence manifest digest:
signed receipt digest:
comparison ID:
policy digest and approval reference:
baseline and subject artifact identities:
canonical schedule digest and approval reference:
baseline and subject runtime digests:
evidence-signer fingerprint and authorization reference:
verifier identity/fingerprint and authorization reference:
built-in metric or scorer identity, observed comparison value, paired interval,
and threshold:
schedule scope and known limitations:
external rerun or attestation references:
decision, decision owner, and decision time:
exceptions and expiry/re-evaluation trigger:
```

The record is review metadata, not part of `evidence-pack-v1`. If it requires
cryptographic authentication, sign it through the review system rather than
adding it to the immutable pack.

See the [assurance case](assurance-case.md) for what these checks establish and
[Security practices](../security/best-practices.md) for operational handling.
