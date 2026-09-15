# Acceptance checklist

!!! abstract "Assurance note"
    **In plain language:** A valid signature is only the start. A decision owner must
    also confirm the independent anchors, decision scope, key authorization,
    and known limitations before relying on the result.

    **Question:** Has a decision owner established every prerequisite needed to rely
    on one signed evidence decision?

    **Decision use:** Use this for recipient-owned acceptance after strict
    verification, before an evidence pack supports a downstream decision.

    **Evidence:** Independently sourced anchors, a verified immutable pack, a
    signed verifier receipt, and the decision-owner-maintained decision record.

Use this checklist for one native exact-match, normalized-NLL or deterministic
extension `invarlock/evidence-pack-v1` decision. For captured comparisons and
bounded judge evidence, use the contract-specific checks below.

This is a decision-owner checklist, not an additional runtime approval workflow.
Apply conditional items only when relevant and record which operational checks
your decision requires. Independent expectations can be maintained by the same
team that evaluates models; their authority comes from approved inputs outside
the submitted evidence. A checked item means those expectations were established,
not merely that the bundle contains a matching assertion.

An unmet verifier requirement or configured policy check prevents acceptance.
Record organizational exceptions outside the pack; do not modify the pack,
suppress a verifier error, or reinterpret a failed policy result.

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
- [ ] When either side uses `llama_cpp`, obtain the normalized-request digest
      from an independently maintained source rather than the submitted pack.
- [ ] Run `invarlock verify` with every required anchor, a distinct verifier
      key and identity, and a receipt path outside the evidence directory.
- [ ] Confirm exit code `0`, `ok: true`, `integrity_ok: true`,
      `policy_verdict: pass`, and evidence-signer authenticity `pinned` in JSON output.
- [ ] Confirm the receipt names the expected manifest digest, policy digest,
      baseline and subject artifact-identity digests, canonical schedule digest,
      runtime digests, evidence-signer fingerprint, the GGUF request digest
      when applicable, verifier identity, and verifier fingerprint.
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
- [ ] For exact match, confirm a current v3 report uses
      `newcombe_hybrid_score_paired_v2`, interval mass `0.95`, and the
      paired-binary-outcomes scope. Historical v2 reports use that same method
      without side-accuracy qualification. For legacy v1 evidence, require
      exact replay with `newcombe_hybrid_score_paired_v1` instead.
- [ ] For normalized NLL, confirm the paired interval uses
      `paired_percentile_bootstrap_sha256_v1`, interval mass `0.95`, `2048`
      replicates, and the authenticated-schedule scope.
- [ ] For a scorer extension, confirm the same paired-resampling method and
      lower-bound `delta_min_pp` rule are used, and that strict replay used the
      explicitly authorized scorer registry.
- [ ] Keep LLM judges outside the deterministic scorer-extension boundary.
      Use the separate bounded judge contract when selecting `judge`; its
      acceptance checks are listed below. A captured `recorded` metric can
      apply policy to approved externally assigned scores, but replay
      checks their aggregation and provenance rather than reconstructing the
      original judgment.
- [ ] Confirm the verdict uses the policy-relevant conservative bound: lower
      for exact-match delta and upper for normalized NLL.
- [ ] If sample qualification is present, confirm its minimum and maximum match
      the independently reviewed policy, the units match the selected metric,
      and the count, width, and combined checks all pass.
- [ ] If exact-match `side_accuracy` is present, confirm its minimum matches the
      independently reviewed `minimum_side_accuracy` and both observed side
      means and the combined check pass.
- [ ] Treat the result as a finite-schedule decision. The paired interval
      describes schedule-composition sensitivity; it does not establish
      population coverage or representativeness.
- [ ] Review runtime, sampling, run-selection, baseline-trust, and execution-
      attestation limitations before approving downstream use.
- [ ] If the result lies on or near a policy boundary, independently rerun it
      or record why one run is sufficient for this decision.

## Captured and judge evidence

For a captured comparison, use `invarlock/trust-inputs-v2` with independently
approved complete baseline and subject run digests, normalized-request digest,
policy bytes and signer fingerprint. Confirm the signed v3 receipt names
`captured_comparison` and that every required metric and slice passes. Read each
row's `scoring_assurance`: recorded scores retain source judgments; recomputed
scores derive from retained facts. Hosted-service descriptors identify declared
configuration and an observation window, not underlying model weights or future
service behavior. See [captured verification](../user-guide/captured-results.md#signed-handoff).

For native, captured or frozen-answer judge evidence, use an independently
maintained `invarlock/judge-measurement-recipient-policy-v1`. Check its exact
run, case-set, plan, measurement, analysis-policy and analysis-result pins, plus
`native_capture_sha256` when present. Confirm `authenticated`, `replayed`,
`verified` and `accepted`, and inspect `decision` separately. A signed judge
receipt is optional for local verification and needed when transporting the
verifier's signed result; its contract differs from native and captured receipts.
See [judge verification](../reference/judge-measurements.md).

A judge policy's `decision_role: required` makes that metric part of the required
conjunction. Reference-label studies and
independent reruns can support a broader reliance decision but are not runtime
prerequisites. Optional per-case references are judge inputs only when the
plan selects `reference_mode: per_case`; they are separate from such studies.

## Present and retain

- [ ] Use `invarlock report evidence/ --html evidence.html --explain` for a
      report; do not treat the rendered file as the signed acceptance
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

The record is review metadata, not part of `invarlock/evidence-pack-v1`. If it requires
cryptographic authentication, sign it through the review system rather than
adding it to the immutable pack.

See the [assurance case](assurance-case.md) for what these checks establish and
[Security practices](../security/best-practices.md) for operational handling.
