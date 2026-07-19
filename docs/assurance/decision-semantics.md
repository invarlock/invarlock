# Decision semantics

!!! abstract "Assurance note"
    **In plain language:** InvarLock recomputes every score from authenticated
    paired records, derives the metric's paired interval, and applies the policy
    to the conservative bound. When the policy qualifies sample size and
    precision, those checks must pass too. A favorable point value cannot
    override a threshold, count, or width failure.

    **Question:** How are a built-in metric or authorized deterministic scorer
    converted into a reproducible policy verdict?

    **Decision use:** Use these definitions to review thresholds, reproduce
    canonical report arithmetic, and interpret a pass or fail at the boundary.

    **Evidence:** Pinned local source identity, canonical schedule, provider
    observations, verifier-derived paired scores, exact policy bytes,
    independent artifact and schedule anchors, and the metric-specific paired
    interval recorded in the report.

InvarLock makes one decision over one authenticated, ordered, finite schedule.
It reports a point comparison and a verifier-replayed paired interval. Exact
match uses a paired Newcombe 95% effect-size interval. Normalized NLL and an
authorized scorer extension use deterministic paired resampling over the
authenticated schedule.

New evaluations emit `invarlock/comparison-report-v2`. Strict verification
continues to replay `invarlock/comparison-report-v1` with its original
exact-match interval method, so signed legacy evidence keeps its original
meaning.

## Notation

| Symbol | Meaning |
| --- | --- |
| $n$ | Number of scheduled records; $1 \le n \le 10{,}000$ |
| $i$ | Schedule position, $i \in \{1, \ldots, n\}$ |
| $B$, $S$ | Baseline and subject sides |
| $y_i$ | Scheduled expected output for record $i$ |
| $\hat{y}_{X,i}$ | Authenticated output text for side $X$ |
| $L_{X,i}$ | Authenticated sum of target-token log probabilities |
| $u_i$ | Positive UTF-8 byte count of $y_i$ |
| $w_i$ | Positive target token count used only for a derived perplexity interpretation |
| $x_{X,i}$ | Verifier-derived record score for side $X$ |
| $\bar{x}_X$ | Arithmetic mean of side $X$ record scores |
| $\tau_{\Delta}$ | Policy floor `exact_match.delta_min_pp` |
| $\tau_{N}$ | Policy ceiling `normalized_nll_per_utf8_byte.ratio_max` |
| $\tau_{S}$ | Policy floor `scorer_extension.delta_min_pp` |
| $[q_L,q_U]$ | Metric-specific paired interval |
| $n_{\min}$ | Optional policy minimum `minimum_record_count` |
| $w_{\max}$ | Optional policy maximum interval width in percentage points or ratio units |

## Shared preconditions

Replay proceeds only when:

1. the normalized request, source identity, and canonical schedule are bounded
   and digest-authenticated;
2. record IDs are unique and both observations have exactly the schedule IDs
   in schedule order;
3. each observed `input_sha256` matches the corresponding scheduled input;
4. every selected observation record has `status: ok`;
5. provider, artifact, observation, schedule, runtime, report, and request
   bindings are internally consistent;
6. the schedule and both providers declare the selected task and provider
   collection metric;
7. the verifier's independently supplied baseline artifact, subject artifact,
   and canonical schedule digests match the identities bound into the signed
   evidence; and
8. the verifier's policy bytes have the same digest as the policy bound into
   the signed evidence.

Failure of a precondition is a verification error, not a poor metric score.
InvarLock does not impute, discard, reorder, or average around an invalid record.

## Exact-match percentage-point delta

For each side and scheduled record:

$$
x_{X,i} = \mathbf{1}\!\left[\hat{y}_{X,i}=y_i\right].
$$

Equality is literal Unicode string equality. It does not trim whitespace,
case-fold, normalize, parse an answer, or compare token sequences.

The side means and point comparison are:

$$
\bar{x}_X = \frac{1}{n}\sum_{i=1}^{n}x_{X,i},
\qquad
\Delta_{\mathrm{pp}} =
100\left(\bar{x}_S-\bar{x}_B\right).
$$

The report records `comparison.value =` $\Delta_{\mathrm{pp}}$. The policy is:

```json
{
  "resolved_policy": {
    "metrics": {
      "exact_match": {
        "delta_min_pp": -2.0,
        "maximum_interval_width_pp": 10.0,
        "minimum_record_count": 400
      }
    }
  }
}
```

The verdict uses the interval lower bound:

$$
\operatorname{pass}_{\mathrm{EM}}
\iff q_L \ge \tau_{\Delta}
\land n \ge n_{\min}
\land (q_U-q_L) \le w_{\max}.
$$

The boundary is inclusive. `delta_min_pp` is an absolute percentage-point floor,
not a relative percentage and not a floor applied only to the point estimate.
The count and width terms apply only when the policy supplies both coupled
fields; without them, the threshold-only policy remains valid.

For every pair, InvarLock also classifies the two binary outcomes:

- baseline pass and subject fail: a regression;
- baseline fail and subject pass: an improvement;
- both pass; or
- both fail.

If $r$ is the regression count and $g$ is the improvement count, then
$\Delta_{\mathrm{pp}}=100(g-r)/n$. The report records all four counts, the
number of discordant pairs $r+g$, and the exact two-sided McNemar probability
conditional on those discordant pairs. The McNemar result describes evidence
of asymmetry between regressions and improvements; it does not control the
acceptance verdict.

The exact-match interval is a paired Newcombe hybrid-score 95% interval for
the subject-minus-baseline effect. The current v2 report uses the
continuity-corrected method:

```json
{
  "method": "newcombe_hybrid_score_paired_v2",
  "scope": "paired_binary_outcomes",
  "interval_mass": 0.95,
  "lower": -1.8,
  "upper": 0.7
}
```

The policy reads `lower`; the point delta and McNemar probability cannot
override a lower bound below `delta_min_pp`.

For a signed v1 report, strict replay instead requires
`newcombe_hybrid_score_paired_v1` and reconstructs the legacy result exactly.
It does not apply the v2 calculation to old evidence.

## Normalized NLL per UTF-8 byte

For each side and record, the provider performs teacher-forced scoring of the
authenticated expected continuation and supplies finite $L_{X,i}$ and positive
byte count $u_i$. Prompt tokens are excluded from the target loss. The verifier
requires the byte count to equal the UTF-8 length of the authenticated scheduled
target, then derives:

$$
x_{X,i} = -\frac{L_{X,i}}{u_i}.
$$

Every derived value must be finite and nonnegative. The side means use equal
record weight:

$$
\bar{x}_X = \frac{1}{n}\sum_{i=1}^{n}x_{X,i}.
$$

The baseline mean must be positive. The point ratio is:

$$
R_{\mathrm{NLL}} = \frac{\bar{x}_S}{\bar{x}_B}.
$$

The policy is:

```json
{
  "resolved_policy": {
    "metrics": {
      "normalized_nll_per_utf8_byte": {
        "maximum_interval_width_ratio": 0.05,
        "minimum_record_count": 400,
        "ratio_max": 1.05
      }
    }
  }
}
```

The verdict uses the interval upper bound:

$$
\operatorname{pass}_{\mathrm{NLL}}
\iff q_U \le \tau_N
\land n \ge n_{\min}
\land (q_U-q_L) \le w_{\max}.
$$

This statistic is a ratio of arithmetic means of per-record normalized scores.
It is not pooled NLL, a byte-weighted mean, token-weighted mean, or perplexity.
Each selected record contributes one score regardless of target length.
It measures expected-continuation likelihood regression under the authenticated
prompt, target, provider, and runtime; it is not a general model-quality score.

The count and ratio-width terms apply only when their coupled policy fields are
present. Their pass does not establish that the schedule represents a broader
population.

### Worked normalized-NLL example

Suppose two authenticated records yield:

| Record | Baseline $x_{B,i}$ | Subject $x_{S,i}$ |
| --- | ---: | ---: |
| 1 | $-(-5)/10 = 0.5$ | $-(-5.5)/10 = 0.55$ |
| 2 | $-(-20)/20 = 1.0$ | $-(-22)/20 = 1.1$ |

Then $\bar{x}_B=0.75$, $\bar{x}_S=0.825$, and the point ratio is $1.1$.
That point fails a `ratio_max` of `1.05`; even when a point ratio is below the
ceiling, the canonical verdict still fails if its interval upper bound exceeds
`1.05`.

## Derived perplexity interpretation

For a normalized-NLL comparison, the verifier may also derive a token-weighted
perplexity ratio when both sides bind one matching authenticated tokenizer
contract and report the same positive target token count $w_i$ for every pair.
It derives token-normalized record NLL:

$$
x_{X,i} = -\frac{L_{X,i}}{w_i}.
$$

The token-weighted mean NLL and side perplexity are:

$$
\mu_X^{(w)} =
\frac{\sum_{i=1}^{n}w_i x_{X,i}}{\sum_{i=1}^{n}w_i},
\qquad
P_X = \exp\!\left(\mu_X^{(w)}\right).
$$

The derived ratio is:

$$
R_{\mathrm{PPL}} = \frac{P_S}{P_B}
= \exp\!\left(\mu_S^{(w)}-\mu_B^{(w)}\right).
$$

The canonical `derived_measurements.perplexity_ratio` object records the
authenticated likelihood basis, method, tokenizer digest, total target-token
count, both side perplexities, and the ratio. If tokenizer contracts differ,
token counts are unavailable or unequal, or exponentiation is non-finite, the
object records a deterministic unavailable reason instead.

This interpretation is not a selectable metric. It has no policy threshold,
confidence interval, or verdict authority. Acceptance continues to use the
normalized-NLL ratio and its schedule-resampling interval.

## Normalized-NLL paired resampling

The report's uncertainty object is fixed:

```json
{
  "method": "paired_percentile_bootstrap_sha256_v1",
  "scope": "authenticated_schedule",
  "interval_mass": 0.95,
  "replicates": 2048,
  "lower": 0.98,
  "upper": 1.04
}
```

Let $d$ be the 32 bytes represented by the canonical schedule's lowercase
SHA-256. For replicate $r \in \{0,\ldots,2047\}$ and draw
$j \in \{0,\ldots,n-1\}$, InvarLock computes:

$$
h_{r,j} = \operatorname{SHA256}
\left(d\,\|\,\operatorname{u64be}(r)\,\|\,\operatorname{u64be}(j)\right),
$$

and selects index:

$$
k_{r,j} = \operatorname{u64be}\!\left(h_{r,j}[0:8]\right) \bmod n.
$$

The baseline and subject normalized-NLL scores at $k_{r,j}$ are drawn together.
InvarLock recomputes the complete ratio for each replicate, sorts the 2,048 values, and
uses linearly interpolated percentiles at $0.025$ and $0.975$. When $n=1$, the
interval is the point value.

This construction provides:

- deterministic replay without a platform PRNG;
- pairing preservation under every draw;
- a versioned algorithm and fixed replicate count; and
- a conservative bound that directly controls the policy verdict.

It does not provide random-sample provenance. The SHA-256-derived index stream
makes replay deterministic; it does not make the original selected schedule a
probability sample.

## Authorized deterministic scorer extension

A request may select one closed `scorer_extension` binding instead of a
built-in metric. For every side and record, the provider authenticates exactly
the scheduled expected output, output text, and output digest. The explicitly
authorized scorer replays those facts and returns:

$$
0 \le x_{X,i} \le 1,
$$

where larger values are always better. The scorer does not own aggregation or
policy arithmetic. The core computes:

$$
\bar{x}_X = \frac{1}{n}\sum_{i=1}^{n}x_{X,i},
\qquad
\Delta_{S,\mathrm{pp}} =
100\left(\bar{x}_S-\bar{x}_B\right).
$$

The independent policy pins the exact scorer identity and configuration:

```json
{
  "resolved_policy": {
    "metrics": {
      "scorer_extension": {
        "scorer_id": "example.token_f1",
        "scorer_version": "1.0.0",
        "descriptor_sha256": "5555555555555555555555555555555555555555555555555555555555555555",
        "configuration_sha256": "6666666666666666666666666666666666666666666666666666666666666666",
        "delta_min_pp": -2.0,
        "maximum_interval_width_pp": 10.0,
        "minimum_record_count": 400
      }
    }
  }
}
```

The scorer runs twice during replay and both canonical results must match. The
core then applies the fixed 2,048-replicate paired schedule-resampling method
defined above to the paired unit-interval values. The verdict is:

$$
\operatorname{pass}_{S}
\iff q_L \ge \tau_S
\land n \ge n_{\min}
\land (q_U-q_L) \le w_{\max}.
$$

Potential separately implemented scorers include deterministic token F1,
structured-field extraction, and VQA answer normalization. The scorer-extension v1 contract
does not admit SQL or code execution, model-based semantic similarity, network
services, human judgment, external models, or LLM judges. Judge outputs may be
authenticated observations, but observations do not enter acceptance.

For every metric, the sample controls are optional but indivisible. Exact
match and scorer extensions use `maximum_interval_width_pp`, whose value must
be positive and no greater than 200. Normalized NLL uses positive
`maximum_interval_width_ratio`. `minimum_record_count` is an integer from 1
through 10,000. The canonical v2 report records the minimum, maximum, observed
values, units, individual results, and combined `sample_qualification.passed`.

## Verifier-owned replay

The evidence signer cannot supply an accepted aggregate directly. Verification:

- reconstructs the canonical schedule and record order;
- derives exact-match or normalized-NLL record scores from authenticated facts,
  or requires an explicitly authorized scorer and replays it twice;
- derives paired exact-match counts, exact McNemar probability, and the
  Newcombe interval, or the normalized-NLL or scorer-extension
  schedule-resampling interval;
- derives a perplexity interpretation only when tokenizer and token-count facts
  are comparable;
- reconstructs canonical paired records;
- recomputes side means, point comparison, the selected interval, threshold
  application, optional count/width qualification, and verdict; and
- requires canonical equality with the stored report.

The implementation is in
[`build_comparison_report`](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evidence_pack_contract.py),
and independent replay is in
[`verify_comparison_evidence`](https://github.com/invarlock/invarlock/blob/main/src/invarlock/evidence_pack_verification.py).

## Finite-schedule interpretation

A pass means that the policy's conservative bound cleared its threshold and
any configured count and width requirements passed for the authenticated
records and runtime configuration. It does not establish:

- equality, equivalence, or non-inferiority in a broader population;
- representativeness of the selected prompts or targets;
- statistical power for a separately defined hypothesis;
- safety, robustness, fairness, or general model quality;
- execution attestation from a runtime-image digest; or
- immunity to repeated tuning or best-run selection bias.

Where population inference is necessary, define and precommit a sampling and
statistical protocol outside InvarLock and retain that evidence separately. Do
not extend either paired interval beyond the authenticated schedule without
those assumptions.

## References

- [RFC 3629: UTF-8, a transformation format of ISO 10646](https://www.rfc-editor.org/rfc/rfc3629), Internet Standard, 2003.
- McNemar, Q., [Note on the sampling error of the difference between correlated
  proportions or percentages](https://doi.org/10.1007/BF02295996),
  *Psychometrika* 12, 153–157, 1947.
- Newcombe, R. G., [Improved confidence intervals for the difference between
  binomial proportions based on paired data](https://pubmed.ncbi.nlm.nih.gov/9839354/),
  *Statistics in Medicine* 17(22), 2635–2650, 1998.
- National Institute of Standards and Technology,
  [Artificial Intelligence Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1),
  2023.
