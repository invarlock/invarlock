# InvarLock comparison report

**Recorded policy result: Policy not met**

The observed loss was within the allowance, but a loss greater than 2 percentage points could not be ruled out. The subject matched the expected answer in 93 of 128 cases, compared with 94 for the baseline. Accuracy changed from 73.44% to 72.66%, a change of -0.7812 pp.

## What was compared


### Artifact: Differs

- **Baseline:** file-sha256:9c35a363920c201c1eeffa8354b6bbd527131ebe9af98603b0836ba1d6b46e54
- **Subject:** file-sha256:ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c

### Model ID: Differs

- **Baseline:** gguf-sha256-9c35a363920c201c1eeffa8354b6bbd527131ebe9af98603b0836ba1d6b46e54.gguf
- **Subject:** gguf-sha256-ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c.gguf

### Matching displayed fields

Matching previews do not establish equality of the complete retained fields.

- **Provider:** llama_cpp

### Additional fields

- **Workflow:** Runtime execution
- **Task:** text_causal
- **Paired records:** 128
- **Schedule digest:** sha256:1237faa42048f2d5c21bd9c7d7e72c25059747a181dd837de56fcd399dd82881
- **Dataset:** lambada-mistral-7b-modelkit-128
- **Dataset split:** integration-subset
- **Dataset source SHA-256:** 5b8a38c5f60b52df4f6562261f3c1b01bf497ed2d92f9244ffa0d63a59324731
- **Dataset source format:** jsonl
- **Selected records:** 128
- **Selection limit:** Not specified

### Recorded changes

- Baseline and candidate have different authenticated artifact digests. These digests alone do not establish how the candidate was produced.

## What was checked

- **Bundle integrity:** Inventory, checksums and embedded evidence signature verified.
- **Recorded policy result:** Read from the authenticated canonical report.
- **Independent recipient acceptance:** Not performed by report. An embedded signer is not a recipient-owned trust anchor.

## Exact-match accuracy: All paired records

**Policy not met**. The observed loss was within the allowance, but a loss greater than 2 percentage points could not be ruled out.

| Baseline | Subject | Change | Observed pairs |
| --- | --- | --- | --- |
| 73.44% | 72.66% | -0.7812 pp | 128 |

Baseline: 94 of 128 matched.

Subject: 93 of 128 matched.

Paired 95% confidence interval &#40;Newcombe hybrid score&#41;: [-4.103, 2.497] pp.

### How to read this comparison

**Change:** The marker shows the observed change between subject and baseline. A change in percentage points &#40;pp&#41; is the subject percentage minus the baseline percentage.

**Interval:** The bar shows uncertainty around that change. Its left endpoint is the lower bound; its right endpoint is the upper bound. The interval level does not tell you how often model runs would pass or fail the policy.

**Policy rule:** The paired lower bound &#40;left endpoint&#41; must be at least the policy minimum. The whole interval must lie on the allowed side of the threshold for this bound check to pass.

**No change:** The dashed line marks no change. An interval crossing it includes changes in either direction. A worse observed score describes this run; policy rejection alone does not prove worse performance beyond these cases.

### How this interval was calculated

Baseline and subject are paired on the same 128 cases. Newcombe hybrid score uses their paired match outcomes to form a nominal 95% confidence interval for the accuracy change.

Both matched: 92; baseline only: 2; subject only: 1; neither matched: 33.

Change is subject accuracy minus baseline accuracy in percentage points, not relative percent change. The 95% confidence level describes the interval method, not the share of correct answers.

This exact-match method uses a fixed 95% confidence level; the allowed-loss threshold is a separate policy choice. Under suitable sampling and independent-pair assumptions, the method aims for intervals to cover the underlying accuracy difference in about 95% of repeated studies. This is not the probability that the subject exceeds the allowed loss, and these cases are not automatically representative of production.

### Decision checks

| Check | Observed | Required | Result |
| --- | --- | --- | --- |
| Paired lower bound | -4.10336 pp | &gt;= -2 pp | Not met |
| Record count | 128 | &gt;= 128 | Passed |
| Interval width | 6.60016 pp | &lt;= 20 pp | Passed |

- Higher scores are better; the change is candidate minus baseline in percentage points.
- The policy tests the interval bound, not just the observed change.
- This policy does not specify an absolute accuracy floor.
- Authenticated observations are supplementary; the paired metric and policy remain the complete acceptance calculation.

## Next steps

1. Review the decision checks and their requirements.
2. Run invarlock verify with your independently supplied trust profile and receipt destination to create the signed acceptance or rejection receipt.
3. Keep this report alongside the original immutable evidence and the separate verification receipt.

## Scope and limitations

- This report summarizes the evidence bundle. Rendering checks its integrity and embedded signature; independent acceptance requires verification with your own trust inputs.
- Results apply to the recorded cases, metric and policy. A pass does not establish general model quality, safety or representative production performance.
- The interval describes paired binary outcomes under its stated method. Population interpretation requires an appropriate sampling design.

## Evidence identities

- **Comparison:** comparison-ce6a526bd381400914497f6753311853
- **Metric:** exact_match
- **Policy:** sha256:0146dae121af096268aaf0ee7a38979621ebc7694f89c86292b8ace09dcc3a97
- **Evidence signer:** sha256:5efd93380192003afc02362e5cc244b1108fb24f4c227637c2c626929e34b8d1
