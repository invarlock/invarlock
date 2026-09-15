# InvarLock captured comparison report

**Recorded policy result: Policy satisfied**

Recorded comparison of baseline and candidate results.

## What was compared

- **Baseline:** Observed model: mistralai/Mistral-7B-v0.1 · Deployment: reference-baseline
- **Subject:** Observed model: mistralai/Mistral-7B-Instruct-v0.1 · Deployment: reference-subject
- **Evaluation mode:** Captured evaluator outputs
- **Baseline provider:** local-mistral-reference
- **Baseline service:** mistral-requalification
- **Baseline deployment:** reference-baseline
- **Baseline requested model:** mistral-requalification
- **Baseline observed model:** mistralai/Mistral-7B-v0.1
- **Baseline exposed revision:** Not exposed
- **Baseline observation window:** 2026-09-13T10:22:21.402677Z → 2026-09-13T10:29:35.627560Z
- **Baseline service provenance:** Captured service declaration. Model weights are not identified.
- **Baseline service harness:** invarlock-http-capture 1
- **Baseline records:** 400
- **Subject provider:** local-mistral-reference
- **Subject service:** mistral-requalification
- **Subject deployment:** reference-subject
- **Subject requested model:** mistral-requalification
- **Subject observed model:** mistralai/Mistral-7B-Instruct-v0.1
- **Subject exposed revision:** Not exposed
- **Subject observation window:** 2026-09-13T10:32:18.931698Z → 2026-09-13T10:38:54.046204Z
- **Subject service provenance:** Captured service declaration. Model weights are not identified.
- **Subject service harness:** invarlock-http-capture 1
- **Subject records:** 400
- **Measurement scope:** Verification checks retained observations; it does not remeasure the service.

### Recorded changes

- Declared service fields differ: deployment, observed model. This comparison does not identify the cause of an observed performance change or establish identical hidden weights.
- Prompt comparison unavailable: complete, uniquely paired effective messages were not recorded.

## What was checked

- **Pack format:** invarlock/evidence-pack-v2
- **Authentication:** Signed manifest verified.
- **Replay and scoring:** Not performed by report.
- **Independent acceptance:** Not performed by report.

## final_word_accuracy — overall

**Policy satisfied**. All configured checks passed.

| Baseline | Candidate | Change | Observed pairs |
| --- | --- | --- | --- |
| 5% | 6% | +1 pp | 400 |

Paired 95% confidence interval: [-1.444, 3.526] pp.

| Check | Observed | Required | Result |
| --- | --- | --- | --- |
| Complete paired results | 400 of 400 | All included pairs | Passed |
| Included pair count | 400 | &gt;= 400 | Passed |
| Allowed change | -1.444 pp | &gt;= -2 pp | Passed |
| Interval width | 4.97 pp | &lt;= 10 pp | Passed |

- Higher values are better.
- Recorded scoring basis: expected and output values, scored when the comparison was created.
- 400 usable pairs; 0 missing results; 400 included pairs. Counts in overlapping slices must not be added together.
- Scoring and replay were not performed by report.

## Next steps

1. Review the stored comparison and its policy.
2. Use independent verification before treating this evidence as accepted.

## Scope and limitations

- Scoring and replay were not performed.
- Model and prompt context is evaluator-recorded provenance, not independent execution or model-identity attestation.

## Evidence identities

- **Manifest:** sha256:701aa0c705c6d024745e45f8fc1f6e1a13a54de7dff4e6fdfcb69e53744ded5a
- **Comparison:** sha256:6ee0438185d0ca7cd61d18c064e8937ca806d4956f6680bd373cfb42e5cac7f2
- **Evidence signer:** sha256:624a7b6a0589618bb7884f741f8841c3214097a8456519f1653bab63d1adb50c
- **Baseline run:** hosted-baseline
- **Baseline run digest:** sha256:4bb9fc6111f6448a9af2d57b512c7ec47dade8676a8e49319eccc3a02680a1ef
- **Baseline service identity:** sha256:597e0dffc557da30f8de6a91ccb1608ffcd3170ee717fe9e7bd506d54d451ef2
- **Baseline evaluator:** invarlock-http-capture 1
- **Baseline service configuration:** sha256:4fe448370c813dfd8f15050afcffea58197f2b920c5c12d27f13814a575d6f68
- **Baseline harness source:** sha256:c411e2eb3997a9bea1fb0e085a54078e94396e76af8df32f070e9595b2ff2b9a
- **Subject run:** hosted-subject
- **Subject run digest:** sha256:bab9cb53705840bd91ee67c4fbdbe9d599967ac9dca9d09c18a6820660ef7679
- **Subject service identity:** sha256:364e1e3d3b7e5895e75eb5b9fa2e4be40b82ae45b8066eead44fe0181afeba49
- **Subject evaluator:** invarlock-http-capture 1
- **Subject service configuration:** sha256:4fe448370c813dfd8f15050afcffea58197f2b920c5c12d27f13814a575d6f68
- **Subject harness source:** sha256:c411e2eb3997a9bea1fb0e085a54078e94396e76af8df32f070e9595b2ff2b9a
