# Independent review of the Python comparison reference

The inspected benchmark outputs have coherent arithmetic, policy decisions,
signatures, receipts and reports under the checks below. This is stronger than
Python/Rust agreement, but does not establish universal verifier correctness or
real-world model quality. Broader numeric inputs expose precision and range
limitations, plus one Python exception-normalization gap. No production semantics
or frozen outputs were changed by this review.

The objective was to assess the correctness of the Python reference used in the
Rust comparison. The fixed reference is
`ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3`; the retained comparison identifies Rust
`673d492dde9d7b320c6f082a5ca2a8191ab37f66`. Installed Python arithmetic and
verification/reporting modules were checked against the fixed source. The
retained Python wheel SHA-256 is
`f3b5c8f2dcc5c230a3963fd6701d68562494ec2c44c97d8e0550f7c3799d5bbb`;
the measured fixture manifest SHA-256 is
`7c50eb8f141073e2b7163a4af24771d3562bb342c27b8d3d58ce7a4e98468821`.
These identify inputs; the observations below supply the correctness evidence.

At the 27 October 2026 follow-up, `staging/next` had advanced to
`fb20493918d829ec9ebabe8c477af1f7a5da43fe`. Of the 188 previously tracked package
files, changes were confined to the version module, CLI, engine exports and report
presentation; numerical and verification modules were unchanged. Separate public
report calls against both source revisions produced identical HTML, Markdown and
JUnit bytes for all eight retained workloads (24 files per revision). This was a
source-path compatibility check, not an installed-wheel performance comparison.
The new client-handoff API and its additional files are outside this experiment.
The performance baseline remains the frozen wheel identified above.

## Actual fixture arithmetic

All five captured NLL fixtures had their per-record values, means and ratios
recomputed from retained log probabilities and UTF-8 byte counts using exact
rational arithmetic. Stored binary64 results agree within rounding. For example,
the 1,000-pair fixture reports baseline mean `1.4740014224751066` and subject mean
`1.4592316642958747`.

| Fixture | Stored ratio | Independent decision evidence |
| --- | ---: | --- |
| Captured NLL 1,000 | 0.9899798209458774 | Every pair ratio is approximately 0.96–1.02 |
| Captured NLL 5,000 | 0.9899923444071640 | Same support |
| Captured NLL 12,000 | 0.9899954128458798 | Same support |
| Captured NLL 50,000 | 0.9899993969051378 | Same support |
| Captured regression, 64 pairs | 1.2368974498225431 | Every pair ratio is approximately 1.20–1.275 |

With positive baseline values, every resampled ratio of means is a weighted
average of individual ratios. Therefore every possible resample for the first
four fixtures is below the threshold 1.1; its support width is below the allowed
0.25. Every regression-fixture resample is above 1.1. These well-separated
fixture decisions are established independently of the chosen sample indexes.
This argument does not establish behavior near a policy threshold.

For the **1,000-pair fixture only**, an additional check followed the
[specified SHA-256 index schedule][bootstrap] for all 2,048 replicates and
2,048,000 draws. It used integer frequency-weighted totals over a common rational
denominator, exact ratios, and exact percentile interpolation, importing no
production numerical helper. Both final endpoint floats equal the stored values:
`0.9886955183668685` and `0.99121849017269`. The preselected comparison tolerance
was 2e-15, about nine binary64 machine epsilons near one; observed differences
after final float conversion were zero. This is independent arithmetic under
the same mandated schedule, not an independent random schedule or a general
floating-point error bound. The larger NLL endpoints and the 64-pair endpoints
were not independently replayed.

The endpoint check's input identities are:

| Artifact | SHA-256 |
| --- | --- |
| Baseline run | `6dc80f77b3c157918e4ade52f24cef819c63518ea11a1882eb483aae5e35041a` |
| Subject run | `3e485166ab0786344f1a583f76d27b55fde07960bd55b7b18cb493e636723da1` |
| Stored comparison | `c17f71b1479c9139140086504fbec61938eeea3ac8c11eaceffca7baf7de7268` |

The native fixture has baseline accuracy 2/2, subject accuracy 1/2, effect -50
percentage points and exact McNemar probability 1. Numerical inversion of the
binomial score inequality, separately from the production Wilson closed form,
gives the retained Newcombe endpoints to floating precision:
`[-90.54687942657694, 27.257278511318162]`. Its pass is correct under its deliberately
permissive minimum of -100 percentage points. This is not a useful quality gate.

The judge fixture has 32 complete trials and 16 declared units. Exact nested
averages give subject mean 1 and effect 1. Substituting the retained endpoint
radii into the Hoeffding family-tail inequality at 150-digit precision confirms
the alpha bound and outward rounding. The effect interval is
`[0.259896406349600, 1]`; the subject interval is
`[0.629948203174800, 1]`. The required zero-degradation gate passes.
[The method][hoeffding] assumes independent bounded units; arithmetic does not
establish that assumption. All 16 answer pairs are the same answer, “Paris,” to
the same question, while synthetic baseline ratings are zero and subject ratings
are one. These fixtures establish faithful replay of attributed ratings, not a
real judge's behavior or a model improvement.

## Verification and reporting

The independent output assessment inspected **32 primary Python measurements**:
four measurements for each of eight workflows. It did not include warm-up or
qualification outputs. Checks used retained bytes, standard-library code and
cryptographic primitives without importing production verifiers or the benchmark
equality function.

Fixture inventories, canonical manifest bytes, Ed25519 signatures, signer pins,
receipt signatures and the distinct judge receipt domain were checked directly.
Receipt verdicts and identities matched the underlying verified results and
recipient policy. Composite stored results matched fresh outputs and the
conjunction of their members. All 32 output inventories matched their recorded
file hashes, including receipts, HTML, Markdown and JUnit. JUnit parsed and
preserved passing versus regression outcomes; the inspected narratives distinguish
rendering and authenticated receipts from recipient acceptance.

Seven workflows passed their domain policies. The captured-regression workflow
retained verification status 7, receipt-authentication status 0 and rendering
status 0: authenticating a failure receipt does not turn the failure into a pass.
The evidence-set result is an accepted conjunction and makes no joint-confidence
claim. Its synthetic components do not supply independent scientific evidence.

Thirteen adverse cases rejected at their intended boundaries. They covered wrong
keys, missing captured/judge files, a correctly re-signed false captured report,
receipt tampering, a wrong verifier anchor, a wrong native schedule anchor,
correctly re-signed judge omissions and parse/status inconsistencies, changed
stored composition acceptance, and a wrong shared subject-run pin. One positive
control authenticated a genuine failure receipt while preserving its failure.
These results cover the exercised boundaries, not every malformed artifact or
filesystem race.

## Numeric limitations and the exception gap

The [run schema][scores-schema] accepts JSON numbers, but recorded scores undergo
[explicit float conversion][float-conversion]. The frozen documentation's
[contribution recipe][float-doc] also converts retained scores to float and
acknowledges a rounding residual. Consequently these observations are accepted
numeric-domain limitations, not proven violations of a promise of exact integer
aggregation:

| Public comparison input | Observed behavior | Mathematical distinction |
| --- | --- | --- |
| Baseline `[2**60+1]`, subject `[2**60]` | Delta 0, interval `[0,0]`, pass with zero allowed degradation | Exact delta is -1; conversion erases it |
| Baseline `[2**53,2**53+2]`, subject `[2**53,2**53]` | Reported delta 0; interval `[-2,0]`, regression | Representable inputs still have exact delta -1 |
| Both sides `[1e308,1e308]` | Numeric-range refusal | Mathematical mean is finite; intermediate sum overflows |
| Both sides `[1e308,1e308,-1e308]` | Numeric-range refusal | Cancellation does not prevent intermediate overflow |
| NLL baseline `[5e-324,5e-324]`, subject `[5e-324,0]` | Point ratio 0; interval `[0,1]` | Mean underflow changes exact ratio 1/2 |
| Identical NLL sides `[0,1]` | Denominator refusal during resampling | Some resamples have undefined 0/0 |

Separately, baseline `[1e308,-1e308]` and reversed subject scores pass input
validation but overflow when paired differences are formed. Python raises raw
`ValueError: -inf + inf in fsum`, escaping the usual `EvaluationRecordsError`
normalization at the [interval call][interval-call]. This is an error-handling
gap, not a false pass. An existing Rust executable returned a structured error
with the same message. The reviewed executable also reproduced the large-integer
and representable-mean behavior; its SHA-256 was
`8b56991cb2879d99fefb436530b9cfe7362e3ffacf5b82603ff3dc42c5a50b75`.
That observation is tied to the executable hash, not asserted as a rebuild of the
measured Rust commit.

None of these extreme numeric inputs appears in the inspected performance
fixtures. Preserving the fixed outputs in this comparison does not make their
broader mathematical limitations disappear. Changing numerical semantics needs
an explicit compatibility decision; no such change was made here.

## Portable reproduction and limits

The [independent helper](https://github.com/invarlock/invarlock/blob/work/rust-evidence-core/experiments/rust/audit/reference_numerics.py) and
[small tests](https://github.com/invarlock/invarlock/blob/work/rust-evidence-core/experiments/rust/tests/test_reference_numerics.py) retain
mathematical expectations without depending on private fixture locations:

```bash
python -m pytest -q experiments/rust/tests/test_reference_numerics.py
python experiments/rust/audit/reference_numerics.py --diagnose
python experiments/rust/audit/reference_numerics.py \
  --baseline baseline.json --subject subject.json --report evaluation.report.json
```

Use Python 3.12 with the package and test dependencies installed. The diagnostic
command prints disagreements and exceptions; it does not turn them into passing
correctness assertions. Fixture replay is bounded to 1,000 pairs and prints
input hashes, exact rational endpoints and comparison results. It does not
validate signatures, complete schemas or provenance.

The small checks cover exact means/ratios, all 1,364 paired binary arrays of
lengths one through five, exact sign probabilities, tiny exhaustive bootstrap
populations, separately accumulated scheduled ratios, judge tail inequalities,
unequal unit sizes, reordering and missing data. They exercise useful independent
expectations, not a substitute for the retained-output checks.

The whole-transaction benchmark still compares candidates with the Python result;
its equality gate cannot detect a shared semantic defect. Fixture creation also
uses Python helpers, and synthetic recipient pins do not demonstrate independent
operational authority. This review adds bounded independent evidence. It does
not establish general statistical coverage, judge independence, truthful source
execution, near-threshold correctness across the numeric domain, every receipt
or schema combination, or all path/race/capacity behavior. Performance and
measurement-method findings are separate from this correctness review.

[bootstrap]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/src/invarlock/evidence_pack_contract.py#L905-L943
[hoeffding]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/src/invarlock/judge_measurements/statistics.py#L179-L195
[scores-schema]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/contracts/evaluation_run.schema.json#L129-L134
[float-conversion]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/src/invarlock/evaluation_comparison/comparison.py#L285-L305
[float-doc]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/docs/reference/evaluation-records.md#L411-L433
[interval-call]: https://github.com/invarlock/invarlock/blob/ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3/src/invarlock/evaluation_comparison/comparison.py#L334-L361
