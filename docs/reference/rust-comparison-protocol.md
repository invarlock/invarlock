# Rust comparison protocol

> **Reference**
>
> **Surface:** Experimental conformance, measurement and adoption criteria
>
> **Stability:** Experimental proposal; released behavior remains authoritative
>
> **Use this page when:** Reproducing or assessing the bounded Rust comparison

This protocol evaluates the [proposed Rust core](rust-core-design.md). It keeps
correctness, performance, distribution and maintenance conclusions separate.
An unfavorable result, partial adoption or no adoption is a valid outcome.

## References and candidates

Freeze original Python at `ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3`, tree
`95553894421fc6d3de83f50360457fff763ccda2`. Retain its wheel and dependency lock.
Record the Rust candidate's commit plus any uncommitted patch and artifact
digests. An unchanged version string is not sufficient artifact identity.

Compare three candidates on the same installed Python package and environment:

1. Original Python: existing implementation with no replacement.
2. Optimized Python: an explicitly recorded narrow optimization preserving the
   method. The first control reuses hash prefixes/counter encodings while keeping
   the same digest bytes, `math.fsum`, arithmetic and percentile order.
3. Rust: release-built native interval, explicitly selected. No silent fallback
   if import, validation or computation fails.

The experiment adapter substitutes only the declared private interval boundary.
It records calls, restores prior functions, and installs before dependent imports
in recipient processes. This qualifies an installed experimental path; it is not
production default dispatch or a completed standalone verifier.

At stage checkpoints fetch `origin/staging/next`, record its SHA/tree and diff,
and run affected conformance against that rolling control. Keep the original
fixed baseline and old results. Never mix baseline versions in one performance
cell or call a stale tracking reference the latest remote state.

## Complete architecture comparison

The three candidates above describe the completed interval experiment only.
For subsequent stages, the primary Rust candidate is a coherent transaction
implementation using the target architecture: contracts, stable snapshots,
validated records, analysis, trust and publication owned by the same core.
The experiment is allowed to restructure these internals and remove repeated
work. Preserve externally observable contracts, limits and historical semantics;
do not require a line-by-line translation of the Python implementation.

Report distinct questions and comparisons:

| Question | Comparison and interpretation |
| --- | --- |
| Is the new system better for users? | Complete Rust paths against fixed original Python and separately pinned current staging; include total architectural improvement |
| How much comes from avoiding repeated work? | A practical Python control with recorded validation/snapshot reuse and other justified optimizations; state its limits |
| What does the native implementation contribute? | Matched semantics, workload and resource allowance; retain component profiles and avoid claiming a pure language effect when internal algorithms differ |
| What does standalone delivery contribute? | Rust executable against the supported Python recipient entry point; measure startup, installation, dependencies and complete outputs |
| What does concurrency contribute? | Separate bounded parallel configurations with recorded worker and CPU budgets; retain a single-worker comparison |

Python controls can be developed alongside the Rust design. They need not become
an entire second redesign before Rust work proceeds. Freeze the controls and
their source identities before each timing matrix, record profiling separately,
and describe practical optimization coverage without claiming globally optimal
Python. Add targeted measurements only where a material cause remains unresolved.

The first complete comparison is captured verification. The final core comparison
covers supported native, captured, judge and combined evidence paths and the
standalone capability list. Optional Python scorer execution must remain visible
as Python-assisted verification; do not silently narrow the baseline's workload
or describe that mode as Python-free. Report installation, wall time, CPU, peak
memory, capacity and failure behavior as separate dimensions.

A component may have little isolated speed benefit while enabling shared data,
fewer allocations or fewer language crossings in a complete path. Such a result
does not by itself stop implementation of the declared coherent architecture.
Correctness and feasibility checkpoints still apply. Preserve completed interval
measurements unchanged: they establish that slice's result, not the full design's
performance ceiling.

The later execution/control comparison includes native command workflows,
bounded process I/O, admission, checkpoint/resume and cleanup. Compare the same
worker protocols, inputs, output limits and concurrency allowances. Retain
whole-process-tree resources where execution creates children. Deterministic
worker fixtures isolate local control overhead; they do not establish actual
provider performance. Qualify each materially changed real execution boundary
before claiming support, within existing authorization and resource constraints.

## Conformance comes first

Require exact canonical bytes and binary64 results on valid input, not just
matching policy verdicts or approximately equal intervals. Freeze hash/index
vectors and all replicate values for small inputs before testing percentiles.

The initial matrix includes count one, non-power-of-two counts, varying
ratios/deltas, repeated values, changed seeds/order/sides, zero subject values,
zero denominators, cancellation, wide magnitudes, subnormal values, overflow and
rounding/interpolation boundaries. Validate the binding's input domain separately
from caller-validated private helpers. Public exceptions and verdicts must retain
their existing meaning even when private invalid-input diagnostics differ.

Later canonicalization work adds integer/float distinction, exponent forms,
negative zero, large integers, escaped controls, combining characters, non-BMP
key ordering, malformed UTF-8, lone surrogates, duplicate members, nesting and
non-finite rejection. Keep distinct newline/digest profiles and all method IDs.

Run all permanent historical compatibility cases. Never edit a fixture or loosen
an expected value solely to accommodate the Rust output.

## Complete-path qualification

Before timing larger workloads, demonstrate the actual configured path:

```text
installed invocation -> retained facts -> scoring and policy -> signed pack
   -> independent offline recipient -> replay -> signed receipt
   -> receipt verification -> report
```

The first varying captured NLL fixture retains exact input records, policy,
pack, external anchors and test-only keys. Each recipient runs as a fresh process
from the installed wheel, without importing checkout `src/`. Verify its module
origin and selected backend. The same input pack and independent pins go to all
candidates. Receipt signatures and bindings must verify; timestamps, fresh
signatures or receipt destinations may be compared semantically only where the
protocol explicitly identifies them as variable. Canonical report bytes remain
exact. Retain receipt bytes as generated.

Exercise pass and regression first, plus a meaningful rejection and denominator
failure. Then include insufficient evidence, unsupported scope, local budget
refusal and operational failure. Preserve statuses and receipt/no-receipt timing.
Network-free model replay does not establish new provider execution qualification.
Any network isolation claim must name the mechanism actually exercised.

## Existing workload inventory

| Workload | Existing evidence and limitation |
| --- | --- |
| Captured exact 12,000/50,000 | `tests/evaluation_comparison/test_capacity.py::test_full_capacity_signed_independent_recipient`; complete signed independent recipient path |
| Captured scalar 12,000 | Same file's `test_capacity_varying_recorded_scalar_interval`; varying arithmetic, but extend to a signed transaction before complete-path claims |
| Captured scalar 50,000 | `test_default_resource_budget.py`; constant differences shortcut resampling, so capacity coverage does not measure expensive arithmetic |
| Native/captured NLL | `test_likelihood.py::test_native_parity_with_varying_paired_ratios` and `test_signed_captured_likelihood_replay_receipt_and_reports`; add larger varying signed fixtures |
| Native signed verification | `test_evidence_verification_transaction.py::test_independent_verification_user_journey_emits_a_signed_external_receipt` |
| Captured frozen contract | `tests/fixtures/evaluation_contracts/captured-v2.json` and `test_captured_contract_freeze.py` |
| Historical native | `tests/fixtures/compatibility/v0.13.0` and every `tests/compatibility` test |
| Judge 7,728 completed trials | `test_inspect_judge.py::test_reference_capacity_7728_completed_trials_shards_and_replays`; synthetic retained import/sharding, not provider execution or a signed CLI path |
| Judge signed offline replay | `test_workflow.py::test_signed_recipient_cli_replay_and_receipt_remain_offline`; native judge recipient-policy journey in `test_native_judge_transaction.py` |
| Judge numerical calibration | `test_statistics.py` and `tests/fixtures/judge_measurements/statistics_calibration.json` |

Include retained real material from `public_evidence/evidence/`, evaluator signed
transactions, captured references including harness likelihood, Mistral likelihood
and K2 routing, judge references, priority workflows and acceptance-handoff golden
evidence. Inventory exact files, formats, source provenance and expected outcome.

`release_reference_journey.py --all-retained-evidence` currently selects seven
public native packs and four evaluator native transactions. Those eleven packs
are not universal captured/judge replay coverage. Maintain explicit additional
entries rather than inferring coverage from the command name.

## Failure, capacity and operational matrix

For affected boundaries include missing/duplicate cases, altered references,
provenance changes, multiple metrics/slices, budget exact/one-less limits and work
admission before selective arithmetic. Test physical and canonical byte ceilings,
including large aggregate output separately from many short records.

Filesystem changes require mutation during reads, symlinks, FIFOs, descriptor
substitution, unsafe destinations, interrupted writes, cleanup and no-clobber
publication. Judge collection changes require bursts, slow consumption,
checkpoint interruption, refusal, duplicate responses, retry accounting,
reservations and retained-byte exhaustion. A rejected oversized input does not
prove a valid maximum-size input succeeds.

Useful suites include `test_resource_budget.py`, `test_default_resource_budget.py`,
`test_missing_capacity.py`, `test_captured_input_capacity.py`,
`test_accounting_sequences.py`, `test_runner_boundaries.py`,
`test_captured_filesystem_failures.py`, `test_captured_security_contracts.py` and
`test_evidence_pack_snapshot.py`. Read configurations before launching anything
named live; do not infer authorization for provider spend from a test name.

## Timing protocol

Freeze the run manifest before reading results. The first bounded matrix is:

| Cell | Counts | Rounds | Scope |
| --- | --- | --- | --- |
| Native NLL kernel | 1,000 and 5,000 | Five paired rounds | Full interval call including binding conversion/allocation; fixture preparation outside timing |
| Installed recipient replay | 1,000 and 5,000 | Five paired rounds | Fresh process, imports, parsing, authentication, recomputation, receipt publication and verification |
| Larger capacity extension | 12,000 and 50,000 | Three paired rounds when separately launched | Same complete path after small qualification; not implied by kernel-only capacity |

Use one machine, sequential candidates and a recorded seeded balanced order.
Use identical fixture/trust bytes, storage/filesystem, workload budgets, numerical
methods and thread controls. Rust must be a release build. Record whether build
and fixture generation are excluded. A fresh process is not a cold filesystem
cache; describe cache conditions accurately.

Retain every sample and report medians, paired differences and the observed
range. Five or three rounds do not establish tail-latency percentiles. Measure
wall time and user/system CPU. Record peak RSS with correct platform units and
scope; a child's `ru_maxrss` includes its startup/setup and is not incremental
kernel memory. If children are launched, account for them explicitly or label
the result as main-process-only. Never add unrelated high-water RSS values and
call the sum a simultaneous process-tree peak.

Keep kernel, warm in-process transaction, fresh-process replay, evaluation
transaction and installation measurements separate. Initial cells do not imply
every category was measured. A signing or output path change requires retained
full-path checks even if the arithmetic result matches.

## Outcome and retry rules

Candidate crashes, timeouts, incorrect output, memory exhaustion, capacity refusal
and installation failure remain outcomes. They cannot be excluded because they
make a candidate look slower. Retain stdout/stderr, exit status, timeout state,
backend call evidence and produced artifacts for each attempt.

Only an objectively recorded harness/host failure unrelated to candidate behavior
can invalidate an entire paired block, for example host shutdown or corrupted
input copying. Permit at most one replacement of that whole block, rerunning all
candidates and retaining both attempts. Do not select the fastest rerun. A shared
measurement defect stops dependent timing until its repair has a decisive check.
Keep profiling runs separate from performance samples.

## Retention and comparison record

For each run retain:

- Objective, selected scope, protocol revision, source commits/trees/patches and
  as-of cutoff.
- Wheel/binary hashes, Rust target/profile/features/toolchain, Cargo lock, Python
  version, installed dependency inventory and build-tool versions.
- OS/architecture, CPU, filesystem/storage, thread controls, cache conditions,
  candidate order, watchdogs and timing boundaries.
- Exact fixture and independent-trust inventories with byte sizes and hashes;
  fixture generation source and all output packs/receipts/results.
- Every attempted sample, exclusions/retries with denominators, conformance
  results, resource scope and explicit untested boundaries.

Test-only keys must be visibly designated and never used for production trust.
Public evidence summaries omit private machine paths and execution-process
markers. Artifact hashes establish identity; their presence does not establish
correctness, completeness or performance.

## Adoption decision

Correctness and historical/trust compatibility are mandatory. Performance and
deployment benefits are measured dimensions, not preset winning thresholds.
Assess absolute saved time and memory alongside native build/release burden,
platform coverage, debugging and ongoing ownership. Compare Rust with both Python
controls. Measure standalone installation separately when that executable exists.

Make adoption decisions for coherent components or groups of components using
complete-path evidence supporting their intended benefit and no unresolved
material compatibility failure. A small local kernel speedup does not justify
a whole-repository migration or set the ceiling for its potential benefit. A
complete verifier may be valuable for distribution even without a large kernel
speedup; that value needs its own evidence. Record the adopted boundary and its
dependencies so a successful shared-core result is not attributed to one isolated
helper. Retain rejected alternatives and distinguish production adoption from
intermediate implementation checkpoints.

The initial implementation has a 45-minute checkpoint; the small/medium timing
matrix has a 30-minute watchdog budget after qualification. A checkpoint is a
scope/recovery decision, not permission to report incomplete work as complete.
Larger capacity, other statistical families, a standalone verifier and production
release qualification remain separate stages until their actual evidence exists.
