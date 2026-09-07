# Pipeline capacity and resource limits

The pipeline SDK compares captured evaluation records on a CPU. Record count,
file size and statistical work have separate limits; a comparison must fit all
of them.

!!! info "Reference"

    - **Surface:** Capacity limits for `invarlock-pipeline` and `invarlock.pipeline`
    - **Stability:** Row and byte limits are fixed by the pipeline formats; the statistical work budget is caller-configurable
    - **Use this page when:** Sizing a comparison, choosing a local work budget, or resolving a capacity error

## Limits and caller controls

| Resource | Limit or default | Caller control |
| --- | ---: | --- |
| Paired cases | 50,000, with at most 50,000 records in each run | Fixed contract limit; no supported override |
| Each input file or normalized run | 128 MiB | Fixed limit; no supported general override |
| Complete signed evidence | 384 MiB | Fixed limit; no supported override |
| Planned scalar bootstrap draws | 102,400,000 per comparison or verification | Python `max_bootstrap_draws` or CLI `--max-bootstrap-draws` |
| Metrics | 16 per policy | Fixed contract limit |
| Named slices | 16 per policy, plus `overall` | Fixed contract limit |

One pair contains the baseline and candidate result for the same case. A
50,000-pair comparison therefore contains up to 100,000 records across its two
runs. Slices select from these cases; they do not create extra record allowances.

The 128 MiB canonical size limit also applies to planned case sets, policies and
comparison results. Physical input bytes and normalized canonical bytes are
checked separately. Signed evidence embeds both runs, the policy and the
comparison, so valid individual inputs do not guarantee that the complete
evidence fits its final limit.

These limits apply to the pipeline comparison SDK. Runtime schedules and
external scoring imports retain their separate 10,000-record limits. See the
[pipeline contracts](pipeline-contracts.md#records-and-identities) for exact
record and pairing requirements.

## Choosing a statistical work budget

Scalar intervals use 2,048 bootstrap repetitions. The planned work is the sum
of the selected pair counts for every scalar metric and every scope, multiplied
by 2,048. Each scope includes either the complete schedule (`overall`) or one
named slice. Overlapping slices count separately.

| Workload | Planned draws |
| --- | ---: |
| 50,000 pairs, one scalar metric, overall only | 102,400,000 |
| 50,000 pairs, two scalar metrics, overall only | 204,800,000 |
| 12,000 pairs, four scalar metrics, overall only | 98,304,000 |

For an existing project with sufficient local resources, allow the second
workload explicitly:

```bash
invarlock-pipeline compare pipeline.json --output comparison \
  --max-bootstrap-draws 204800000
```

The `verify` command accepts the same option. In Python, `compare_runs`,
`create_evidence` and `verify_evidence` accept `max_bootstrap_draws` as a
non-negative integer. An explicit `None` disables this additional work bound
in the Python API; row and byte limits still apply.

Each recipient chooses its own verification budget. An author's larger
allowance cannot increase the recipient's allowance. Changing the budget does
not change the policy, acceptance thresholds or bootstrap repetitions.

Recorded numeric metrics incur the scalar charge even when their values are
zero or one. Missing values and constant differences do not reduce the planned
charge. Binary built-in metrics use a different interval method and incur no
bootstrap charge. They still require CPU work. See the
[counting rules](pipeline-contracts.md#local-statistical-work-budget).

## Sizing a workload

The record ceiling is a supported contract boundary, not a recommended sample
size or a demonstrated maximum for every machine. At 50,000 records, a 128 MiB
input allows approximately 2.68 kB per record before shared file overhead.
Document context, long answers, nested objects and retained traces can exhaust
the byte allowance at a much smaller case count.

JSON parsing, copying, signing and output publication consume memory beyond
the encoded input size. Incremental canonical encoding bounds intermediate
serialization work, but does not make the complete workflow stream through
constant memory. Large individual strings also require their own allocations.

The pipeline SDK sets no universal wall-time or process-memory limit. Configure
those limits in the CI runner, container or job scheduler. Size resources using
the intended record shapes, metrics and slice overlap, measuring both evidence
creation and independent verification. A bootstrap draw allowance is not a
promise about elapsed time or peak memory.

Comparison and replay load neither model weights nor the model's tokenizer.
Model choice affects their workload indirectly through captured outputs and
context. Generation has separate model, token, runtime and hardware requirements.

## Capacity errors and recovery

Exceeding a capacity bound raises `PipelineError`. The CLI returns integration
error status 2; it does not issue a quality verdict or publish a completed
comparison directory. It does not silently truncate cases, omit slices or
reduce statistical repetitions to fit the budget.

For a bootstrap budget error, assess the required work and increase the local
allowance explicitly if the machine can support it. For a row or byte error,
there is no supported setting that raises the fixed bound. A larger allowance
requires an implementation change, including matching schema changes for
schema-enforced limits, and verification support on the recipient.

A supported input projection must preserve the intended pairing, task meaning
and authenticated provenance. Splitting a complete comparison into separate
verdicts does not preserve its original policy assessment. Any smaller study
needs its own explicit case selection and policy; it must not silently replace
the planned comparison.

Capacity does not establish statistical power, representative sampling or
independent observations. Plan the sample for the intended decision and source
dependence. See [policy and statistics](pipeline-contracts.md#policy-and-statistics)
for the shipped methods and their assumptions.
