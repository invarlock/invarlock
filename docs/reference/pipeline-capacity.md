# Pipeline capacity and resource limits

!!! info "Reference"

    **Surface:** Pipeline record, byte and statistical-work limits.

    **Stability:** Unreleased defaults with measured workload-specific results.

    **Use this page when:** Sizing a complete comparison or choosing local
    resources and work allowances for independent replay.

The pipeline SDK compares captured evaluation records on a CPU. Its limits
govern different resources: 50,000 paired records, 128 MiB per input, 384 MiB
per complete signed evidence file, and a default of 102,400,000 planned scalar
bootstrap draws. All limits apply together. See the
[contract reference](pipeline-contracts.md#local-statistical-work-budget) for
the exact counting rules and caller-owned overrides.

## Why these limits are separate

The reviewed evaluation tools establish no common 12,000- or 50,000-row
standard. Inspect exposes dataset memory and concurrency controls separately;
promptfoo exposes concurrent-call and test-range controls. These are useful
precedents for separating resources, not claims of equivalent signed replay or
unlimited capacity. See [Inspect parallelism](https://inspect.aisi.org.uk/parallelism.html)
and [promptfoo CLI controls](https://www.promptfoo.dev/docs/usage/command-line/).

The earlier 12,000-row target followed a particular evaluation design. Actual
4,000-case exports then showed why row count alone was insufficient. Larger
performance fixtures derived from enriched financial-QA exports needed
95.39 MiB per input at 12,000 records, or 96.49 MiB with four scalar metrics.
The latter produced 192.87 MiB of complete signed evidence. The previous
64 MiB input and 192 MiB evidence limits excluded these workloads.

The 128 MiB input allowance provides about 33% headroom above the measured
96.49 MiB input. The 384 MiB evidence allowance accommodates two embedded inputs
and additional policy/comparison data, subject to final size validation. It is
not a proof that every combination of independently saturated inputs, policy
and comparison will fit.

Compact exports can fit 50,000 records; larger enriched shapes in the same study
needed roughly 251–397 MiB per input at that count and remain outside the byte
limit. A 128 MiB input provides approximately 2.68 kB per record at 50,000 rows,
before top-level overhead. Any supported projection must preserve the intended
pairing and authenticated provenance. Splitting a complete comparison into
independent verdicts does not preserve its original policy assessment.

## Measured CPU envelope

Measurements on September 7, 2026 used Python 3.12.3 and a Linux container on an
AMD EPYC 9555 host, restricted to two CPUs and 8 GiB memory with no extra swap,
network or GPU access. The target, declared before execution, was at most 120 seconds and 4 GiB
process peak RSS for **each** creation and independent replay phase. Fixture
preparation and container setup were outside those phase times. This resource
choice was informed by modest private CI runners; these measurements were not
made on GitHub-hosted runners. See [GitHub runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

| Workload | Largest input | Creation / replay | Peak RSS, creation / replay |
| --- | ---: | ---: | ---: |
| 4,000 captured routing records | 18.95 MiB | 6.01 / 4.81 s | 0.23 / 0.17 GiB |
| 12,000 financial-QA performance records, four scalar metrics with varying differences | 96.49 MiB | 33.34 / 27.54 s | 0.79 / 0.69 GiB |
| 12,000 nested performance records with Unicode text | 128 MiB exactly | 27.64 / 21.03 s | 2.12 / 1.94 GiB |
| 50,000 compact structured performance records, one scalar metric with varying differences | 128 MiB exactly | 63.38 / 51.56 s | 2.68 / 2.44 GiB |

Each boundary run produced slightly more than 256 MiB of signed evidence and
independently replayed their declared decisions. The measured implementation was commit
`8361eeb81b20ad691ef697b2d57e6112385def0d`; the 50,000-case run explicitly used
102,400,000 draws. The subsequent default adopts that same allowance. The
12,000-case four-metric workload uses 98,304,000 draws and also fits.

These larger fixtures repeat captured record shapes with original-source
identities and explicitly authored performance scores. They add no independent
model-quality observations. Every phase used a fresh Python process, but OS
caches were not flushed. Two earlier 18-cell matrices included repeated visits;
container memory reached its 8 GiB limit without recorded OOM kills. Container
memory includes file cache and controller memory and differs from process RSS.
The table is a tested envelope, not a universal latency or memory guarantee for
every valid JSON shape, policy or host.

## Implementation tradeoffs

An initial incremental-encoding implementation was slower than the original
whole-object encoder: creation phases increased by 19–41% and replay phases by
22–48% in the matched 18-cell matrices. Most process-memory changes were small.
The final bounded-subtree implementation recovered most of that overhead. In
the two final matched workloads, it remained approximately 2.5–14% slower than
the original, depending on phase and workload.
Those earlier matrices used temporary copies with identical higher capacity
constants to measure otherwise-rejected shapes. Their exploratory limits were
not release defaults. The final table used the unmodified production row and
byte limits.

The retained benefit is earlier byte rejection and avoiding whole-artifact
serialization buffers during validation and hashing. It is not a general
throughput improvement. Full JSON parsing, copying, signature construction and
output publication still consume memory proportional to the artifact. Large
individual strings retain the standard encoder's allocation behavior.

## Tokens and statistical sample size

Pipeline replay loads neither model weights nor a model tokenizer. Generation
capacity separately depends on the exact tokenizer and chat template, input,
output and reasoning tokens, architecture, cache precision and concurrency.
Prompt bytes are not token counts. See [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating).
Model choice can indirectly increase replay cost through longer captured
answers, reasoning traces or additional context in the exported records.

Supported row count also does not establish power, interval coverage or
representative sampling. Sample planning depends on the intended effect,
uncertainty and source dependence. Repeated shapes used to measure capacity
cannot increase the effective quality sample. See
[NIST sample-size guidance](https://itl.nist.gov/div898/handbook/ppc/section3/ppc333.htm).
