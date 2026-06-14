# Report Outline

This page defines the renderer-neutral structure for modern InvarLock
evaluation reports. It is the bridge between the canonical
`evaluation.report.json` payload and human-readable renderers such as Markdown,
HTML, evidence-pack summaries, and benchmark comparison pages.

The outline is implemented by
`invarlock.reporting.report_outline.build_evaluation_report_outline`.

## Why This Exists

The original report body was built as a linear Markdown document. That worked
for early GPT-2/BERT preservation reports, but the repo now emits richer report
shapes:

- policy failures, warning-mode guard movement, and strict warning policies
- causal, MLM, seq2seq, image-text, and MoE evidence lanes
- primary-metric tail checks and measured accuracy floors
- public assurance-basis reports with runtime manifests and model revisions
- guard-value evidence and benchmark-style bare-vs-guarded comparisons

Renderers should not decide this information architecture independently. They
should render the shared outline.

## Canonical Section Order

| Section | Purpose | Typical source blocks |
| --- | --- | --- |
| Decision | Overall verdict, evidence mode, model/edit identity, warning count. | `validation`, `assurance`, `meta`, `primary_metric`, `guard_warnings` |
| Primary Metric | Task metric, final value, baseline-relative comparison, CI, tail gate. | `primary_metric`, `primary_metric_tail`, `validation` |
| Policy Gates | Hard verify gates and thresholds. | `validation`, `policy_digest`, `resolved_policy` |
| Guard Signals | Guard observations and warnings separate from hard failures. | `guard_warnings`, `invariants`, `spectral`, `rmt`, `variance`, `moe` |
| Benchmark Comparison | Optional bare-vs-guarded scenario deltas. | `benchmark_comparison`, `benchmark`, `guard_effect_benchmark` |
| Evidence And Provenance | Dataset, windows, runtime/policy/provider digests, device, seed. | `dataset`, `provenance`, `policy_digest`, `meta`, `artifacts` |
| Technical Appendix | Verbose raw measurements, resolved policy, plugins, artifacts. | `plugins`, `resolved_policy`, `policy_provenance`, `system_overhead`, `classification`, `structure`, `artifacts` |

The benchmark section is omitted when no benchmark block is present.

## Renderer Rules

- Keep policy failures, guard warnings, and guard-value evidence distinct.
- Keep primary metric interpretation task-aware: ppl-like metrics use ratios;
  accuracy uses percentage-point deltas.
- Put benchmark deltas after guard signals, not in provenance or appendix.
- Keep verbose policy YAML, plugin provenance, and raw artifacts in the
  technical appendix unless they are needed to explain the verdict.
- Treat the outline as the source for visible section order in future Markdown
  and HTML renderers.
