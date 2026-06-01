# Evidence Scope

An integration example produces regression evidence for one configured
baseline-vs-subject checkpoint comparison.

The evidence is strongest when the workflow runs in the default container-backed
strict mode and the generated `evaluation.report.json` verifies with its sibling
`runtime.manifest.json`.

## Example Status Labels

| Label | Meaning |
| --- | --- |
| `runnable` | The target README provides commands that generate an evaluation report, verifier JSON, and HTML report in the documented environment. |
| `exploratory-host` | The target workflow runs in host mode with `--assurance off`; useful for local debugging and backend bring-up. |
| `compatibility-investigation` | The target artifact cannot yet be loaded or verified through the documented InvarLock path; the README records the blocker. |

## Maintainer-Facing Wording

Use scoped language:

> This example produces regression evidence for one configured baseline
> checkpoint versus one edited subject checkpoint.

Keep public claims tied to the generated artifacts, the selected model family,
the adapter, the dataset/window plan, and the verifier result.

## Strict Evidence Checklist

- `invarlock evaluate` ran with `--assurance strict` or the default strict mode.
- The default container execution path produced `runtime.manifest.json`.
- `invarlock verify --assurance strict` passed for `evaluation.report.json`.
- `verify.json` was generated from the same report with `--json`.
- `evaluation.html` was rendered from the same report.
- The target README records optional dependencies and any backend limitations.
