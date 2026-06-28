# Evidence Pack Scripts

## Operator Entrypoints

| Command | Purpose | Runtime |
| --- | --- | --- |
| `scripts/evidence_packs/run_pack.sh` | Run a suite and package a distributable evidence pack. | Medium to long; network/GPU optional by suite. |
| `scripts/evidence_packs/run_suite.sh` | Run configured scenarios without packaging an evidence pack. | Medium to long; useful for diagnostics and raw campaigns. |
| `scripts/evidence_packs/verify_pack.sh` | Verify an existing evidence pack, including nested reports and signatures. | Fast to medium; offline. |
| `scripts/evidence_packs/run_mini_pack_gate.sh` | Local dry-run and targeted mini-pack gate used by tests. | Fast; offline by default. |

Use `run_pack.sh --release-review` for reviewer-facing evidence. `run_suite.sh`
is intentionally not a pack publisher; it executes the same scenario machinery
but leaves packaging and strict pack verification to `run_pack.sh`.

Keep `run_pack.sh`, `run_suite.sh`, and `run_mini_pack_gate.sh` as stable
operator entrypoints. Refactor their internals behind those names when needed
instead of merging or renaming them.

## Repository Layout

Everything under `lib/`, `python/`, and `tests/` is an implementation helper for
the entrypoints above. The top-level manifests are part of the harness contract:

- `scenarios.json`: generated edit, deployable edit, and error/probe scenario
  definitions.
- `tuned_edit_params.json`: tuned clean-edit parameters used by generated edit
  scenario creation.

- `lib/core/`: portable runtime, retry/fault-tolerance, remote setup.
- `lib/config/`: dataset/provider and InvarLock config rendering.
- `lib/tasks/`: model creation, task execution, and task JSON serialization.
- `lib/queue/`: queue lifecycle, GPU scheduling, and worker loops.
- `lib/validation/`: suite orchestration and verdict compilation.
- `python/`: Python helpers for workflow planning, model/edit generation,
  manifests, validation state, report extraction, and verdict generation.
- `tests/`: Bash harness tests for entrypoints and shell helpers.

## Tests and Coverage

There are two relevant evidence-pack test surfaces:

- `make test-evidence_packs` runs the Python pytest suite under
  `tests/evidence_packs`.
- `scripts/evidence_packs/tests/run.sh` runs the Bash harness tests for
  evidence-pack shell entrypoints and `lib/**/*.sh` helpers.

The Bash harness supports focused runs and strict shell coverage modes:

```bash
scripts/evidence_packs/tests/run.sh --filter 'test_model_creation'
scripts/evidence_packs/tests/run.sh --coverage
scripts/evidence_packs/tests/run.sh --coverage --line-coverage --jobs 8
```

`--coverage` runs the Bash tests under xtrace and enforces 100% branch-arm
coverage for evidence-pack Bash scripts. `--line-coverage` also enforces 100%
executable-line coverage. The target inventory includes
`scripts/evidence_packs/*.sh` and `scripts/evidence_packs/lib/**/*.sh`; when a
coverage miss occurs, the harness prints an owner-hint test file to update.
Coverage artifacts are written under `scripts/evidence_packs/tests/.coverage/`
for local inspection. These strict shell coverage flags are explicit audit
gates, not part of the default `make test-evidence_packs` target.
Use `--jobs N`, or set `EVIDENCE_PACK_TEST_JOBS=N`, to run selected Bash tests
in parallel. Each coverage worker writes an isolated xtrace hit file that is
merged before branch and line coverage are checked.

Python coverage for evidence-pack helper modules is enforced by the top-level
`make coverage-enforce` target and `scripts/coverage/check_coverage_thresholds.py`.
Do not treat the Bash harness coverage flags as a substitute for the Python
coverage gate.

## Implementation Boundaries

New JSON/state/path validation logic should be Python-first under
`scripts/evidence_packs/python/`; shell wrappers should stay thin and
process-focused. Direct non-help invocations of `run_pack.sh`, `run_suite.sh`,
and `run_mini_pack_gate.sh` route through
`scripts/evidence_packs/python/workflow_frontdoor.py`, which builds the typed
`scripts/evidence_workflows` plan used for dry-run/execution, status logs,
summaries, and artifact manifests. Shared verification parsing now lives in
`scripts/evidence_packs/python/verify_pack_checks.py` instead of inline shell
heredocs.

The remaining `lib/*.sh` files are acceptable where they coordinate processes:
locking queue directories, moving task files between states, launching workers,
handling signals, and dispatching remote commands. Structured state mutation
belongs in Python. Queue retry/progress JSON is handled by
`scripts/evidence_packs/python/queue_state.py`; future queue changes that parse
or rewrite task JSON should extend that helper instead of adding more `jq`
programs or shell heredocs.

Generated edit creation currently includes quantization, magnitude pruning,
low-rank SVD, FP8 quantization, LoRA merge, and tiny fine-tune validation
subjects. Scenario requirements determine whether a lane is `must_pass`,
`must_fail`, or informational; adding a generator does not make a scenario
claim-bearing by itself.

## Python Helpers

- `python/create_edit_model.py`: one-shot validation-subject edit creation
  (`quant-rtn`, `magnitude-prune`, `lowrank-svd`, `fp8-quant`,
  `lora-merge`, `fine-tune`).
- `python/create_edits_batch.py`: batched edit creation from tuned edit specs.
- `python/editing/`: shared edit metadata, targeting, implementation, save,
  validation, and deployable artifact helpers used by edit entrypoints.
- `python/preset_generator.py` and `python/preset_calibration.py`: generated
  preset selection and calibration support.
- `python/runtime_tools.py`: runtime environment checks plus shared Hugging Face
  model-loading helpers.
- `python/task_tools.py create-error-model` + `python/error_model/`:
  structural/error injection subject creation.
- `python/task_tools_*.py`: report, model, preset, and task helper commands used
  by shell orchestration.
- `python/workflow_frontdoor.py`: typed workflow plan construction for the shell
  entrypoints.
- `python/verdict/`: verdict table and verdict-generation internals.

## Workflow Ownership

| Area | Owner | Notes |
| --- | --- | --- |
| Local smoke | `workflow_frontdoor.py` -> `run_mini_pack_gate.sh` | Must support `--dry-run`; shell owns scenario selection and worker dispatch only after workflow launch. |
| Release evidence build | `workflow_frontdoor.py` -> `run_pack.sh`/`run_suite.sh` | Use explicit suite/scenario manifests; workflow layer owns command plan, status log, summary, and artifact manifest. |
| Verification | `verify_pack.sh`, Python verification helpers | Offline-first; signing-key pinning is forwarded when supplied. |
| CUDA campaigns | scenario manifests and model evidence sweep callers | Keep campaign-specific state out of root `scripts/`. |

## Guard-value publishing rule

- PM-only acceptance is necessary but not enough.
- A public guard-value case must compare against the matching noop baseline and
  count only baseline-relative guard movement, such as a new capped spectral
  module, an increased cap count, an RMT epsilon violation relative to baseline,
  or a VE sidecar signal absent from the baseline self-probe.
- This is stricter than ordinary paired evaluation: the primary metric already
  compares baseline and subject, while guard-value publishing also requires the
  guard signal itself to move beyond the no-op basis.
- Evaluation reports may also carry `guard_warnings`. Evidence-pack summaries
  preserve those warnings for review, but guard-value publishing still requires
  reproduced scenario evidence and clean confirmation reruns.
- Clean confirmation reruns are required before publishing a case as guard-value
  evidence.
- The current reference package is
  `public_evidence/published_basis/mistral_7b/guard_value_demo/`, especially
  `artifact_package/reports/guard_value_all_guard_probe_sweep.json`.

## Cleanup Rule

Scripts under `python/` are not public package APIs. If a helper is unreferenced
or only preserves an obsolete internal call path, remove or move it in the same
change that updates repo-owned shell callers, docs, and tests.
