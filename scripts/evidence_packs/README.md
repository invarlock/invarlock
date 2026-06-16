# Evidence Pack Scripts

Stable front doors:

| Command | Purpose | Runtime |
| --- | --- | --- |
| `scripts/evidence_packs/run_pack.sh` | Build a full evidence pack from configured scenarios. | Medium to long; network/GPU optional by suite. |
| `scripts/evidence_packs/verify_pack.sh` | Verify an existing evidence pack, including nested reports and signatures. | Fast to medium; offline. |
| `scripts/evidence_packs/run_mini_pack_gate.sh` | Local dry-run and targeted mini-pack gate used by tests. | Fast; offline by default. |

Everything under `lib/`, `python/`, `tests/`, and `fixtures/` is an
implementation helper for those entry points. `lib/` is split by concern:

- `lib/core/`: portable runtime, retry/fault-tolerance, remote setup.
- `lib/config/`: dataset/provider and InvarLock config rendering.
- `lib/tasks/`: model creation, task execution, and task JSON serialization.
- `lib/queue/`: queue lifecycle, GPU scheduling, and worker loops.
- `lib/validation/`: suite orchestration and verdict compilation.

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

Python helper boundaries:

- `python/create_edit_model.py`: one-shot validation-subject edit creation
  (`quant-rtn`, `magnitude-prune`, `lowrank-svd`, `fp8-quant`).
- `python/create_edits_batch.py`: batched edit creation from tuned edit specs.
- `python/editing/`: shared edit metadata, targeting, implementation, save,
  validation, and deployable artifact helpers used by edit entrypoints.
- `python/runtime_tools.py`: runtime environment checks plus shared Hugging Face
  model-loading helpers.
- `python/task_tools.py create-error-model` + `python/error_model/`:
  structural/error injection subject creation.
- `python/verdict/`: verdict table and verdict-generation internals.

Workflow boundaries:

| Area | Owner | Notes |
| --- | --- | --- |
| Local smoke | `workflow_frontdoor.py` -> `run_mini_pack_gate.sh` | Must support `--dry-run`; shell owns scenario selection and worker dispatch only after workflow launch. |
| Release evidence build | `workflow_frontdoor.py` -> `run_pack.sh`/`run_suite.sh` | Use explicit suite/scenario manifests; workflow layer owns command plan, status log, summary, and artifact manifest. |
| Verification | `verify_pack.sh`, Python verification helpers | Offline-first; signing-key pinning is forwarded when supplied. |
| Remote/GPU campaigns | scenario manifests and model evidence sweep callers | Keep campaign-specific state out of root `scripts/`. |

Guard-value publishing rule:

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

Breaking cleanup rule: scripts under `python/` are not public package APIs. If a
helper is unreferenced or only preserves an obsolete internal call path, remove
or move it in the same change that updates repo-owned shell callers, docs, and
tests.
