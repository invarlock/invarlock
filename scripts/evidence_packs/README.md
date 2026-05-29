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
orchestration-focused. Shared verification parsing now lives in
`scripts/evidence_packs/python/verify_pack_checks.py` instead of inline shell
heredocs.

Workflow boundaries:

| Area | Owner | Notes |
| --- | --- | --- |
| Local smoke | `run_mini_pack_gate.sh`, shell tests | Must support `--dry-run`. |
| Release evidence build | `run_pack.sh`, `run_suite.sh` | Use explicit suite/scenario manifests. |
| Verification | `verify_pack.sh`, Python verification helpers | Offline-first; signing-key pinning is forwarded when supplied. |
| Remote/GPU campaigns | scenario manifests and model evidence sweep callers | Keep campaign-specific state out of root `scripts/`. |

Deprecation rule: mark stale helpers in this README or
`scripts/scripts_inventory.toml` first, keep the stable front door for one
release cycle, then remove the helper once Makefile, CI, docs, and tests no
longer reference it.
