# Scripts Directory

`scripts/` contains repo-maintainer tooling, not install-time package APIs. The
root directory is intentionally small; stable commands live under family
subdirectories and are normally reached through `make` targets or documented
workflow paths.

The checked taxonomy lives in `scripts/scripts_inventory.toml`:

- `scripts-governance`: scripts-tree inventory, hygiene, and Python selection.
- `repo-contracts`: repo consistency checks for contracts, claims, config, CLI,
  guards, and version metadata.
- `docs-assurance`: docs, examples, links, and assurance cross-reference checks.
- `coverage`: coverage surface selection and threshold checks.
- `smoke-runtime`: local smoke, runtime, tiny-model, and guard-validation entry
  points. See `scripts/smoke/README.md` for the per-script smoke map.
- `catalog-lane-production`: single-lane catalog evaluation, verification, and
  signed staging-pack production.
- `model-evidence`: bounded public-input materialization for maintained catalog
  lanes.
- `evidence-packs`: local artifact validators and evidence-pack verification.
- `release`: release evidence and offline-bundle helpers.
- `security-supply-chain`: SBOM, CVE audit, pip-audit, requirements pinning, and
  scorecard helpers.

High-signal workflow front doors:

- `bash scripts/smoke/run_tiny_all_matrix.sh`: write a dry-run checklist for the
  tiny model evaluation matrix. Set `RUN=1` to execute it and `NET=1` when model
  downloads are allowed. This covers compact causal-LM, encoder-MLM, and quant
  demo evaluation paths.
- `python scripts/smoke/run_tiny_fine_tune_byoe_smoke.py`: run the local
  CPU-only BYOE fine-tune smoke against a cached tiny GPT-2 model, then verify
  the enriched report with evaluation-realism, topology, and delta/privacy
  metadata. Deterministic synthetic dense-update validation-subject coverage
  lives in the evidence-pack harness; that generator does not perform
  fine-tuning or use training data.
- `python scripts/checks/check_model_candidate_compatibility.py`: run the
  offline candidate compatibility audit used by `make contracts-check` before
  evaluating entries from the public evidence catalog.
- `python scripts/model_evidence/run_catalog_lane.py`: evaluate one catalog lane,
  strictly verify its report, assemble and sign its evidence pack with a
  caller-supplied key, strictly verify the pack, and expose the result in a
  staging directory. See `scripts/evidence_packs/README.md` for the inputs and
  execution boundary.
- `python scripts/checks/benchmark_command_hotpaths.py --json`: measure the
  schema-validation, bootstrap, report-assembly, evidence-snapshot/verification,
  and evidence-verification hot paths with deterministic input/output digests. Results
  are written only to standard output. Use repeatable `--benchmark` selectors for
  a subset and `--cuda` to include CUDA availability and peak-memory fields.
  For optimization acceptance, supply a prior payload with `--baseline-json`,
  identify changed paths with repeatable `--target`, and retain the default
  requirement of at least 10% target improvement with no untargeted regression.
- `scripts/evidence_packs/verify_pack.sh`: verify an evidence pack against the
  core public contracts.

Each family records owner, purpose, stability, audience, expected runtime,
network/GPU needs, and known callers. The audit expands those family entries to
one row per file:

```bash
python scripts/check_scripts_inventory.py --json
```

The JSON payload includes `referenced_by` and `unreferenced` fields based on
Makefile, GitHub workflow, docs, and test references. Unreferenced files are not
deleted automatically; use the list to mark deprecations or consolidate helpers
around stable entry points.

Current top-level files are limited to this README, the inventory, the inventory
checker, and `select_workspace_python.sh`. Run
`make scripts-inventory-check` after adding, moving, or deleting anything under
`scripts/`; run `make scripts-audit` when reorganizing the tree. New broad
workflow code must go under an existing family subdirectory unless it is the
inventory checker itself.

`make architecture-fragmentation-check` enforces the category-based policy in
`contracts/architecture_policy.toml` across production source, operational
Python and shell, and tests. Category soft limits identify review hotspots;
hard limits block release. The checker also measures Python function span and
deterministic AST complexity, rejects import-only facades and tiny production
owners, limits direct files per package, and enforces library dependency
direction. These rules apply consistently by category across the current
module layout.

Protocol and constants modules may be declared as contract owners by semantic
path pattern. They must remain declaration/constant-only; runtime logic fails
the contract-owner check. This is an ownership overlay, not a separate size
category, so contract files continue to count in the same source-package
concentration budget.

The policy has no generated-code exclusion. Python and shell files under its
governed roots are checked like maintained code, so a declaration cannot create
an unverified bypass. Uncategorized governed files fail the policy instead of
silently escaping it.

Temporary hard-limit exceptions belong only in
`contracts/architecture_debt.toml`. Each debt entry identifies one deterministic
finding key and records a ceiling, owner, reason, and expiry. Debt is diagnostic
only: it never suppresses the original release blocker. New, expired, duplicate,
malformed, regressed, and stale-ceiling entries also fail the check. Expiry is
effective at the start of the recorded date. The release ledger must remain
empty.
