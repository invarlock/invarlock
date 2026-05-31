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
  points.
- `model-evidence`: model-evidence sweep planning and remote launch helpers.
- `evidence-packs`: evidence-pack orchestration, helpers, and shell tests.
- `release`: release evidence and offline-bundle helpers.
- `security-supply-chain`: SBOM, CVE audit, pip-audit, requirements pinning, and
  scorecard helpers.

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

`make architecture-fragmentation-check` also includes tracked `scripts/` metrics
for large shell files, small-file churn, evidence-pack script concentration, and
ignored generated cruft. Use those metrics when deciding whether to consolidate
helpers before public outreach.
