# Scripts Directory

`scripts/` contains repo-maintainer tooling, not install-time package APIs. Treat
top-level paths that appear in `Makefile`, `.github/workflows/`, or published docs
as compatibility entry points.

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

Run `make scripts-inventory-check` after adding, moving, or deleting anything
under `scripts/`; run `make scripts-audit` when reorganizing the tree. New broad
workflow code should go under an existing family subdirectory when possible; keep
a top-level compatibility wrapper only when a documented command or CI workflow
already depends on that path.
