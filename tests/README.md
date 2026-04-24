# Tests

## Markers

- `integration`: uses external tools, larger datasets, or slower paths.
- `slow`: long-running tests; usually skipped in default local runs.
- `manual`: requires human inspection or environment not present in CI.

The entire `tests/integration/` subtree is auto-marked as `integration`
via `tests/integration/conftest.py`.

## Organization

- Keep executable tests under the directory that matches the production owner surface.
- Final executable owner dirs: `tests/adapters`, `tests/calibration`, `tests/ci`, `tests/cli`, `tests/core`, `tests/docs`, `tests/edits`, `tests/eval`, `tests/fuzzing`, `tests/guards`, `tests/integration`, `tests/lint`, `tests/observability`, `tests/plugins`, `tests/evidence_packs`, `tests/reporting`, `tests/runtime`, `tests/scripts`.
- Non-executable support and data dirs stay isolated: `tests/_stubs`, `tests/artifacts`, `tests/fixtures`, `tests/schemas`.
- `tests/artifacts` is the golden-data bucket. Keep stable report fixtures, evidence-pack payloads, and other maintainer-owned artifact snapshots there instead of folding them into `tests/fixtures`.
- Nested owner buckets are allowed when they clarify ownership. Current nested buckets:
  - `tests/guards/property`
  - `tests/guards/differential`
- Owner examples:
  - `tests/core`: orchestration, contracts, runner internals, and core policy logic.
  - `tests/cli`: command-line shells, CLI serialization, and command-facing UX behavior.
  - `tests/eval`: metrics, providers, datasets, probes, and evaluation-specific validation.
  - `tests/reporting`: report generation, normalization, rendering, validation, evidence-pack report assembly, and report-facing helper modules.
  - `tests/runtime`: runtime security, network policy, container/runtime image, and runtime-manifest verification behavior.
  - `tests/guards`: guard math, policies, runtime behavior, and guard-specific extraction logic.
- Report-generation, report-rendering, and report-validation tests belong in `tests/reporting`, not `tests/eval`, unless the test is explicitly about eval-time metric validation.
- Shared helper modules inside test areas should use a support name such as `_support_*.py` or `_internal_*.py`; avoid mixing generic helper filenames with actual test modules.
- Deprecated generic buckets such as `unit`, `api`, `packaging`, `security`, `utils`, and the legacy top-level guard property/differential buckets should not be reintroduced.

## Naming

- Prefer behavior-based names such as `test_report_builder_render.py` or `test_run_retry_and_exit.py`.
- Avoid transitional names like `additional`, `extra`, `more`, `cases`, `edgecases`, `split`, `tail`, or `part2` when a behavior-oriented name is available.
- Reserve broad `*_regression_matrix.py` files for genuinely cross-cutting edge-case matrices that do not fit a single module-focused test file.
- Prefer named imports from support modules. Wildcard imports are legacy-only and should not be used in new test files.

## Size

- Split large files by behavior before they turn into monoliths. Once a tracked test file crosses `800` LOC, split it into behavior-local siblings before adding more cases.
- If a file name needs suffixes like `split`, `tail`, or `part2`, that is usually a signal the tests should be reorganized around behavior instead of chronology.

## Typical invocations

Run fast/unit tests:

```
INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0 \
pytest -q -m "not integration and not slow and not manual" tests
```

Or use the Makefile fast lane:

```
make test-fast
```

Run the slower integration/smoke backstop separately:

```
make test-integration
```

Run the curated CI subset locally:

```
INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0 \
pytest -q \
  tests/cli/test_python_m_invarlock.py \
  tests/cli/test_report_help_and_html.py \
  tests/cli/test_doctor_json.py \
  tests/cli/test_doctor_cross_checks.py \
  tests/cli/test_doctor_json_cls_pseudo_counts.py \
  tests/cli/test_doctor_json_cls_measured_no_d012.py \
  tests/cli/test_doctor_json_tiny_relax_note.py \
  tests/integration/scripts/test_tiny_matrix_checklist.py \
  tests/reporting/test_report_schema_v1_accuracy_tags.py \
  tests/reporting/test_report_markdown_estimated_suffix.py \
  tests/reporting/test_report_markdown_no_estimated_for_measured.py
```

The curated CI subset is intentionally narrower than `make test-fast`; use it
when reproducing the fast PR gate selection or debugging that specific lane.

## Runtime artifacts

- CLI commands and scripts write run artifacts under `runs/` and evaluation reports under `reports/eval/` at the repository (or working directory) root.
- Container-backed evaluation outputs include `runtime.manifest.json` adjacent to `evaluation.report.json`; archive both when they are emitted.
- Test fixtures should live under `tests/fixtures`, `tests/artifacts`, or per-area test dirs, not under `tests/runs` or `tests/reports`.
