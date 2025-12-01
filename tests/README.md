# Tests

## Markers

- `integration`: uses external tools, larger datasets, or slower paths.
- `slow`: long-running tests; usually skipped in default local runs.
- `manual`: requires human inspection or environment not present in CI.

The entire `tests/integration/` subtree is auto-marked as `integration`
via `tests/integration/conftest.py`.

## Typical invocations

Run fast/unit tests:

```
INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1 \
pytest -q -m "not integration and not slow and not manual" tests
```

Run the curated CI subset locally:

```
INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1 \
pytest -q \
  tests/cli/test_python_m_invarlock.py \
  tests/cli/test_report_help_and_html.py \
  tests/cli/test_doctor_json.py \
  tests/cli/test_doctor_cross_checks.py \
  tests/cli/test_doctor_json_cls_pseudo_counts.py \
  tests/cli/test_doctor_json_cls_measured_no_d012.py \
  tests/cli/test_doctor_json_tiny_relax_note.py \
  tests/integration/scripts/test_tiny_matrix_checklist.py \
  tests/reporting/test_certificate_schema_v1_accuracy_tags.py \
  tests/reporting/test_cert_markdown_estimated_suffix.py \
  tests/reporting/test_cert_markdown_no_estimated_for_measured.py
```

## Runtime artifacts

- CLI commands and scripts write run artifacts under `runs/` and certificates under `reports/` at the repository (or working directory) root.
- Test fixtures should live under `tests/fixtures` (or per-area test dirs), not under `tests/runs` or `tests/reports`.
