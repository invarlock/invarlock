---
name: "Chore: Ruff lint sweep (tests + scripts)"
about: Fix ruff violations across tests/ and scripts/
labels: chore, lint, tests
assignees: ''
---

Summary
- Perform a targeted ruff lint sweep across tests/ and scripts/ to make `make verify` green without disabling rules.

Scope
- Include: `tests/`, `scripts/`
- Exclude: `src/` (separately tracked), heavy refactors, behavior changes

Why
- `make verify` currently fails on ruff with ~300 findings, mostly style issues in tests and a few scripts. Fixing them stabilizes CI and reduces noise for future changes.

Rules Observed (from latest run)
- UP038 Use `X | Y` in `isinstance` (autofixable)
  - scripts/check_device_drift.py:23,29,33
  - scripts/golden_runs.py:193,249,297
  - tests/utils/pm.py:8
  - tests/reporting/test_certificate_analysis_ci.py:61
  - tests/reporting/test_pm_ratio_from_primary_metric.py:39
  - tests/reporting/test_pm_ratio_identity_fallback.py:52
- F601 Dictionary key literal repeated (manual)
  - tests/reporting/test_certificate_primary_metric.py (duplicate "plugins")
  - tests/reporting/test_certificate_primary_metric_multimodal.py (duplicate "plugins")
- F841 Local variable assigned but never used (manual)
  - tests/reporting/test_evidence_bundle.py: saved
  - tests/reporting/test_certificate_schema_v1_accuracy_tags.py: pm
- B017 Do not assert blind exception (manual)
  - tests/reporting/test_certificate_schema_strict_validation.py: use jsonschema.ValidationError
- C405 Unnecessary list literal (rewrite as set literal) (autofixable)
  - tests/reporting/test_certificate_schema_v2_confidence.py:64

Acceptance Criteria
- `make verify` passes locally (tests + smoke + ruff).
- No behavior changes to tests beyond lint fixes (keep assertions intact).

Proposed Steps
1) Autofix safe rules
   - Run: `python -m ruff check tests scripts --select UP038,C405 --fix`
2) Manual fixes
   - Remove duplicate dict key literals (F601) in the two reporting tests.
   - Remove unused vars (F841) or use a wildcard underscore.
   - Replace blind `pytest.raises(Exception)` with `pytest.raises(jsonschema.ValidationError)`.
3) Re-run: `make verify` and adjust any remaining stragglers.

Notes
- Keep edits surgical; avoid widening rule ignores. If a rule cannot be reasonably satisfied without behavior risk, discuss before suppressing.
