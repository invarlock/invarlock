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
- Ruff sweep debt can accumulate across `tests/` and `scripts/`; addressing it
  keeps `make verify` green and reduces noise for future changes.

Rules Observed (from latest run)
- Capture the current `ruff check tests scripts` output when opening this task.
- Do not copy historical file names forward; several pre-report-rename paths are
  no longer valid.
- Typical rule families in this sweep:
  - `UP038` Use `X | Y` in `isinstance` (autofixable)
  - `F601` Dictionary key literal repeated (manual)
  - `F841` Local variable assigned but never used (manual)
  - `B017` Do not assert blind exception (manual)
  - `C405` Unnecessary list literal (autofixable)

Acceptance Criteria
- `make verify` passes locally (tests + smoke + ruff).
- No behavior changes to tests beyond lint fixes (keep assertions intact).

Proposed Steps
1) Autofix safe rules
   - Run: `python -m ruff check tests scripts --select UP038,C405 --fix`
2) Manual fixes
   - Fix duplicate dict key literals (F601) in the currently reported files.
   - Remove unused vars (F841) or use a wildcard underscore.
   - Replace blind `pytest.raises(Exception)` with a concrete exception type where possible.
3) Re-run: `make verify` and adjust any remaining stragglers.

Notes
- Keep edits surgical; avoid widening rule ignores. If a rule cannot be reasonably satisfied without behavior risk, discuss before suppressing.
