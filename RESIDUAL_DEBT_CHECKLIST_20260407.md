# Residual Static Debt Checklist

Date: 2026-04-07
Base branch: `staging/next @ 41916a5`
Execution branch: `work/residual-debt-audit-20260407`

## Scope

This checklist covers every residual static-debt item called out in the April 7 follow-up audit:

- remaining broad-catch hotspots:
  - `src/invarlock/core/runner_eval_phase.py`
  - `src/invarlock/cli/app.py`
  - `src/invarlock/cli/commands/export_html.py`
  - `src/invarlock/cli/overhead_utils.py`
- largest remaining structural owners:
  - `src/invarlock/core/run_orchestrator_execute.py`
  - `src/invarlock/core/runner_eval_metrics.py`
  - `src/invarlock/runtime_security_helpers.py`
  - `src/invarlock/adapters/hf_mlm.py`
  - `src/invarlock/proof_pack.py`

## Acceptance Criteria

- [x] The remaining broad `except Exception` hotspots in the four boundary files are narrowed to explicit exception families or helper paths with focused regression coverage.
- [x] Each large-owner file receives at least one bounded complexity reduction that extracts a coherent helper seam without regressing behavior.
- [x] Architecture guardrails or focused regression tests are updated where needed so the same debt cannot silently reappear.
- [x] `make lint`, `make verify`, and `make coverage-enforce` pass on the work branch.
- [ ] The work branch is merged back into `staging/next`, `staging/next` reruns repo gates cleanly, and the result is pushed.

## Checklist

### Phase 1: Broad-Catch Hotspots

- [x] `runner_eval_phase.py`: extract the debug-trace calibration snapshot path into typed helpers and replace blanket defensive catches with explicit index/shape/runtime error families.
- [x] `app.py`: narrow `_emit_version()` fallback handling so package-metadata, schema import, and package import failures are caught explicitly instead of with blanket `Exception`.
- [x] `export_html.py`: narrow read/render/write failure handling to explicit JSON, import, and filesystem error families while preserving current exit-code behavior.
- [x] `overhead_utils.py`: narrow the three primary-metric extraction fallback catches to explicit report-shape / import / compute error families.

### Phase 2: Structural Complexity

- [x] `run_orchestrator_execute.py`: extract at least one coherent setup/execution/finalization seam out of `execute_run_request_impl()` to reduce local ownership concentration.
- [x] `runner_eval_metrics.py`: extract at least one metric-dispatch / normalization seam out of `compute_real_metrics()` to reduce the single giant function surface.
- [x] `runtime_security_helpers.py`: extract at least one command/manifest assembly seam to reduce mixed responsibilities inside the runtime security helper surface.
- [x] `hf_mlm.py`: extract at least one model-structure probing seam from `can_handle()` / `describe()` to reduce wrapper/direct-encoder branching density.
- [x] `proof_pack.py`: extract at least one proof-pack build/verify seam so `build_proof_pack()` / `verify_proof_pack()` own less orchestration directly.

### Phase 3: Guardrails + Closeout

- [x] Update [tests/lint/test_architecture_guardrails.py](/Users/ospc/Documents/Projects/invarlock-public/tests/lint/test_architecture_guardrails.py) if new hardened files should be budgeted.
- [x] Add focused regression coverage for each changed seam; avoid mock-only tests for critical logic.
- [x] Run `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make lint`.
- [x] Run `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make verify`.
- [x] Run `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make coverage-enforce`.
- [ ] Commit the completed work to `work/residual-debt-audit-20260407`.
- [ ] Merge `work/residual-debt-audit-20260407` into `staging/next`.
- [ ] Re-run repo gates on `staging/next` and push the result.

## Validation Matrix

- Focused boundary tests:
  - `PYTHONPATH=src /Users/ospc/anaconda3/envs/invarlock-py312/bin/python -m pytest tests/core/test_runner_eval_phase_tail_evidence.py tests/core/test_runner_context_and_debug_trace.py tests/cli/test_app.py tests/cli/test_app_version.py tests/cli/test_export_html.py tests/cli/test_export_html_cases.py tests/cli/test_export_html_io_errors.py tests/unit/test_overhead_extraction.py`
- Focused large-owner tests:
  - `PYTHONPATH=src /Users/ospc/anaconda3/envs/invarlock-py312/bin/python -m pytest tests/core/test_run_orchestrator_split_paths.py tests/core/test_run_orchestrator_paths.py tests/core/test_runner_eval_metrics_paths.py tests/unit/test_runtime_security_core.py tests/unit/test_runtime_security_container.py tests/unit/test_runtime_security_manifest.py tests/adapters/test_adapter_errors.py tests/adapters/test_adapters.py tests/reporting/test_proof_pack_verify_paths.py tests/reporting/test_proof_pack_split_modules.py`
- Full gates:
  - `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make lint`
  - `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make verify`
  - `PYTHON=/Users/ospc/anaconda3/envs/invarlock-py312/bin/python make coverage-enforce`
