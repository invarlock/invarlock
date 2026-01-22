# InvarLock – Changelog

All notable changes to the InvarLock framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.7] - 2026-01-22

### Added
- Role-based HuggingFace adapters with updated auto-routing (replaces model-name adapters).
- Proof packs: v2 pack layout, scenarios manifest, and assurance verdict generation.
- CLI flags: `invarlock run --edit-label` and `invarlock certify --baseline-report`.
- CI notebook smoke runner (`scripts/verify_notebooks_smoke.py`).

### Changed
- Proof pack workflows hardened: baseline-report reuse, calibrate-only behavior, tuned-params hygiene, and improved task sizing/memory planning.
- Certificate reporting refreshed: revamped certificate markdown, enhanced HTML output + glossary, and “Safety Certificate” renamed to “Evaluation Certificate”.
- Presets/overlays updated for new adapter roles and additional model families.
- CI: bump `actions/download-artifact` to v7; remove the legacy B200 backend validation harness.

### Fixed
- Adapters: Mixtral support, improved auto-detection, and hardened causal describe/weight tying.
- Proof packs: enforce CI floor constraints, mitigate OOM/missing-tensors cases, and make verification more resilient.
- Reporting/eval: avoid duplicate synthetic samples and preserve primary-metric drift band handling.

### Documentation
- Expanded and consolidated guides across CLI, configs, datasets, guards, proof packs, and notebooks.

## [0.3.6] - 2026-01-13

### Added
- Measurement contracts for guard estimators (approximation-only, GPU/MPS-first) recorded in certificates and enforced by `invarlock verify --profile ci|release`.
- Proof pack suite workflow split: `scripts/proof_packs/run_suite.sh --calibrate-only` (stop after preset generation) and `--run-only` (resume remaining tasks).
- Proof pack suite knob for controlled experiments: `PACK_GUARDS_ORDER`.

### Changed
- B200 calibration configs now default to `guards.order: [invariants, variance, invariants]` (drops spectral/rmt) to avoid CPU-bound SVD (`torch.linalg.svdvals` / MKL `sgesdd`) dominating wall time and making GPUs appear idle during calibration.
- B200 calibrated presets now include `guards.order`, and only include `guards.spectral` / `guards.rmt` sections when those guards are enabled (run a smaller follow-up calibration pass if you need spectral caps or an RMT ε).
- B200 bootstrap defaults HuggingFace caches under `${OUTPUT_DIR}/.hf` (override with `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE` / `TRANSFORMERS_CACHE`) to avoid small `/root` partitions on GPU nodes.
- `invarlock certify` now honors `guards.order` when provided by `--preset` (instead of always forcing `["invariants", "spectral", "rmt", "variance", "invariants"]`), so certify matches the calibration preset’s intended guard set.

### Dependencies
- Bump katex from 0.16.25 to 0.16.27.
- Bump markdownlint-cli2 from 0.19.1 to 0.20.0.

## [0.3.5] - 2026-01-02

### Added
- Proof pack bash test suite (`scripts/proof_packs/tests/*`, `scripts/proof_packs/tests/run.sh`) with deterministic command mocks and optional branch/line coverage checks.
- Proof pack runtime helpers (`scripts/proof_packs/lib/runtime.sh`) plus pack build/verify helpers (`scripts/proof_packs/run_pack.sh`, `scripts/proof_packs/verify_pack.sh`) to capture artifacts during long runs.
- Perplexity token-id sanitization to mask out-of-range IDs (and ignore them in labels) instead of triggering device-side asserts.

### Changed
- WikiText-2 window stratification now uses a deterministic offline byte-level n-gram scorer (replaces the GPT‑2 scorer) to keep window selection stable across model families and avoid implicit model downloads.
- B200 validation suite is dynamic-scheduling only; dependency promotion is centralized to reduce queue lock contention and improve throughput.
- B200 generated configs default to `guards.order: [invariants, rmt, variance]` to avoid slow CPU SVD during calibration; spectral caps are not produced unless you re-enable spectral calibration separately.
- B200 bootstrap defaults HuggingFace caches under `${WORK_DIR}/hf_home` to avoid small `/root` partitions on GPU nodes.

### Fixed
- B200 harness: treat 30B+ models as “large” for overhead-skip heuristics to avoid double-loading stalls.

### Removed
- `INVARLOCK_SCORES_BATCH_SIZE` (the WikiText‑2 difficulty scorer no longer batches on device).

### Documentation
- Updated CLI/dataset/env-var references for the new difficulty scorer and removal of `INVARLOCK_SCORES_BATCH_SIZE`.

## [0.3.4] - 2025-12-28

### Added
- Chunked snapshot/restore support for HF adapters to reduce peak memory during retries.
- Proof pack workflow helpers (run_suite + scheduler/queue utilities + model creation tooling).

### Changed
- CI/Release baseline pairing is fail-closed: `invarlock run --baseline ...` now requires valid `evaluation_windows` evidence and enforces dataset/tokenizer/masking parity.
- CI/Release certificate generation now requires `paired_windows` evidence and rejects non-perfect window pairing.

### Documentation
- Updated artifacts, CLI, and environment variable references for snapshot fallback and baseline pairing requirements.

## [0.3.3] - 2025-12-21

### Added
- Token-weighted paired Δlog-loss bootstrap support (core bootstrap + primary metric + variance guard).
- New strictness/override toggles: `INVARLOCK_EVAL_STRICT`, `INVARLOCK_GUARD_PREPARE_STRICT`,
  `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE`, `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE`.
- RMT activation helper paths for outlier collection and activation-required guard flows.
- Report metadata for guard prepare failures and evaluation soft-fail context (`metrics.eval_error`).

### Changed
- Window pairing enforcement now tracks overlap vs duplicate fractions and detects count mismatches;
  CI/Release certificates require perfect pairing, non-overlapping windows, and coverage floors.
- Determinism preset chooses `CUBLAS_WORKSPACE_CONFIG` based on GPU memory and disables
  `TOKENIZERS_PARALLELISM` under strict settings.
- Guard overhead metric fields standardized to `bare_ppl`/`guarded_ppl`; primary metric `display_ci`
  is aligned with log-space CI for ppl-like metrics.
- B200 validation workflow upgraded to v2.1.0 with dynamic scheduling, GPU lock management,
  and expanded task orchestration scripts.

### Fixed
- Calibration data slicing now supports iterables with optional materialization and clearer errors.
- Sequence hashing now includes per-sequence lengths to avoid ambiguous digests.
- Variance guard predictive gating improves min-effect and regression reasoning.

### Documentation
- Expanded B200 validation guide with v2.1.0 workflow details and scheduler/queue notes.
- Assurance docs, CLI guidance, and environment variable references refreshed for new behavior.

## [0.3.2] - 2025-12-14

### Added
- Calibration CLI (`invarlock calibrate`) and runtime modules for policy and guard tuning.
- Determinism utilities and CLI flows to exercise repeatable runs and presets.
- Bench policy regression harness and additional regression tests for guards and certificates.
- Benchmark policy regression golden `bench-golden-2025-12-13` (`2627b8872cd6bfc37bda31fbc11b78ed814751cbf2a9ad1396e173f1f4e5383a`) tracked to guard guard-effect CI against silent gate/output shifts.

### Changed
- Guard policies and tier runtime configuration updated to support calibration and determinism flows.
- CLI commands (`run`, `verify`, `doctor`, `explain-gates`) extended with calibration and reporting surfaces.

### Fixed
- Additional edge cases in certificate reporting, policy utilities, and guard analysis covered and hardened via new tests.

### Documentation
- Expanded assurance docs for calibration, guard contracts, determinism, and BCA/bootstrap methods.

## [0.3.1] - 2025-12-10

### Fixed
- **Memory leak in run.py reload fallback** - GPU memory is now freed before reloading models, preventing OOM on 70B+ runs.
- **B200 validation script bugs** - Fixed preset path resolution, model size detection, and error propagation in dynamic scheduling workers.

### Added
- **INVARLOCK_SKIP_OVERHEAD_CHECK env var** - Skip guard overhead measurement even with ci/release profiles for large models.
- **Configurable PM acceptance range** - Set via preset config or `INVARLOCK_PM_ACCEPTANCE_MIN/MAX` environment variables.
- **Comprehensive proof pack guide** - New documentation at `docs/user-guide/proof-packs.md`.

### Changed
- B200 validation scripts updated to v2.0.1 with improved cleanup traps and progress monitoring.

### Deprecated
- `INVARLOCK_TINY_RELAX` for PM acceptance - prefer `INVARLOCK_PM_ACCEPTANCE_MAX` and presets instead.

## [0.3.0] - 2025-12-05

### Added
- **Quantization-aware capabilities module** (`invarlock.adapters.capabilities`)
  - `ModelCapabilities` dataclass for declaring model properties
  - `QuantizationConfig` frozen dataclass for quantization metadata
  - `QuantizationMethod` enum (NONE, BNB_8BIT, BNB_4BIT, AWQ, GPTQ, ONNX)
  - `detect_quantization_from_config()` and `detect_capabilities_from_model()` helpers
- **Safe device movement** via `_safe_to_device()` in `HFAdapterMixin`
  - Prevents `.to()` calls on BNB/AWQ/GPTQ models that handle device placement internally
  - Fixes "`.to` is not supported for 8-bit bitsandbytes models" error
- **Pre-quantized checkpoint detection** in `hf_bnb_adapter`
  - `_detect_pre_quantized_bnb()` reads `config.json` to detect existing quantization
  - Prevents re-quantization when loading saved BNB checkpoints
- **Quantization-aware auto-adapter routing**
  - `_detect_quantization_from_path()` and `_detect_quantization_from_model()` in `auto.py`
  - Auto-routes to `hf_bnb`, `hf_awq`, or `hf_gptq` based on checkpoint metadata
- **Comprehensive adapter test coverage** (46 new tests)
  - `test_capabilities.py` - QuantizationMethod, QuantizationConfig, ModelCapabilities
  - `test_safe_device.py` - Safe device movement and capability detection
- **Observability module test coverage** (230 new tests across 6 files)
- **Test documentation** - README files for `tests/guards/` and `tests/observability/`

### Changed
- `hf_causal.py`: Uses `_safe_to_device()` instead of direct `model.to()` call
- `hf_awq_adapter.py`: Uses `_safe_to_device()` with AWQ capabilities
- `hf_gptq_adapter.py`: Uses `_safe_to_device()` with GPTQ capabilities

### Fixed
- BNB 8-bit model loading error when subject is a saved quantized checkpoint
- Empty sample handling in variance guard (`_safe_mean()` helper)

### Documentation
- Added quantized adapter section to `docs/reference/model-adapters.md`
  - BNB adapter usage and pre-quantized detection
  - AWQ adapter (Python 3.12 compatible)
  - GPTQ adapter (requires Python 3.10/3.11)
  - Quantization auto-detection flow

## [0.2.0] - 2025-12-01

First public release on GitHub and PyPI.

### Added
- Core compare & certify pipeline and guard chain for edit‑agnostic robustness certificates.
- Evaluation Certificate schema v1 and CLI entry points (including `invarlock certify`).
- Torch‑optional core install with optional extras (e.g., `invarlock[hf]`, `invarlock[adapters]`).
- Initial documentation set: quickstart, user guides, and CLI reference.

### Notes
- 0.2.0 is the first public version of the InvarLock framework.
- Until 1.0.0, **minor** releases (0.x.y → 0.(x+1).0) may include breaking changes. Refer to the README and CLI help for the current surface and behavior.
