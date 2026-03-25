# InvarLock – Changelog

All notable changes to the InvarLock framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added an offline release-verification bundle generator and reference docs for
  auditing release artifacts without network access.
- Added public model-family and runtime-manifest contracts, packaged contract
  artifacts in wheels, and contract-sync automation for shipped distributions.
- Added stronger proof-pack manifest and attestation tooling, package-native
  proof-pack verification, and new proof-pack `inspect` / `build` command
  flows for packaged verification artifacts.
- Added replacement-model support lanes, pilot presets, and automated model
  evidence-sweep tooling/workflows for maintaining shipped support claims.

### Changed
- Simplified the public CLI contract around `evaluate`, `verify`, `report`,
  `doctor`, and `advanced`; proof-pack, policy, plugin, and calibration flows
  now live under the `advanced` namespace, and core trusted-host evaluation now
  uses `--mode local`.
- Replaced the hidden proof-pack `_run` shim with a repo-only Python config
  runner backed by a shared internal config-execution API, so proof-pack and
  calibration internals no longer depend on a shadow CLI command surface.
- Tightened evaluate/verify isolation so generated configs stay invocation
  local and policy/coverage recomputation remains aligned with current runs.
- Moved runtime and repo workflows to secure-by-default behavior, including
  safer runtime-image resolution/container defaults and tighter integration
  protections around Dependabot activity.
- Optimized evaluation data loading, Hugging Face adapter/model-loading paths,
  model-profile resolution, and CLI run/bootstrap startup flows to reduce local
  evaluation overhead.
- Refactored grouped `make test-*` targets to share a single recipe body, and
  `make verify` now includes `make runtime-verify` so the Rust runtime-manifest
  verifier is exercised as part of the main verification gate.
- Pinned workflow and proof-pack helper dependencies into checked-in
  requirements files, and updated CI/release automation to run against the
  configured `setup-python` interpreter with tighter permission scopes.
- Hardened the exhaustive CLI smoke runner, expanded active eval coverage
  thresholds, and retargeted Dependabot automation to `staging/next`.
- Refreshed shipped model lanes and presets around evidence-backed support,
  including `hf_text` causal-eval defaults, updated pilot/backlog family
  coverage, and removal of the legacy ONNX adapter surface.
- Simplified the human-readable Markdown evaluation report by folding the
  dashboard into a single Executive Summary section and removing the
  hand-maintained contents block.

### Fixed
- Hardened CLI backend, doctor, plugin, and verification checks, including
  safer remote-code defaults, plugin catalog/install surfaces, and
  release-profile overhead enforcement.
- Fixed the CLI runtime-verifier test shim to use the active test interpreter,
  which keeps nested verify/proof-pack attestation tests aligned with the
  installed Python environment.
- Tightened core profiling, security, typing, report-type validation, and local
  model-profile resolution behavior.
- Proof-pack scenario, staging, and shell execution flows now honor one-sided
  manifests, pin helper installs, normalize sparse YAML/JSON staged presets,
  use the active Python interpreter, keep no-`jq` paths deterministic, and
  remain portable across hosts.
- Proof-pack remote/bulk-run and replay flows now fail fast on missing
  `INVARLOCK_ALLOW_REMOTE_CODE`, default to eager attention plus copied
  baselines for secure-default remote runs, keep bounded queues authoritative,
  log the effective runtime mode, reuse generated checkpoints, and keep
  maintained sentinel lanes aligned with the actual evaluated window plan.
- Secure-default runtime delegation now mounts absolute preset, baseline,
  subject, model, and output paths, passes CUDA GPUs through, preserves
  delegated reports written outside the repo mount, and mounts external
  symlink targets needed by local-checkpoint flows.
- Fixed per-file coverage enforcement to include the full thresholded surface
  in generated coverage reports, and ratcheted additional CLI/core/reporting
  branch floors to 95% and 100% where the current suite now supports them.
- Fixed secure-default direct `invarlock evaluate` to mount absolute
  `--preset` and `--baseline-report` paths, and updated the maintained Qwen2.5
  14B sentinels to stage and normalize their evaluate inputs against the saved
  baseline schedule before replaying saved-model checks.
- Aligned the Scorecards workflow with upstream pinning, tightened
  Scorecards/CodeQL permissions, and fixed notebook ordering needed for release
  pre-commit validation.
- Repaired local runtime evaluation and security-default quickstart flows so
  repo checkouts prefer a locally built runtime image, respect the runtime
  container entrypoint, and document the current `plugins list` CLI form.
- Fixed remote-evidence launcher Python discovery and aligned proof-pack nested
  verification expectations with the packaged verifier behavior.
- Hardened ClusterFuzzLite/runtime security integration and policy-pack digest
  verification in fail-closed paths.
- Fixed container-backed model evidence sweeps and exported checkpoint flows to
  use container-safe preset/report paths, publish generated artifacts back to
  the requested host output root, and save tokenizer assets alongside edited
  model weights for local reruns.
- Fixed Markdown report rendering for schema-valid reports that omit
  `artifacts.generated_at`, and suppressed empty window-plan placeholders in
  first-screen summaries.

### Removed
- Removed the `QwQ-32B` model lane from the repo, including its maintained
  catalog/support references and its shipped preset and calibration configs.

### Dependencies
- Bumped `actions/download-artifact` from `7` to `8`.
- Bumped `actions/upload-artifact` from `5` to `7`.
- Bumped `katex` from `0.16.28` to `0.16.33`.

### Documentation
- Rewrote the public onboarding flow around `evaluate` → `verify` →
  `report html`, moved advanced command guidance behind the `advanced`
  namespace, and added migration notes for the simplified CLI surface.
- Added a release-verification guide covering the new offline bundle flow and
  refreshed related security best-practice references.
- Clarified proof-pack wheel-boundary, scenario, and verification guidance, and
  refreshed related CLI, contracts, and adapter reference material.
- Documented the maintained Qwen2.5-14B proof-pack sentinels, fresh-worktree
  remote guidance, and the new secure-default proof-pack bulk-run defaults.
- Updated report-reading/reference docs to match the streamlined Executive
  Summary-first Markdown report layout.
- Added live execution verification for runnable Markdown examples in the
  maintainer docs workflow, documented the new `docs-live` path plus
  runtime-image prerequisites for repo quickstarts, and kept hosted docs CI on
  the non-live validation path.
- Documented proof-pack wheel verification and the nongated replacement backlog
  lanes used for evidence-backed model support planning.

## [0.4.0] - 2026-03-14

### Added
- Published stable public contracts for support matrices, adapter capabilities,
  plugin compatibility, proof-pack manifests, and policy packs, along with new
  CLI policy tooling and shipped public evidence fixtures for published-basis
  lanes.
- Added dataset and RMT provenance to evaluation reports and expanded
  claim-surface consistency checks across docs and verification flows.

### Changed
- Refactored the trust-critical `verify`, runner, variance, and spectral paths
  into thinner orchestration shells with split helper modules and stronger
  per-file coverage thresholds.
- Raised the project-wide coverage floor to 90%, expanded the enforced critical
  surface, and reorganized CLI/core/eval/reporting tests around behavior-based
  layouts and names.
- Default local tooling now resolves to Python 3.12, docs linting includes
  spellcheck, and repo housekeeping/ignore rules were tightened around generated
  outputs and shipped fixtures.
- Docs-only CI now runs on `staging/next` and `main`, with markdown and
  spellcheck lint enforced as blocking checks instead of advisory-only steps.
- Removed legacy CLI/reporting/config surfaces and dropped legacy proof-pack
  layout compatibility.

### Fixed
- Enforced verify-policy parity and preserved guard-contract parity when runs
  reuse or compare baseline evidence.
- Repaired quickstart evaluation flows, smoke helper runners, and plugin JSON
  listing so lightweight command paths no longer instantiate dataset providers.
- Repaired the trusted-publishing workflow pin, enabled idempotent reruns for
  existing version uploads, and updated GitHub Release bundling to accept the
  current Sigstore JSON signing artifacts used by the `v0.4.0` pipeline.
- Hardened proof-pack verification to reject stray JSON outputs, use portable
  UTC helpers, and keep pack verification behavior fail-closed.
- Preserved tiny-relax provenance, MLM telemetry, and verify drift parity
  across reporting and CLI flows, and skipped overhead gating during
  calibration sweep runs where appropriate.
- Suppressed known benign GPT-2-style Hugging Face load-report noise while
  preserving actionable missing, unexpected, and mismatched checkpoint warnings.
- Cleared the remaining CodeQL backlog and completed the current OpenSSF
  hardening pass.

### Dependencies
- Added docs spellcheck tooling and pinned repo formatter/build tooling for
  reproducible local and CI verification.
- Bumped GitHub `actions/cache` to v5.

### Documentation
- Renamed and tightened assurance notes, narrowed the public claim surface, and
  expanded reference docs for contracts, calibration, proof packs, and policy
  provenance.
- Refreshed README and test/example wording to match the stabilized
  evaluate/report/verify contract and current repo structure.
- Updated public docs to describe the canonical five-stage guard chain,
  including the terminal invariants pass shown by current CLI output.

## [0.3.12] - 2026-02-27

### Added
- Coverage thresholds now enforce split-module branch floors for critical CLI/reporting paths.

### Changed
- Refactored CLI run/report builder flows into smaller modules and injected explicit run-command dependencies.
- Tightened exception-hygiene handling across `run`, `report`, and `doctor` command paths.
- Repository housekeeping now excludes research pipeline artifacts from tracked source files.

### Fixed
- Hardened config include resolution and plugin subprocess path handling in CLI flows.
- Normalized doctor/plugin command exit semantics for stable profile-specific failure behavior.
- Strengthened reporting fail-closed schema behavior with network refcounting and schema patch hardening.
- Hardened overhead/tiny-relax guard handling and config/profile gate-control enforcement.
- Made observability alerting import-safe when `requests` is unavailable.
- Hardened docs command runner security checks and enforced pip-audit execution.

### Dependencies
- Bumped `katex` from `0.16.27` to `0.16.28`.
- Bumped `markdownlint-cli2` from `0.20.0` to `0.21.0`.

### Documentation
- Replaced remaining certification wording with evaluation terminology in docs.
- Clarified calibration policy/preset guidance and aligned ASCII diagram connector formatting.

## [0.3.11] - 2026-02-12

### Added
- Added targeted regression coverage for quantization clipping, spectral guard branches, and report-schema edge cases.

### Changed
- Plugin detection flow updated to detect AWQ support through the lightweight `awq` module path.
- Spectral guard handling updated to treat `gate_proj` as an FFN projection in gating paths.

### Fixed
- CLI plugin listing avoids importing AWQ at discovery time.
- Reporting schema accepts nullable dataset window seeds and structured `system_overhead` payloads.
- Quantization RTN outlier clipping path is hardened for fp16-safe behavior.

### Documentation
- Release notes and metadata updated for `v0.3.11`.

## [0.3.10] - 2026-02-08

### Added
- Proof packs: new guard showcase suite and expanded scenario coverage (scenario filtering/errors-only mode, suite-scoped scenarios, and model override support).
- Proof packs: new demo/probing artifacts (verdict tables generator, VE `ve_probe` sidecar, and additional RMT/spectral/variance showcase injections).
- CI: add Python 3.12 smoke and scheduled weekly verification.

### Changed
- CI: make release/CI verification more reproducible (deterministic `verify-full`) and improve local `act` ergonomics.
- Docs CI: allow on-demand runs via `workflow_dispatch`.
- Proof packs: strengthen “evidence signal” outputs and tighten fail-closed behavior for verdict/task failures.

### Fixed
- Guards/variance and VE: improve Mixture-of-Experts compatibility (fused expert weight layouts, broader VE layer discovery, and Mixtral `block_sparse_moe` support) and harden variance defaults/probes.
- Proof packs: improve reliability and determinism of demos (retuned injections/detectors, more robust packaging of probe sidecars, and safer behavior when reports exist but evaluation exits nonzero).
- Assurance: close verification/baseline evidence gaps and tighten audit coverage.
- CLI/eval/tests: stabilize CI help-smoke output, accept extra `load_dataset` kwargs, and allow warn-only determinism.

### Dependencies
- Proof packs: harden dependency preflight and net-enabled install behavior (require `huggingface_hub` where needed; ensure `accelerate` is available).

### Documentation
- Docs: fix markdown link fragments.
- Proof packs: clarify evidence vs proof-grade posture and document new artifacts (intervention summary + VE probe sidecar).

## [0.3.9] - 2026-02-03

### Fixed
- CI: update workflow test paths after the report/certificate rename.
- Tests: apply ruff-format to warning suppression coverage test.
- CLI: `invarlock report explain` drift gate now prints the resolved drift band (no hard-coded threshold).
- CLI: align `invarlock report` “ARTIFACTS” block so artifact paths start in the same column.
- Observability: CPU health check no longer fails when platform CPU count is unavailable.
- Proof packs: config generator can emit configs to stdout without relying on `/dev/stdout`.
- Tests: stabilize the end-to-end pipeline memory management integration test with a PyTorch warm-up.
- Tests: build-wheel packaging test uses `build --no-isolation` to avoid network in offline environments.
- Tests: import-safety venv integration test skips cleanly when network is unavailable.

### Documentation
- README: refresh above-the-fold header layout, including a banner-sized logo lockup and centered badges.
- Branding: make the README logo lockup more logomark-dominant and add a dark-mode logo variant.
- Branding: logomark-only avatar asset (`docs/assets/invarlock-mark.svg`) for GitHub profile usage.

## [0.3.8] - 2026-02-02

### Added
- CLI: `--version` / `-V` flag (alias of `invarlock version`) to print the InvarLock version (plus report schema version when available).
- `invarlock evaluate` summary now includes total runtime and confidence interval.
- Proof packs: `verify_pack.sh --strict` (or `PACK_STRICT_MODE=1`) to fail closed on missing/invalid GPG signatures and unexpected pack contents.

### Changed
- **Breaking:** Rename “certificate” → “report” across artifacts, docs, scripts, notebooks, and Python API surfaces.
- **Breaking:** CLI terminology unified on `evaluate` (replaces `certify`).
- Config: reject legacy HF v4 load keys `model.torch_dtype`, `model.load_in_8bit`, and `model.load_in_4bit`; use `model.dtype` and/or `model.quantization_config`.
- Evaluation report bundle filenames updated (JSON: `evaluation.report.json`, Markdown: `evaluation_report.md`).
- Presets: bump default WikiText-2 dataset seed for the causal LM preset from `42` → `43`.
- Proof packs: `manifest.json` records `checksums_sha256_digest` (sha256 of `checksums.sha256`) and may record `signing_key_fingerprint` when signed.

### Fixed
- HuggingFace/Transformers v5 compatibility: migrate load contracts and use `dtype=` where required.
- Reduce noisy HuggingFace/Transformers warnings in `ci`/`release` CLI output.
- Adapters: snapshot config serialization no longer emits deprecated attributes.
- Scripts: CLI example validator ignores internal tool dirs and supports external paths.
- CLI: keep `invarlock calibrate` import-safe so docs/example validation can run without torch installed.
- Proof packs: fix `verify_pack.sh` cert discovery to verify `certs/**/evaluation.report.json`.
- Proof packs: close a tamper-evidence gap by binding `checksums.sha256` to the signed manifest (and enforcing “no extra files” in strict verification).

### Dependencies
- Require `transformers>=5.0.0` and `huggingface_hub>=1.0.0`.

### Documentation
- Update guides and notebooks for evaluation reports and renamed commands/pages.
- README: add logo, community links, citation snippet, limitations, and quickstart output excerpt.
- Drop legacy Transformers v4 config key documentation and fix minor formatting/typos.

## [0.3.7] - 2026-01-22

### Added
- Role-based HuggingFace adapters with updated auto-routing (replaces model-name adapters).
- Proof packs: v2 pack layout, scenarios manifest, and assurance verdict generation.
- CLI flags: `invarlock run --edit-label` and `invarlock evaluate --baseline-report`.
- CI notebook smoke runner (`scripts/verify_notebooks_smoke.py`).

### Changed
- Proof pack workflows hardened: baseline-report reuse, calibrate-only behavior, tuned-params hygiene, and improved task sizing/memory planning.
- report reporting refreshed: revamped report markdown, enhanced HTML output + glossary, and “Safety report” renamed to “Evaluation report”.
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
- Measurement contracts for guard estimators (approximation-only, GPU/MPS-first) recorded in reports and enforced by `invarlock verify --profile ci|release`.
- Proof pack suite workflow split: `scripts/proof_packs/run_suite.sh --calibrate-only` (stop after preset generation) and `--run-only` (resume remaining tasks).
- Proof pack suite knob for controlled experiments: `PACK_GUARDS_ORDER`.

### Changed
- B200 calibration configs now default to `guards.order: [invariants, variance, invariants]` (drops spectral/rmt) to avoid CPU-bound SVD (`torch.linalg.svdvals` / MKL `sgesdd`) dominating wall time and making GPUs appear idle during calibration.
- B200 calibrated presets now include `guards.order`, and only include `guards.spectral` / `guards.rmt` sections when those guards are enabled (run a smaller follow-up calibration pass if you need spectral caps or an RMT ε).
- B200 bootstrap defaults HuggingFace caches under `${OUTPUT_DIR}/.hf` (override with `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE`) to avoid small `/root` partitions on GPU nodes.
- `invarlock evaluate` now honors `guards.order` when provided by `--preset` (instead of always forcing `["invariants", "spectral", "rmt", "variance", "invariants"]`), so evaluate matches the calibration preset’s intended guard set.

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
- CI/Release report generation now requires `paired_windows` evidence and rejects non-perfect window pairing.

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
  CI/Release reports require perfect pairing, non-overlapping windows, and coverage floors.
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
- Bench policy regression harness and additional regression tests for guards and reports.
- Benchmark policy regression golden `bench-golden-2025-12-13` (`2627b8872cd6bfc37bda31fbc11b78ed814751cbf2a9ad1396e173f1f4e5383a`) tracked to guard guard-effect CI against silent gate/output shifts.

### Changed
- Guard policies and tier runtime configuration updated to support calibration and determinism flows.
- CLI commands (`run`, `verify`, `doctor`, `explain-gates`) extended with calibration and reporting surfaces.

### Fixed
- Additional edge cases in report reporting, policy utilities, and guard analysis covered and hardened via new tests.

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
- Core compare & evaluate pipeline and guard chain for edit‑agnostic robustness reports.
- Evaluation report schema v1 and CLI entry points (including `invarlock evaluate`).
- Torch‑optional core install with optional extras (e.g., `invarlock[hf]`, `invarlock[adapters]`).
- Initial documentation set: quickstart, user guides, and CLI reference.

### Notes
- 0.2.0 is the first public version of the InvarLock framework.
- Until 1.0.0, **minor** releases (0.x.y → 0.(x+1).0) may include breaking changes. Refer to the README and CLI help for the current surface and behavior.
