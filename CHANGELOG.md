# InvarLock – Changelog

All notable changes to the InvarLock framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Clarified the minimal-install onboarding path and the exact
  `verify`/`report`/proof-pack command inputs so wheel-only users can validate
  artifacts without cloning the repository.
- Added the released
  `invarlock-<version>-public-contract-bundle.tar.gz` asset, including the
  manifest/schema inventory downstream consumers can verify without cloning the
  repository.
- Added PR-time supply-chain enforcement with shipped-surface SBOM generation,
  `pip-audit` coverage for the base/`hf`/`advanced` install surfaces,
  repo-history `gitleaks` artifacts, tighter security-sensitive `CODEOWNERS`
  rules, and release-time supply-chain gates before publish.
- Added a canonical coverage-policy source plus PR-time `coverage-enforce` and
  typed-surface `mypy` gates, along with the new split owner modules they
  validate across config loading, reporting, runtime security, CLI execution,
  and orchestration.
- Added explicit `actionlint` and minimal wheel-install packaging smoke gates
  to the local make surface and CI workflow.
- Added an explicit workspace-Python selector for live smoke and telemetry
  scripts so local end-to-end wrappers prefer the repo `.venv` before generic
  interpreter fallback.

### Changed
- Tightened public docs, config comments, and CLI guardrails so the published
  OSS surface stays standalone, repo-agnostic, and aligned with the canonical
  `evaluate` / `verify` / `report` / `doctor` / `advanced` command set.
- Refactored config loading, runtime-security, evaluate execution, registry,
  and report assembly into smaller owner modules with stricter core/shell
  boundaries and no compatibility shims.
- Expanded the public contract catalog and JSON surfaces to publish
  `validation_keys`, `console_labels`, and canonical `metric_kinds`, and kept
  CLI/docs handling fail-closed when directories contain both `report.json`
  and `evaluation.report.json`.
- Simplified import surfaces so reporting and probe namespaces stay
  light-import-safe while concrete heavy implementations live behind their
  owning modules.

### Fixed
- Fixed accuracy-confidence labeling so accuracy metrics evaluate confidence
  width in true percentage points, while non-accuracy ratio metrics keep their
  ratio-width behavior.
- Fixed docs consistency CI by making the assurance cross-reference linter use
  self-contained sample reports instead of importing runtime report-building
  modules that require `numpy` under the docs-only workflow dependency set.
- Fixed the repo-owned Python selector so local `make`, smoke, and packaging
  paths all resolve through the workspace-aware selector, with the workspace
  `.venv` preferred before generic system Python fallback.
- Fixed metric-kind handling to fail closed across config resolution,
  multimodal metric propagation, report validation, and verification; unknown
  or stale aliases no longer bypass accuracy-specific checks.
- Fixed report validation and verification to fail closed when
  `validation_keys.json` is missing or malformed, reject ambiguous directories
  that contain both `report.json` and `evaluation.report.json`, and keep
  optional dependency failures precise at the command boundary.
- Fixed report validation to return invalid when `jsonschema` is unavailable or
  broken, instead of accepting partially validated payloads.
- Fixed MI probe subsampling so activations and targets stay paired, and
  normalized tensor handling for GPU and grad-tracking tensors before NumPy /
  scikit-learn processing.
- Fixed release, security, and live-smoke workflow behavior around immutable
  tag resolution, fail-closed CodeQL execution, shipped-surface dependency
  auditing, and repo-managed interpreter selection for end-to-end scripts.

### Dependencies
- Removed test/scientific-only packages from the base runtime where they were
  not required, moved MI-probe dependencies behind narrower extras, and
  refreshed workflow lockfiles to match the tightened CI and release surfaces.

### Documentation
- Refreshed workflow, security, CLI, config, and contract docs to match the
  new assurance model, supply-chain gates, fail-closed report validation, and
  light-import-safe module boundaries.
- Documented the standalone public contract bundle and the minimal-install
  verify/report/proof-pack onboarding path.
- Clarified secure-default runtime docs and user-facing guidance to describe an
  OCI container engine requirement first, keep Podman/Docker examples explicit,
  and scope Docker-only language to the local `act` workflow.

## [0.7.0] - 2026-04-09

### Added
- Added first-class GPT-OSS causal support and pilot Ministral 3 8B/14B
  presets, calibration configs, and support-matrix/catalog coverage.
- Added a CUDA-capable attested runtime image path for GPU hosts, smoke-sized
  calibration configs, and a split CLI smoke matrix with fast, negative-path,
  and GPT-2 realistic lanes.
- Added broader explicit per-file coverage thresholds, repo-wide debt
  checklists, and tighter architecture guardrails for the newly split core,
  reporting, and test owners.

### Changed
- Refactored large runtime, reporting, proof-pack, and test owners into
  smaller modules with tighter shell/core boundaries and clearer ownership.
- Reorganized the test suite around behavior-based file placement, `_support_*`
  helper modules, and split previously monolithic core/CLI/reporting tests into
  focused suites.
- Updated smoke and calibration maintainer flows to use the repo-selected
  Python, dedicated smoke configs, and lane-oriented scripts instead of the
  earlier monolithic CLI smoke harness.

### Fixed
- Fixed `quant_rtn` and report-generation fail-closed behavior so noop edits,
  failed subject runs, malformed primary-metric outputs, and invalid baseline
  pairing states no longer emit misleading downstream artifacts.
- Fixed delegated config execution by routing runtime delegation and
  calibration through a package-internal config-runner module instead of a
  hidden public CLI command, and aligned attested vs trusted-local tiny-smoke
  semantics by forwarding and resolving `tiny_relax` provenance consistently.
- Fixed trusted-local and attested live-demo paths across GPT-2 and 14B model
  flows, including CUDA runtime selection, HF cold-cache fallback handling,
  non-GPT-2 layer-count reporting, regenerated report runtime manifests, and
  primary-metric acceptance handling.
- Fixed developer-path regressions around Python interpreter selection, the
  mypy gate, Dependabot/CodeQL updater stability, ClusterFuzzLite Docker
  inputs, and repo gate coverage expectations.
- Reduced repo-wide static debt by removing remaining source `type: ignore`
  suppressions, narrowing broad exception fallbacks, and hardening
  observability, eval, calibration, and adapter boundary paths.

### Dependencies
- Bumped workflow and dev-security dependencies including `cryptography` to
  `46.0.7`, `ruff` to `0.15.9`, `katex` to `0.16.45`, and refreshed the pinned
  CodeQL action state and Dependabot handling.
- Added a `cu128` runtime-image lockfile for the CUDA attested runtime path.

### Documentation
- Refreshed maintainer and user docs around profile-driven token floors, smoke
  strategy, calibration surfaces, and the current trusted-local / attested
  operating model.

## [0.6.0] - 2026-04-04

### Added
- Added `google/gemma-4-E2B-it` as the shipped `supported_experimental`
  Gemma 4 text lane with a causal preset, calibration config, and support-
  matrix/catalog updates.
- Added phase-1 multimodal evaluation support with the built-in
  `hf_multimodal` adapter, the `vision_text` dataset provider, multimodal
  pairing/provenance wiring, and a Gemma 4 image-text preset plus local demo
  fixtures.
- Added proof-pack evidence levels and reviewer summaries to generated
  manifests and report outputs so artifact bundles surface review context more
  directly.
- Added targeted regression coverage for Gemma 4 loading/profile resolution,
  multimodal batching, baseline pairing, measured classification reporting, and
  the new public CLI assurance surface.

### Changed
- Unified the public local-vs-attested UX under
  `--assurance attested|trusted-local` across `evaluate`, `verify`, and
  `report verify`, replacing the earlier split between `--mode local` and the
  explicit unattested-artifact verify bypass.
- Refined `report` and `verify` CLI input handling, help text, and artifact
  loading so local verification and report review flows behave more
  consistently.
- Hardened shipped smoke and evidence-sweep helpers around tiny/GPT-2 bootstrap,
  tokenizer loading, smoke follow-ups, markdown example rewriting, and trusted-
  local verification so local and CI command surfaces stay aligned.
- Tightened proof-pack remote-setup and shell-harness checks and kept the
  support/evidence docs in sync with the post-`v0.5.1` assurance model.
- Hardened reporting, evaluation, orchestration, and guard helper contracts
  around numeric coercion, fallback validation, retry signaling, iterator
  handling, and config isolation, with expanded targeted regression coverage.

### Fixed
- Fixed Gemma 4 causal and multimodal loading paths so text and image-text
  runs resolve through the intended adapters, stay on the supported
  Transformers surface, and no longer fail on the Gemma 4 image-token
  truncation mismatch path.
- Fixed dataset loading on read-only or unwritable cache paths by retrying with
  a writable fallback cache.
- Fixed multimodal reused-baseline reporting so raw run artifacts now preserve
  measured `metrics.classification` counts instead of falling back to
  `pseudo_config` on successful image-text runs.
- Fixed report schema fallback validation so allowlist state stays isolated
  across runs and blank identifiers are rejected consistently when
  `jsonschema` is unavailable.
- Fixed snapshot-restore retry/fallback signaling, multimodal metric scalar
  handling, boolean-as-number validation gaps, and invariant iteration failure
  handling in core evaluation and reporting paths.
- Fixed trusted-local evaluation and verification ergonomics so local host runs,
  report verification, and docs/notebook examples all use the same explicit
  assurance vocabulary.

### Dependencies
- Bumped workflow `aiohttp` from `3.13.3` to `3.13.4` and pinned runtime/fuzz
  builder inputs more strictly for deterministic post-release smoke and
  packaging behavior.
- Updated Hugging Face runtime requirements and locks to `transformers==5.5.0`
  for Gemma 4 support.

### Documentation
- Refreshed CLI/reference/user-guide pages, shipped preset comments, and
  notebooks to teach the new assurance UX and the current trusted-local verify
  pattern consistently.
- Documented the proof-pack reviewer-summary surface and the writable dataset
  cache fallback path in the relevant reference and user-guide pages.
- Updated support and dataset docs to document the Gemma 4 E2B pilot lane and
  the new `vision_text` image-text evaluation flow.

## [0.5.1] - 2026-04-02

### Added
- Added a lightweight attested push smoke lane built around
  `sshleifer/tiny-gpt2`, a local JSONL fixture, and the new `Tiny Attested
  Smoke` workflow.
- Added a heavier GPT-2 canary preset and workflow for scheduled and manually
  dispatched end-to-end attestation checks.
- Added a tracked broad-exception review-bucket contract so remaining blanket
  catches are explicitly classified and linted instead of drifting silently.
- Expanded the coverage-enforcement inventory to include newly split
  implementation owners and helper surfaces as first-class critical files.
- Added package-native Ed25519 proof-pack manifest signing, verification, and
  key-generation flows so signed proof-pack verification no longer depends on
  host `gpg` tooling.
- Added stricter proof-pack remote-setup smoke coverage and higher-level
  harness checks around package installation, source provenance, and remote
  validation preflights.

### Changed
- Drove a repo-wide hardening and architecture cleanup pass across trust-
  critical evaluation, runtime attestation, proof-pack verification,
  determinism, registry, invariants, run orchestration, and reporting flows.
- Continued the shell/core split so CLI shells hand policy and owner logic to
  typed core and reporting helpers instead of owning fallback decisions.
- Decomposed the largest owner modules across runtime security, run
  orchestration, run execution, report building, verification checks, and
  proof-pack handling into smaller implementation files with stronger guardrail
  coverage.
- Converged runtime-manifest verification onto a single package-native Python
  path so product attestation, `invarlock-runtime-verify`, and
  `make runtime-verify` all exercise the same verifier implementation.
- Reworked proof-pack signing and verification around the same package-native
  Ed25519 manifest-signature contract used by the installed CLI and shell
  harnesses.
- Hardened attested smoke and tiny-matrix flows so they rebuild the local
  runtime image when needed, prefer the repo-selected interpreter, bootstrap
  the CPU-only Hugging Face stack deterministically, and keep local and CI
  runtime behavior aligned.
- Ratcheted refactored split owners to stricter 95% and 100% per-file coverage
  thresholds where the current suite supports it.

### Fixed
- Delegated and containerized evaluation reports now emit attested execution
  provenance into their runtime manifests.
- Runtime attestation and proof-pack verification now fail closed by default on
  unattested artifacts, mutable runtime-image refs without digests, and
  unsigned or unverifiable proof-pack manifests unless the explicit unattested
  override is set.
- Runtime attestation now uses the packaged Python runtime-manifest verifier
  directly, removing path-dependent behavior from product verification.
- Tiny attested smoke exports now write to host-writable paths, and unsigned
  proof-pack smoke runs use an explicit unattested-artifact override instead of
  implicitly depending on legacy behavior.
- Narrowed active-path broad exception fallbacks across core, guards, and CLI
  flows, and removed the remaining trust-critical broad catches.
- Restored calibration and evaluate/report edge behavior after the refactors,
  and resolved the post-split typing and coverage regressions surfaced by the
  tighter repo gates.
- Fixed release publishing and recovery paths around existing tags and
  dist-only uploads.
- Proof-pack maintainer packaging now fails closed when Git-backed source
  provenance cannot be collected, and explicit `--device cuda` delegation now
  rejects hosts without visible NVIDIA runtime support instead of silently
  dropping GPU passthrough.
- Fixed the runtime image and smoke bootstrap paths so attested Linux smoke
  runs install the CPU-only torch stack deterministically, reuse writable HF
  caches, and no longer depend on stale local runtime images or host `PATH`
  quirks.
- Restored 100% proof-pack shell-harness coverage and fixed warning-path shell
  helpers that had been swallowing finalize, evaluate, or verify failures.

### Removed
- Removed remaining compatibility surfaces that no longer fit the stabilized
  architecture, including legacy command shims, reporting facades, owner-layer
  patch-sync wrappers, the retired legacy RMT module, stale lazy export
  placeholders, and other shell-leaking or test-only indirections that had
  survived earlier migrations.
- Removed the repo-local Rust runtime verifier crate and the
  `INVARLOCK_RUNTIME_VERIFIER` product override so runtime attestation now has
  a single package-native verifier path.
- Removed the proof-pack `gpg` signing and verification path in favor of the
  package-native Ed25519 manifest-signature flow.

### Dependencies
- Patched vulnerable workflow locks and tightened smoke-workflow dependency and
  asset caching behavior for more deterministic CI execution.
- Updated verification and coverage gates so the packaged verifier and the
  newly split owner modules are exercised directly in local and CI runs.
- Bumped workflow and release security pins including `cryptography` to
  `46.0.6`, `pygments` to `2.20.0`, and the Sigstore GitHub Action used by the
  release workflow.
- Bumped `aiohttp` from `3.13.3` to `3.13.4` in workflow requirement locks and
  landed the corresponding Dependabot-equivalent fix on `staging/next`.

### Documentation
- Refreshed docs to match the post-`v0.5.0` architecture and operations model,
  including the shell/core redesign, current evaluate contract, and updated
  report-artifact guidance.
- Added remediation closeout records from the refactor program and updated the
  maintainer smoke notes to distinguish the push-gated tiny attested smoke from
  the heavier GPT-2 canary workflow.
- Documented the Python-only runtime-verifier contract and removed the obsolete
  external-verifier environment-variable guidance.
- Updated the architecture/security references so runtime attestation
  ownership now explicitly points at the package-native verifier instead of an
  external-binary model.

## [0.5.0] - 2026-03-25
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
- Bumped `ruff` from `0.15.6` to `0.15.7`.
- Bumped `actions/cache` from `5.0.3` to `5.0.4`.
- Bumped `actions/download-artifact` from `7` to `8`.
- Bumped `actions/upload-artifact` from `5` to `7`.
- Bumped `katex` from `0.16.28` to `0.16.38`.
- Bumped `flatted` from `3.4.1` to `3.4.2`.

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
