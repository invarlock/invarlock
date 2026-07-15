# InvarLock – Changelog

All notable changes to the InvarLock framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a first-party runtime-provider ABI and registry with built-in Hugging
  Face support through the optional `[hf]` extra and process-isolated
  GGUF/llama.cpp and TensorRT-LLM connectors. Machine-readable inventory
  distinguishes connector availability, backend delivery, and the metadata-only
  `runtime_qualification: not_probed` state. GGUF and TensorRT-LLM use the
  `first_party_experimental` plugin-maturity tier; connector readiness and
  strict-contract eligibility do not qualify a backend, image, platform, or
  model artifact.
- Added installed `build-schedule`, `prepare-binding`, `build-policy`,
  `run-side`, and `verify-pair` commands for the native providers. They derive
  canonical schedules and role bindings, authorize exact per-role inputs,
  produce each strict side separately, and replay the baseline/subject pair into
  a positive digest-only receipt. Strict Hugging Face runtime-behavior evidence
  uses the provider-owned exact model/artifact-bound scorer through the Python
  API, while normal `evaluate` behavior is unchanged. Cross-runtime claims are
  limited to policy-bound exact-match behavior and do not imply weight,
  activation, numerical, performance, export, or backend equivalence.
- Added provider-owned `inspect-inputs` derivation for GGUF/llama.cpp and
  TensorRT-LLM. The installed command authenticates the native artifact,
  backend, runner, source or tokenizer inputs and writes complete path-free,
  no-clobber settings without accepting caller-supplied hashes. A runnable
  mixed-provider example and operator guide now cover the full schedule,
  directed-policy, side-production, and paired-verification transaction.
- Added pinned native runtime-image qualification targets. GGUF requires its
  two-container behavior black-box before assigning the stable local tag.
  TensorRT-LLM requires a reviewed pinned-model inventory, an exact-base
  two-device hardware preflight, a candidate CUDA smoke, and derived
  engine-tree, tokenizer, and fixed-output bindings; its canary requires
  byte-identical provider evidence across two fresh sessions per GPU and the
  same canonical provider-receipt digest across GPUs. That digest is bound into
  the closed qualification summary. The isolated runner requires InvarLock's
  deterministic execution marker and fixed decoding settings, and records CUDA
  runtime version separately from driver version. Each TensorRT-LLM session
  authenticates and executes the fixed installed launcher from the reviewed
  runtime image's read-only root filesystem, verifies a no-new-privileges and
  zero-capability boundary, and rechecks launcher and interpreter identities
  around each invocation. Failed qualification leaves
  the existing stable local tag unchanged, and TensorRT-LLM still requires
  separate NVIDIA platform and real-engine qualification. CUDA and TensorRT-LLM
  qualification targets accept explicit Docker GPU selectors; CUDA smokes now
  require a visible selected device and execute a real tensor kernel. A
  maintained two-GPU TensorRT-LLM flow now snapshots a reviewed pinned model,
  builds an engine independently on each device, cross-executes the frozen
  primary engine on both devices, validates closed artifact-bound results
  through the immutable candidate image ID, rejects selector aliases and
  non-matching compute capabilities, sizes the isolated canary workspace for
  authenticated engine snapshots, and promotes only the digest bound by the
  qualification summary.
- Added deterministic compact runtime release-evidence assets with public
  qualification-receipt and asset-index contracts. Each asset embeds canonical
  sanitized provider summaries, source and image bindings, independently
  validated schedule-level receipts, and a closed hash inventory while
  excluding model files, engine bundles, raw logs, and host paths. Path-free
  qualification names allow one provider to carry multiple receipt-bound
  results with exact set validation; names inventory reviewed runs but do not
  claim independent execution. A no-clobber release handoff stages the archive
  and checksum under source-, tag-, and digest-bound names, revalidates both
  evidence and release bindings, and verifies the uploaded GitHub Release
  assets without treating the evidence source commit as the release commit.
- Added policy-bound `observe` and `enforce` authority for spectral, RMT, and
  variance findings in `policy-pack-v2`. Observation mode retains complete
  guard execution, provenance, replay, and reporting while leaving primary
  metrics, drift, invariants, and guard-metric impact as mandatory blockers.
  Current strict reports use claim set `invarlock-weight-edit-regression-v2`
  and require `assurance.guard_authority` to exactly mirror
  `resolved_policy.guard_authority`; shipped tiers remain all-`enforce` by
  default.
- Added a hash-bound historical guard-scenario observation index and closed
  semantic replay for the published Mistral asset, covering PM-pass spectral,
  RMT, and variance signals plus a spectral negative control without upgrading
  legacy reports into current strict-assurance results.
- Added immutable real-training profiles and fail-closed receipts for tiny
  full-parameter fine-tuning and PEFT LoRA train/serialize/reload/merge flows.
- Added complete raw-baseline, independently supplied policy-pack, and
  runtime-image inputs to strict report and evidence-pack verification.
- Added catalog-bound evidence verification with immutable catalog, source,
  runtime-image, and signer anchors plus a repository-owned command for running
  one evidence lane.
- Added canonical report-contract, policy-provenance, dataset-identity,
  checkpoint-identity, and guard-recomputation checks for release-grade
  verification.
- Added architecture, coverage-ratchet, mutation-smoke, documentation,
  distribution, and release-preflight checks to the maintained validation
  entrypoints.
- Added catalog profiles, adapters, and deterministic dataset materialization
  for current masked-language, sequence-to-sequence, causal-language, MoE, and
  vision-language model families.
- Added a design-partner diagnostic runbook and checked case template for one
  immutable baseline, one genuinely transformed Hugging Face subject,
  immutable subject revision, reviewer-owned trust inputs, strict verification,
  and decision-packet handoff. The handoff hash-binds the report and copied
  transformation receipt, retains the declared change kind, and verifies both
  artifacts against tampering, with explicit acceptance criteria and runtime
  non-goals.

### Changed

- Strengthened cross-runtime behavioral authorization in `policy-pack-v3` with
  directed baseline and subject bindings for the schedule, provider, artifact
  format and identity, outer runtime image, and execution-settings digest.
  Strict side verification reloads and cross-checks the provider receipt and
  scoring observation before paired replay, and exact-match scoring compares
  literal typed values without coercion. Side publication is
  descriptor-relative, atomic, and no-clobber; it checks staging and parent
  identities immediately around publication and rolls back if a directory is
  replaced during the operation. Portable cross-runtime comparisons use one
  sequence per scheduled record (`batch_size=1`); same-artifact no-change
  comparisons remain valid when the directed policy authorizes them, without
  claiming producer independence.
- Replaced the `published_basis` support-tier name with
  `maintained_catalog`, separating maintained lane eligibility from the
  independently reported `available` and `not_created` evidence states. This
  migration introduces `support-matrix-v2`, `model-family-catalog-v2`,
  `public-evidence-index-v2`, `plugins-v2`, and `policy-pack-v2`, renames the
  compact index to `catalog_evidence_index.json`, and renames the corresponding
  Python helpers and plugin metadata. The model lifecycle contract is now
  `model-classification-v2`, where `cataloged` describes maintained catalog
  scope without implying evidence publication. Frozen `policy-pack-v1` inputs
  and historical release-asset paths remain verifiable but are not emitted by
  new builders.
- Made guard-value scenario verdicts require baseline-relative spectral and
  variance signals, including an explicit no-new-cap negative control and a
  positive measured variance signal rather than proposed scales alone.
- Reclassified deterministic low-rank and dense perturbation generators as
  synthetic edit fixtures. Real `lora_merge` and `fine_tune` labels now require
  training provenance rather than generated look-alike edits.
- Reworked guard reporting around measured primary-metric impact while keeping
  runtime and memory overhead as separate system measurements.
- Strengthened the report schema and renderer-independent outline so JSON,
  console, Markdown, and HTML surfaces share the same decision, metric, policy,
  guard, provenance, and appendix structure.
- Clarified that runtime-manifest binding and image-digest matching establish
  declared identity consistency, not execution attestation.
- Standardized catalog evidence execution on the repository-owned lane command.
- Hardened public-evidence archive inspection to reject unsafe paths, duplicate
  members, and non-regular entries, and to recompute the unique regular-file
  count and byte total before accepting an external asset.
- Published strictly verified frozen-v1 evidence packs for 31 model-catalog
  lanes through a hash-bound GitHub Release asset and compact source/wheel
  index. The remaining catalog rows use the **Evidence not yet created** status
  from the support matrix and documentation. These are `noop`
  same-checkpoint compatibility runs covering the evidence mechanics;
  transformed-subject detection and effectiveness remain separate experimental
  claims. The current verifier accepts the frozen-v1 packs through its explicit
  compatibility path; these packs do not exercise the new v2 guard-authority
  fields.
- Updated integration examples to distinguish real training, serialization,
  reload, merge, pruning, and quantization workflows from synthetic fixtures
  and to validate their emitted artifacts fail closed.
- Refocused the root and documentation landing pages on the evaluate, verify,
  and report workflow, current contracts, supported model families, and public
  evidence status.
- Updated the pinned CodeQL, uv setup, Ruff, and Setuptools validation
  toolchain.
- Extended isolated installed-wheel release preflight to reject checkout or
  namespace-package leakage, import the shipped runtime modules, validate the
  exact first-party provider inventory and command surface, and smoke schedule
  and directed-policy construction.

### Removed

- Removed full public evidence packs from source and wheel distributions; the
  compact catalog index now binds the corresponding GitHub Release asset.
- Removed the standalone negative-fixture publisher and source-tree bundle
  contract. Release preflight now audits the compact current public-evidence
  index; deterministic guard-scenario and fail-closed verifier suites remain
  repository gates, while historical observations remain non-authoritative.
- Removed superseded evidence-generation entry points in favor of the
  catalog-bound lane command.
- Removed duplicate tiny fine-tuning and PEFT materializers in favor of the
  immutable training profiles.

### Fixed

- Fixed RMT and variance probe validation to recompute epsilon and measured-gain
  arithmetic, bind ordinary and baseline-relative statuses, and reject
  internally inconsistent sidecars.
- Fixed strict evidence-pack verification to bind each subject report to signed,
  checksummed raw baseline material and independently supplied policy inputs.
- Fixed verifier recomputation and fail-closed handling for paired windows,
  bootstrap statistics, primary-metric drift, guard evidence, policy digests,
  runtime manifests, and signer authorization.
- Fixed masked-language-model variance targeting and deterministic WikiText
  calibration inputs used by the catalog profile.
- Fixed vision-language materialization and paired bare-control replay so local
  image paths are rehydrated only from authenticated current inputs while
  prompt, answer, and image identity bindings remain verified.
- Fixed modern dense and MoE adapter routing, quantized-wrapper compatibility,
  snapshot restoration, and runtime dependency pins exercised by catalog lanes.
- Fixed TensorRT-LLM qualification to accept an explicit Docker GPU selector,
  allowing independent single-GPU workers on multi-GPU hosts without exposing
  every device to each container.
- Fixed TensorRT-LLM engine authentication to ignore read-driven access-time
  changes while still rejecting changes to names, sizes, device/inode identity,
  mode, modification time, status-change time, or content.
- Fixed TensorRT-LLM qualification to initialize the pinned vendor library
  environment without contaminating strict machine output, keep required
  runtime and engine-build timing caches inside the bounded temporary
  filesystem, validate content-derived engine bundle names, and recognize both
  canonical proprietary and open-kernel NVIDIA driver-version records.
- Fixed documentation and CLI examples so strict commands include every
  independently supplied verifier input.
- Fixed public-evidence, packaged-data, support-matrix, and model-catalog
  consistency checks so superseded results cannot be presented as current.
- Fixed public-evidence privacy screening to inspect decoded JSON values,
  avoiding false host-path matches in escaped model output while retaining
  rejection of actual host paths.
- Fixed source-checkout CLI and immutable training-profile CI isolation, Torch
  wheel-version normalization, strict source-matrix import-path preservation,
  and training-output ownership checks against immediate inode reuse.
- Fixed training-artifact hashing to detect same-size concurrent rewrites even
  when the underlying filesystem does not advance modification timestamps.
- Fixed error-injection `must_pass` verdicts so satisfied detector expectations
  cannot hide a report-level primary-metric, drift, invariant, or guard failure.
- Fixed installed-package plugin discovery so InvarLock's identical shipped
  entry points are not rejected as duplicate built-ins, while packaging drift
  and third-party name collisions continue to fail closed.
- Fixed runtime release-evidence publication to normalize non-collision
  filesystem link failures into closed CLI errors after cleaning temporary
  files, avoiding raw exception and host-path disclosure.
- Fixed runtime release-evidence staging to create archives and checksum
  handoffs with owner-read-only permissions and to fail closed if command
  dispatch completes without a validated result.
- Fixed native GGUF and TensorRT-LLM run-directory setup to close descriptors
  and remove temporary state if the initial filesystem identity snapshot fails.
- Fixed static-analysis findings in runtime type-alias exports, evidence utility
  CLIs, pruning-contract key validation, report-config cleanup, and verification
  export helpers.

## [0.12.1] - 2026-07-05

### Added

- Added stock-clean attention guard-value control evidence for the Mistral 7B
  guard-value demo package.

### Changed

- Updated guard-value scenario contracts so targeted spectral, RMT, and
  variance/VE probes are required detections while FP8 stress remains
  informational historical context.
- Refined docs asset configuration for MathJax loading and documentation
  focus/selection styling.
- Updated markdownlint-cli2 to 0.23.0 for docs linting.

### Removed

- Removed the retired attention guard-value control report and references from
  the current public evidence package.

### Fixed

- Fixed strict verification handling for `must_detect` guard-value probe
  reports and added coverage for targeted guard-value verdict scenarios.

## [0.12.0] - 2026-06-30

### Added

- Added knowledge/self-edit evidence metadata, public documentation, and
  report/provenance coverage for self-edit workflows.
- Added LoRA, full fine-tune, and magnitude-prune integration lanes, including
  generated evidence-pack lane support and tiny public BYOE smoke entrypoints.
- Added a training evidence campaign surface for PEFT LoRA merge and full
  fine-tune runs with checkpoint references, runtime manifests, verification
  summaries, and hash inventories.
- Added model-editing evidence-bundle manifests, verification summaries, and
  training-matrix planning artifacts for release review.
- Added CUDA 12.8 runtime backend compatibility evidence for bitsandbytes,
  GPTQModel, HQQ, Quanto, TorchAO, and compressed-tensors lanes.
- Added attention-backend compatibility evidence covering FA2-unavailable
  eager execution behavior.
- Added evidence-pack lifecycle stress/resume evidence and larger-model
  validation findings as summarized public evidence classes.
- Added larger-model published-basis evidence and presets for Gemma 4 31B,
  Qwen3.5 27B scoped, Qwen3.6 27B scoped, and GPT-OSS 20B.
- Added catalog and support-matrix entries for modern model families, including
  updated published-basis distinctions for Gemma, Qwen, and GPT-OSS lanes.
- Added a Qwen linear-MoE causal adapter spec and tests for Qwen MoE model
  routing.
- Added scheduled full-history secret scanning and expanded supply-chain PR
  scan coverage for release and dependency validation.

### Changed

- Synced the post-`v0.11.0` integration branch and optimized release workflow
  maintenance so the next release branch starts from current CI and evidence
  workflow state.
- Reworked public evidence category naming around runtime backend
  compatibility, attention-backend behavior, lifecycle stress, and larger-model
  validation findings.
- Reorganized the model family catalog and support matrix to better reflect
  current adapter support, catalog candidates, and published-basis evidence.
- Expanded public evidence audits and packaged public evidence synchronization
  for the new training, backend compatibility, lifecycle, and larger-model
  evidence classes.
- Improved evidence-pack queue/task tooling for model creation, scheduling,
  preflight validation, task serialization, runtime checks, and failure
  classification.
- Updated evidence-pack docs and public run examples to use repo-relative
  artifact wording and avoid environment-specific execution details.
- Consolidated MkDocs JavaScript and stylesheet overrides under
  `docs/assets/`.
- Updated workflow and development dependency pins, including Ruff,
  actions/cache, actions/checkout, actions/setup-python, and
  actions/attest-build-provenance, and refreshed the affected workflow lock
  files and contract tests.

### Removed

- Removed the legacy Qwen14 sentinel script/test entrypoints after replacing
  that coverage with the model-evidence sweep and catalog-driven validation
  paths.

### Fixed

- Fixed evidence-pack reusable baseline schedules and validation harness gaps
  that could make release-review evidence runs less representative.
- Fixed generated LoRA and fine-tune evidence-pack parity, including edit
  metadata propagation and strict verification coverage.
- Fixed training evidence campaign validation so generated LoRA/full fine-tune
  reports, manifests, and hashes verify consistently.
- Fixed model-catalog smoke materialization for VQAv2 and materialized evidence
  lanes with preset overrides.
- Fixed verifier failure classification for larger-model catalog evidence so
  failed, negative, and compatibility findings are reported distinctly.
- Fixed snapshot reload fallback behavior for large HF model lanes without
  retrying an already-failed snapshot restore path.
- Fixed public evidence catalog consistency, support-matrix synchronization,
  and packaged public evidence indexes after larger-model evidence updates.
- Fixed dependency workflow contract drift after the merged workflow/action
  updates and increased the supply-chain PR scan timeout to allow advanced
  dependency setup to complete.

## [0.11.0] - 2026-06-16

### Added

- Added restrained InvarLock branding and version metadata to human-readable
  Markdown/HTML reports and evidence-pack verdict summaries while keeping JSON
  reports, manifests, and signed metadata machine-stable.
- Added first-class baseline-relative guard warnings in reports and verification,
  including strict warning-policy handling so users can treat guard movement as
  advisory by default or fail verification when warnings are present.
- Added published guard-value evidence for Mistral 7B with clean reruns covering
  spectral, RMT, and variance/VE guard movement, plus scenario-backed PM-only
  versus PM+guards comparison artifacts.
- Added modern published-basis evidence for expanded dense causal model
  families, including Mistral/Ministral, Qwen2/Qwen2.5/Qwen3/Qwen3.5,
  Granite 4.1, DeepSeek R1 Qwen variants, Phi-4, SmolLM3, TinyLlama,
  OLMo 2, OpenLLaMA, and Falcon lanes.
- Added published-basis coverage for additional architecture families,
  including FLAN-T5 seq2seq, OLMoE, Mixtral, Qwen3 30B-A3B MoE, Qwen3.5 4B
  image-text, Gemma 4 E2B image-text, Gemma 4 E4B image-text, Gemma 4 12B
  image-text, and Gemma 4 26B-A4B image-text MoE evidence lanes.
- Added public VQAv2 materialization support, scarce vision-text evidence
  window splitting, multimodal replay preservation, and processor digest
  evidence for image-text model runs.
- Added a public image-text published-basis adequacy gate requiring measured
  accuracy, enough final examples, and concise answer-shaped generations when
  prediction records are embedded.
- Added JSON-answer extraction for `vision_text` evaluation so public VQA runs
  can prompt models for `{"answer": "..."}` structured output without breaking
  exact-answer scoring.
- Added a renderer-neutral report outline view model that groups modern
  evaluation evidence into decision, primary-metric, policy-gate, guard-signal,
  benchmark-comparison, provenance, and appendix sections before renderer work.
- Added seq2seq evidence-run support for FLAN-T5, including label preservation,
  shuffled split handling, T5 guard targets, and calibration preview labels.
- Added model-evidence GPU backlog lanes, preset overrides, remote-code
  opt-in propagation, worktree-aware remote launch handling, and GPU preflight
  warnings for underprovisioned MoE lanes.
- Added an offline model-candidate compatibility audit to `contracts-check` so
  named lanes and catalog candidates must have coherent adapter routes, presets,
  materialization metadata, and large-model loading hints before GPU launch.
- Added large-model and MoE memory controls for evidence runs, including
  memory-sensitive HF loads, bounded calibration windows, snapshot policy
  controls, container scratch cleanup, and safer cleanup between phases.
- Added Gemma 4 12B calibration configuration and documented gated/access
  requirements for Gemma candidate lanes.
- Added a time-boxed, issue-tracked `pip-audit` allowlist entry for the
  unfixed Torch `CVE-2025-3000` advisory affecting optional HF and advanced
  install surfaces.
- Added production identity assets for the README and external surfaces,
  including light/dark logo and mark variants, app icon, and favicons.

### Changed

- Tightened evidence-pack guard-value detection so published guard claims must
  be baseline-relative and scenario-backed rather than relying on guard signals
  that already appear in the no-edit baseline.
- Updated guard-value claim wording across evidence-pack docs, public evidence,
  and support-matrix surfaces to distinguish policy failures, guard warnings,
  and published guard-value proof.
- Updated evaluation-report HTML export to render directly from the shared
  report outline, including benchmark-comparison and guard-warning sections,
  instead of converting the historical Markdown body, and aligned report HTML
  colors with the site Ledger ink branding tokens.
- Updated evaluation-report HTML layout to use a flatter ledger-style document
  treatment with a sticky brand/theme row, light/dark toggle, and active-section
  highlighting in the left rail, and surfaced baseline identity in the summary
  and Decision sections.
- Updated support-matrix organization and evidence grouping so published-basis,
  experimental, multimodal, seq2seq, MoE, and blocked/access-gated lanes are
  easier to scan without adding a redundant grouping column, and normalized
  hardware wording for published evidence rows.
- Updated HF adapter support for newer model shapes, including resilient
  multimodal loader fallback, tokenizer remote-code propagation, ChatGLM layout
  and loader support, Falcon-H1/Falcon-Mamba projection classification, dense
  gate-up FFN classification, and Gemma 4 multimodal runtime extras.
- Raised the Hugging Face adapter Transformers floor for modern model support
  and refreshed optional Torch requirements to avoid `torch==2.12.0` while the
  current supply-chain advisory has no fixed upstream version.
- Centralized permissive-license model classification so support, calibration,
  public-evidence, and catalog checks draw from one audited model inventory.
- Updated adapter examples to reflect the newly published dense, multimodal,
  seq2seq, and MoE model shapes.
- Updated ordinary CI expectations so long GPU evidence sweeps are treated as
  manually dispatched model-evidence work instead of blocking the standard
  green-main gate.
- Updated public evidence packaging to use compact evaluation reports for large
  published-basis artifacts while retaining the evidence manifests, runtime
  manifests, model revisions, and provenance needed for audit.
- Updated installed-wheel public evidence packaging to ship a compact
  `published_basis_index.json` with hashes, sizes, lane coverage, and carrier
  policy instead of duplicating the full source-tree public evidence artifact
  corpus.
- Updated Qwen3.5 4B, Gemma 4 E2B, Gemma 4 E4B, Gemma 4 12B, and Gemma 4
  26B-A4B image-text evidence status to published basis after structured
  JSON-answer reruns passed the public VQAv2 quality floor with strict
  verification and no guard warnings.
- Updated model-evidence sweeps to use an explicit repo-visible Hugging Face
  cache by default so container GPU runs and revision capture inspect the same
  downloaded model snapshots.
- Aligned the Ruff 0.15.17 bump across the pre-commit hook and hashed workflow
  requirement locks.
- Updated Dependabot-managed CI action pins and README public-evidence notes
  alongside the refreshed packaged public-evidence index.
- Moved model-evidence lane execution, remote launches, evidence-pack direct
  front doors, status logs, summaries, retries, and artifact manifests onto a
  shared typed evidence-workflow layer while keeping shell scripts as thin
  process/worker dispatch entrypoints.
- Updated model-family lane routing to use a shared catalog/support-matrix route
  index that resolves presets and adapters from model task role, modality, and
  evidence metadata instead of rebuilding model-evidence-local override tables.
- Moved `evaluate` baseline, subject, and report-generation phases onto typed
  request/runtime helpers with explicit dependencies and a thinner command body.
- Moved `CoreRunner.execute` onto an explicit typed execution plan with named
  prepare, guard-preparation, edit, guard-collection, evaluation, and finalize
  phase owners.
- Updated report summaries, Markdown output, schema validation, and
  `report explain` to share primary-metric interpretation, honor configured
  acceptance ranges, and render accuracy deltas consistently in percentage
  points.
- Shared report-outline facts with Markdown reports and `report explain` so
  the human Markdown, HTML, and CLI explain surfaces expose the same high-level
  decision, policy, and guard-signal facts before their detailed sections.
- Made `report explain --evaluation-report` explain the supplied
  `evaluation.report.json` directly so portable reviewer bundles no longer need
  linked raw baseline/subject run artifacts for the explain path.
- Updated README branding to the Ledger ink palette, including static
  GitHub/PyPI-compatible Shields badges and refreshed light/dark logo, mark,
  app-icon, and favicon assets synced from the current site brand surface.

### Removed

- Removed the broken scheduled full-CI trigger and scheduled GPU model-evidence
  sweep triggers; full verification remains available by manual dispatch,
  release supply-chain checks remain part of the release workflow, and
  model-evidence sweeps remain manually runnable.
- Removed the stale `tests/fuzzing` owner expectation from the active test-tree
  contract after the fuzzing surface was retired from the repo layout.
- Removed inactive hosted model checkpoints from active support, calibration,
  and public-evidence surfaces.

### Fixed

- Fixed exported HTML reports to use the packaged InvarLock bracket-and-signal
  mark instead of the temporary `IL` text fallback.
- Fixed exported HTML report rendering for summary status fallback, policy
  threshold wording, sticky navigation/active-section behavior, and empty table
  detail columns.
- Fixed the root README CI badge to use Shields' semantic status colors for
  passing/failing/default states instead of forcing the static brand accent.
- Fixed evidence-pack queue and GPU-runner stability issues, including memory
  helper resolution, container scratch cleanup, retry handling after interrupted
  evaluate phases, and host/container output publication paths.
- Fixed remote model-evidence branch sync for `work/...` branches by fetching
  the current branch ref explicitly before fast-forwarding the remote checkout.
- Fixed Mistral guard-demo manifests, run-log tracking, and public evidence
  scope notes so the published demo accurately identifies
  non-baseline-relative FP8 guard signals.
- Fixed Gemma/Qwen multimodal evaluation gates by using classification-count
  accuracy intervals, delta semantics for accuracy drift, paired multimodal
  window reporting, scarce-window split handling, and a public-basis absolute
  image-text quality floor.
- Fixed MoE and large-model guard/report behavior, including router warning
  handling, Qwen variance calibration bounds, and variance calibration
  truncation fallbacks.
- Fixed Gemma 4 26B evidence source provenance after publishing its MoE
  assurance basis.
- Fixed `adapter:auto` routing for named image-text Gemma/Qwen candidates and
  Marian/OPUS/MBART/Pegasus seq2seq model IDs.
- Fixed seq2seq evidence correctness for shuffled splits and preserved labels
  in paired evidence runs.
- Fixed GPTQModel compatibility with newer Transformers imports.
- Fixed typed-surface checking for tokenizer load kwargs attached to dataclass
  model profiles.
- Fixed local verification hygiene after the published-basis expansion by
  keeping test files under the repository size guideline and restoring coverage
  enforcement edge-path tests.
- Fixed the PR-time supply-chain scan to run `gitleaks` over pull request
  changed-file contents instead of timing out on repository-history scans;
  release scans still cover full-history secret scanning.
- Fixed PR supply-chain scans by raising `aiohttp`, `cryptography`, and
  security-tool `pip` floors, then refreshing workflow locks past vulnerable
  pins.
- Fixed `gitleaks` false positives for published evidence tokenizer digest
  fields by adding a narrow config allowlist used by PR and release scans.

## [0.10.0] - 2026-06-03

### Added

- Added evidence-pack artifact taxonomy and edit metadata sidecars so validation
  checkpoints, fault-injection fixtures, deployable optimized subjects, and
  evidence-only packs are labeled explicitly in scenarios, packaged evidence,
  and summary outputs.
- Added package-native evidence-pack report-assurance controls and release-review
  hardening so pack integrity strictness and nested report assurance are
  configured separately.
- Added backend-inventory, load-smoke, inference-smoke, memory-report, and
  deployable-sidecar validation contracts for optional quantized subject
  adapters.
- Added detached evidence-pack source snapshot support and a clean quant-runtime
  evidence path for optional quantized-subject runs.
- Added evidence-pack signer authentication via `--expected-fingerprint`, local
  trust stores, and verifier JSON `authenticity` states (`pinned`, `unpinned`,
  `mismatch`).
- Added signed public GPT-2 evidence pack fixtures, strict public GPT-2/BERT
  evidence reports, real tiny-GPT-2 quant and external magnitude-prune BYOE
  runs, checkpoint-reference artifact packages, non-quant BYOE examples, and
  caught-regression fixtures for guard and policy failures.
- Added `make public-evidence-audit` and CI coverage for classifying public
  evidence fixtures, verifying signed pack metadata, and enforcing evidence
  scope in packaged public evidence.
- Added public evidence walkthrough documentation with pinned evidence-pack
  verification commands and explicit integrity-versus-authenticity guidance.
- Added scripts inventory governance, architecture-fragmentation tracking, and
  guard fallback diagnostic checks so repository maintenance surfaces are
  classified and auditable.
- Added guard fault-injection seam documentation and regression coverage for
  spectral, RMT, variance, and fallback diagnostic paths.
- Added public contract stability documentation for report schemas,
  evidence-pack formats, verifier outputs, CLI stability classes, adapter
  support tiers, and the pre-1.0 package stability posture.
- Added optional `hf_torchao` adapter discovery for torchao int8 runtime
  quantization, including module maps for guard targeting and public adapter
  capability metadata, plus a runnable TorchAO integration example that proves
  the `hf_torchao` subject adapter path.
- Added optional `hf_hqq` adapter discovery for HQQ runtime quantization,
  including capability metadata, guard targeting, backend inventory, and a
  runnable HQQ integration example with host/off and CUDA/container strict
  lanes.
- Added optional `hf_quanto` adapter discovery for Quanto runtime
  quantization, including capability metadata, backend inventory, a narrow
  example-only CUDA runtime image, and a runnable Quanto integration example
  with host/off and CUDA/container strict lanes.
- Added optional `hf_ct` adapter discovery for compressed-tensors
  pre-quantized checkpoints, including capability metadata, backend inventory,
  and a narrow example-only CUDA runtime image smoke path.
- Added the root README evaluation and verification flow diagram as a versioned
  SVG asset.

### Changed

- Clarified plugin support tiers and public descriptions: built-in guards and
  adapters now expose support metadata, `quant_rtn` is described as an RTN
  dequantized weight-edit simulation, and the demo hello guard is registered as
  demo-only.
- Updated evidence-pack validation edits to use shared edit implementations,
  shared artifact saving, schema-aware `edit_metadata.json`, and metadata-aware
  artifact validation in both single-edit and batch-edit paths.
- Updated evaluate and evidence-pack command surfaces for separate baseline and
  subject adapters, deferred optional report rendering, canonical WikiText
  dataset identifiers, stale pack-staging report ignores, and Qwen14 sentinel
  cleanup guidance.
- Updated quantized adapter documentation and runtime image guidance around
  GPTQModel-backed GPTQ/AWQ loaders, platform-dependent BNB loading, and
  torchao runtime quantization.
- Updated quant runtime image builds to use portable build dependencies, a CUDA
  devel base, and a retained JIT toolchain for quantized adapter evidence paths.
- Expanded CUDA quant runtime-image smoke coverage across BNB, GPTQModel-backed
  GPTQ/AWQ, TorchAO, HQQ, Quanto, and compressed-tensors adapter families.
- Re-scoped grouped evidence evaluation to the simpler ungrouped path after
  remote timing showed no useful default speedup.
- Consolidated run-orchestrator execution helpers and reporting render/context
  modules to reduce re-export shims and one-helper-per-file fragmentation.
- Reorganized `scripts/` into maintained families (`checks`, `coverage`,
  `docs`, `security`, `smoke`, `model_evidence`) with an inventory manifest and
  a `make scripts-audit` gate.
- Refactored evidence-pack script helpers so queue state, validation state,
  edit implementations, edit metadata, artifact saving, artifact validation, and
  pack verification logic live in Python modules instead of brittle shell-only
  state handling.
- Refactored smoke and maintenance scripts around shared helpers, clearer
  runtime modes, explicit model-download behavior, and stronger exit/status
  reporting for the tiny all-model matrix and evidence script flows.
- Refactored source and test maintainability hotspots, including core source
  helper extraction, run-command test helper consolidation, core test topology
  cleanup, and validation-suite orchestration splits.
- Standardized assurance document openings, glossary examples, math rendering,
  and README/doc references so public docs match the current implementation and
  render consistently on GitHub.
- Reframed assurance-boundary prose around scoped evidence and configured
  weight-edit regression reviews.
- Updated repository surface metadata, security response wording, npm package
  identity metadata, and third-party notices to match the current packaged and
  optional dependency surfaces.
- Updated dependency floors and tool pins, including Torch, `matplotlib`,
  setuptools, `shellingham`, Ruff, markdownlint, and KaTeX.
- Updated `CODEOWNERS` to use the organization `core-maintainers` team for
  protected ownership rules.
- Updated the pre-commit workflow check context to report as `pre-commit`
  instead of the generic `run` job name.
- Updated contributor guidance with PR-ready local gate expectations, including
  docs, coverage, typed-surface, supply-chain, packaging, and public-evidence
  checks before protected-branch PRs are pushed for review.

### Removed

- Removed unverified TorchAO deployable evidence plumbing and kept deployable
  edit generation absent until a backend can provide complete sidecars and
  target-stack evidence.
- Removed public ClusterFuzzLite/fuzzer surfaces from the OSS repo layout and
  ignore surface.
- Removed transient release-hardening and release-checklist pages from the
  published docs tree so durable docs stay focused on current repository
  behavior and stable user-facing contracts.
- Removed stale docs workflow targeting for the retired `develop` branch.

### Fixed

- Fixed evidence-pack release-review edge cases, including report-assurance
  forwarding, dev-profile rejection, strict PASS requirements, runtime sidecar
  requirements, signed-pack requirements, and deployable sidecar semantic
  validation failures.
- Fixed evidence-pack verification and host evidence semantics so
  expected-failure reports and host model evidence are ineligible for strict
  assurance.
- Fixed invariant-vocabulary guard compatibility for quantized wrapper models.
- Fixed pseudo-accuracy handling so non-dev report generation fails unless
  pseudo metrics are explicitly allowed, and report output marks pseudo or
  non-assurance results visibly.
- Ratcheted critical coverage thresholds and added edge-path coverage for
  registry metadata and metrics runtime behavior.
- Fixed typed-surface mypy coverage after orchestrator consolidation by removing
  stale file paths from the maintained check list.
- Fixed stale workflow/check metadata after GitHub ruleset hardening so
  protected branches require current CI contexts only.
- Fixed smoke configuration presets for seq2seq model paths and the tiny matrix
  checklist network metadata.
- Fixed public documentation to refer to the packaged runtime tier policy as
  logical `runtime/tiers.yaml` while retaining the documented override path.
- Fixed secret-scanning false positives on model architecture names without
  changing adapter behavior.
- Fixed reporting test import ordering to satisfy the active pre-commit gate.
- Fixed additional guard and reporting compatibility edge cases, including
  variance fault-injection preservation, telemetry monkeypatch stability,
  canonical reporting imports, and shared check-script IO helpers.

## [0.9.0] - 2026-05-25

### Added

- Added strict assurance mode for `evaluate` and `verify`, including the
  `invarlock-weight-edit-regression-v1` claim set, central report verdicts,
  strict paired-length checking, and structured report-build evidence for
  synthesized, repaired, and fallback fields.
- Added adversarial verifier coverage for strict guard-chain enforcement,
  runtime provenance failures, mutated report fields, missing strict assurance
  claims, unsupported guard-status shapes, and installed-wheel strict
  verification/report rendering outside the repository tree.
- Added trust-model, strict assurance checklist, failure examples, alternatives
  comparison, runtime provenance guide, one-run lifecycle, guard-validation
  smoke, and empirical guard-evidence documentation.
- Added maintainer release and evidence gates: CVE audit reporting,
  `make dist-check`, content-validating `make release-evidence-check`,
  `make guard-validation-smoke`, release-evidence checklist coverage,
  wheel/sdist hash checks, SBOM checks, runtime image digest checks, strict
  example report bundle checks, and offline bundle validation.

### Changed

- Default `evaluate` assurance posture is now fail-closed strict mode for
  assurance evidence generation, and strict generated reports now use a
  pending-verifier top-level verdict until runtime provenance is checked by
  `verify`.
- Generated reports now distinguish declared runtime provenance from verifier
  confirmation, preserve a separate report-local verdict, and record the actual
  runtime-provenance verification result in `verify` output.
- Consolidated open dependency/security PR content into this branch: CodeQL
  action SHA refresh, Ruff 0.15.14, `idna>=3.15`, and
  `pymdown-extensions>=10.21.3`.
- Durable assurance and reference docs now describe the current strict contract
  without patch-release dating, and assurance documentation filenames now use
  numbered prefixes consistently except for the glossary.
- Config-driven run execution now uses `ConfigExecutionRequest` as the
  canonical request object across public command plumbing, internal delegated
  execution, delegated argparse, and container-launch argv serialization.
- GPT-2 smoke coverage now uses a user-journey runner with local, container,
  strict report-bundle, configurable no-op, quantized-subject, custom-edit,
  evidence-pack, and expected verifier-failure journeys, plus a final results
  table and `INVARLOCK_SMOKE_DEVICE` control for CUDA/CPU hosts.
- Release, empirical guard-evidence, verifier, runtime-provenance, guard
  evidence, and report-finalization paths now use typed request/result or
  manifest objects at their main trust boundaries, reducing duplicated
  ad hoc validation and serialization logic; strict guard blocking checks now
  reuse the canonical guard-evidence normalizer.
- `quant_rtn` is now explicitly an RTN quantize/dequantize simulation edit:
  reports distinguish theoretical packed-memory estimates from actual
  floating-point dequantized storage, canonical plan digests include meaningful
  edit parameters and selected targets, runtime-local parameter object IDs are
  kept under `runtime_debug` instead of normal report metadata, tied weights
  distinguish selected modules from physically quantized modules, and the edit
  emits per-module and aggregate quantization-error metrics.

### Removed

- Removed transient release-hardening and release-checklist pages from the
  published docs tree so durable docs stay focused on current repository
  behavior and stable user-facing contracts.

### Fixed

- Removed ambiguous `group_size`/4-bit paths from the built-in `quant_rtn`
  edit contract and sample overlays; real packed quantized artifacts should use
  adapter-backed or external subject workflows instead of this simulation edit.
- Fixed `quant_rtn` preview and digest metadata for tied weights so previews
  deduplicate the same physical parameters as apply, theoretical packed-memory
  estimates use unique parameters, and plan digests distinguish stable tied
  module groups without using runtime-local object IDs.
- Fixed `quant_rtn` report-plan naming so selected module names and physically
  quantized module names are no longer duplicated under ambiguous aliases, and
  runtime-local debug IDs are omitted from CI/release edit results unless
  verbose/debug output is explicitly requested.
- Fixed the strict assurance trust boundary so host/unverified provenance,
  custom guard order, dev/aggressive profiles, unsupported blocking statuses,
  fallback fields, and missing guard evidence cannot pass as strict assurance.
- Fixed the curated assurance test lane so it runs the strict assurance
  contract, verifier guard-chain, and strict paired-metric regression tests.
- Fixed empirical guard-evidence manifest validation so non-object diagnostics
  are reported once.
- Remediated `pip` and `urllib3` CVEs across the uv and workflow lock
  surfaces, cleared the stale `pip-audit` allowlist, and aligned the
  supply-chain workflow contract tests with the current disk-cleanup steps.

## [0.8.0] - 2026-04-23

### Added
- Added shipped public evidence under
  `invarlock/_data/public_evidence/published_basis/...` together with matching
  repo-visible `public_evidence/published_basis/...` source copies.
- Added maintainer smoke lanes for runnable docs, installed-wheel front-door
  flows, and default container journeys, including `make docs-live-fast`,
  `make packaging-smoke-front-door`, `make container-default-smoke`, and
  `make container-front-door-smoke` plus Podman siblings where applicable.
- Added `make workflow-lint` as a compatibility alias for the GitHub Actions
  workflow lint gate.
- Added `make security` for local supply-chain SBOM and `pip-audit` checks
  through an isolated `uv` security toolchain.
- Added versioned documentation publishing to GitHub Pages so release docs can
  live under stable paths such as `/0.8.0/` instead of sending installed users
  to moving `main` blob URLs.
- Added the `Qwen/Qwen2.5-7B` lane to declared `supported_experimental`
  coverage and expanded the maintained model inventory and evidence-sweep
  surfaces for the promoted set.
- Added repo-ready deferred candidate surfaces for `open_llama_7b`,
  `opt-1.3b`, `falcon-7b`, `glm-4-9b-chat`, and `distilbert-base-uncased`,
  including shipped presets/calibration configs, tuned reduced-pack
  parameters, and targeted sweep/test coverage.
- Added a bounded `scripts/evidence_packs/run_mini_pack_gate.sh` maintainer
  lane plus remote setup/smoke coverage for narrower evidence-pack recovery
  and promotion checks.
- Added explicit first-touch CLI inventory coverage for `invarlock --version`,
  `invarlock report --help`, and the package-native `invarlock advanced
  runtime-verify --help` surface across docs, tests, and smoke lanes.

### Changed
- Replaced the old assurance toggle with explicit
  `evaluate --execution-mode container|host` and
  `verify --runtime-provenance container|host`, and aligned the
  maintained docs/live-example/workflow surfaces to the same terminology.
- Renamed the public and maintainer bundle surface from proof packs to
  evidence packs across the CLI, contracts, docs, notebooks, scripts, and
  packaged public-evidence recipes.
- Versioned runtime-manifest verifier JSON as `runtime-verify-v1`, moved shipped
  evidence paths to logical `public_evidence/published_basis/...` locations, and
  made packaged runtime-profile provenance portable through `runtime/tiers.yaml`.
- Tightened the installed-wheel, docs-lint, and container maintainer gates
  around the supported public CLI and report paths.
- Refreshed pinned GitHub Actions for Node and uv setup in the CI, docs, and
  repo-hygiene workflows.
- Aligned Ruff pins across `pyproject.toml`, workflow lockfiles, and the
  pre-commit hook so local and CI lint formatting use the same Ruff release.
- Reworked the public onboarding docs around explicit wheel-user, evaluator,
  and repo-maintainer entry points, and replaced repeated report filename
  caveats with a shared artifact model centered on `evaluation.report.json`.
- Moved the runtime-manifest verifier onto the main CLI under
  `invarlock advanced runtime-verify` with the same Typer/Rich-style help and
  output conventions as the rest of the command surface.
- Made `invarlock report explain` accept `--evaluation-report` and resolve the
  linked subject/baseline run reports from bundle provenance when available.
- Promoted the HTML report from a minimal markdown wrapper to a structured
  browser surface with summary chips and quick-link navigation.
- Expanded the machine-readable model inventory to distinguish declared
  support, implemented coverage, usage-only checkpoints, and
  promotion-candidate inventories, while keeping tracked `public_evidence`
  limited to shipped published-basis fixtures.
- Standardized the shared human-readable CLI output layer across `verify`,
  `report html`, `advanced policy`, `advanced evidence-pack`,
  `advanced runtime-verify`, and the top-level `doctor` findings and health
  summaries so status lines, warnings, and detail rows render consistently.
- Refreshed the `evaluate` first-screen banner and the exported HTML report
  shell with clearer visual hierarchy, stronger context framing, and
  dark-mode-aware styling without changing the underlying report content.

### Removed
- Removed the standalone `invarlock-runtime-verify` console script before it
  became a supported public entry point; use `invarlock advanced
  runtime-verify` instead.

### Fixed
- Fixed the evidence-pack clean-prune contract so the clean pruning lane is now
  model-tuned under the generic `prune_clean` scenario name, and retuned the
  Mistral 7B clean prune from 12% to 10% after a real H200 rerun showed that
  the old setting tripped the RMT clean-pass gate.
- Fixed live-example and installed-wheel smoke reliability by reusing writable
  Hugging Face caches, normalizing host-mode docs replay onto smoke-sized
  assets, tightening Python selection, aligning docs-only CI dependencies, and
  stabilizing rerun cleanup/output handling.
- Fixed minimal wheel and advanced evidence-pack verification coverage so
  installed distributions keep covering `doctor`, `verify`, `report html`,
  `advanced runtime-verify`, packaged contract loading, and strict
  evidence-pack verification outside the repo tree.
- Fixed evidence-pack correctness and recovery across MoE edit targeting,
  strict checksum path normalization, large-model baseline-report reuse,
  atomic artifact creation/reuse, and shared causal-model loading for
  non-standard causal checkpoints.
- Fixed evidence-pack remote recovery and mini-pack execution on recheck hosts
  by honoring model-specific queue overrides, normalizing remote repo aliases,
  tightening scenario-manifest drift checks, and hardening resume/setup smoke
  coverage.
- Fixed the `evidence_pack` / `evidence_pack_support` module boundary so the
  helper surface no longer relies on a cyclic runtime import that CodeQL flags
  as an unsafe import-order dependency.
- Fixed shell-harness and maintainer-surface drift by tightening the `python3`
  stub contract, pruning stale ignore/lint/deep-clean paths, and cleaning up
  remaining public runtime-provenance wording and standalone-product phrasing.
- Fixed release-hygiene drift by marking user-visible Makefile helper targets
  phony, documenting every active `pip-audit` allowlist entry, and aligning the
  workflow docs with the Node.js 22.18+ toolchain contract.
- Fixed release-branch workflow coverage so `release/v*` pushes run the normal
  CI, CodeQL, docs-validation, and tiny container smoke gates before tagging,
  while preserving PR docs artifacts before cleanup.
- Fixed report and observability export hardening by escaping report-derived
  HTML before browser rendering, normalizing Prometheus exposition output and
  Pushgateway path segments, and computing exporter success rates from total
  attempts.
- Fixed local release-gate ergonomics by documenting the CI-pinned `actionlint`
  install command used by `make workflow-lint`.
- Fixed typed-surface drift across pairing, dataset-plan, guard-policy,
  runtime-verifier, report-generation, and observability helper surfaces so
  the full `make lint` gate passes cleanly on the release branch.

## [0.7.2] - 2026-04-15

### Changed
- Refactored the test tree around owner-aligned surfaces, including runtime,
  reporting, guards, CLI, integration, lint, and CI buckets, and aligned the
  Makefile and path-contract checks to the normalized layout.
- Switched release publishing to a tag-plus-PyPI flow, removed public
  release-page asset uploads from the release workflow, and aligned release
  verification docs, workflow assertions, and citation metadata to the new
  release surface.
- Tightened scheduled verification and CI smoke coverage for the GPT-2 lane,
  gitleaks workflow assertions, and the maintained dependency pins used by the
  release and verification paths.

### Fixed
- Refreshed the security-sensitive dependency set used by the validated
  workflow surfaces, including `pytest 9.0.3`, `pillow 12.2.0`, and
  `cryptography 46.0.7`.
- Resolved code-scanning findings in the evidence-sweep and proof-pack helper
  scripts, removed unreachable CLI example logic, and completed the remaining
  ruff cleanup needed for the repository verification gate.

## [0.7.1] - 2026-04-12

### Added
- Clarified the minimal-install onboarding path and the exact
  `verify`/`report`/proof-pack command inputs so wheel-only users can validate
  artifacts without cloning the repository.
- Added the released
  `invarlock-<version>-public-contract-bundle.tar.gz` asset, including the
  manifest/schema inventory users can verify without cloning the
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

- Removed test/scientific-only packages from the base runtime where they were
  not required, moved MI-probe dependencies behind narrower extras, and
  refreshed workflow lockfiles to match the tightened CI and release surfaces,
  including `aiohttp==3.13.5` and `linkchecker==10.6.0` on the shipped docs
  and security paths.
- Pinned typed and packaging workflow lockfiles to include `mypy==1.20.0` and
  `wheel==0.46.3` so the new typed-surface and minimal wheel-install gates run
  under the shipped CI surface instead of relying on implicit tool installs.
- Bumped docs spell-check tooling from `cspell` `9.7.0` to `10.0.0`.
- Added pinned CI/release workflow tooling for `actionlint` `v1.7.7` and
  `gitleaks` `v8.30.0` as part of the new fail-closed workflow and
  supply-chain gates.
- Bumped the release publish action to `pypa/gh-action-pypi-publish`
  `1.14.0`.
- Raised the docs CI Node runtime from `18` to `22` so the refreshed
  spell-check toolchain remains on a supported engine.

- Refreshed workflow, security, CLI, config, and contract docs to match the
  new assurance model, supply-chain gates, fail-closed report validation, and
  light-import-safe module boundaries.
- Documented the standalone public contract bundle and the minimal-install
  verify/report/proof-pack onboarding path.
- Clarified secure-default runtime docs and user-facing guidance to describe an
  OCI container engine requirement first, keep Podman/Docker examples explicit,
  and scope Docker-only language to the local `act` workflow.

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
- Fixed the Python 3.13 advanced install surface so shipped `gptq`/`advanced`
  extras no longer pull unsupported upstream `auto-gptq` builds on the PR
  supply-chain path; GPTQ remains available on the narrower Linux stacks that
  upstream packaging currently supports.

## [0.7.0] - 2026-04-09

### Added
- Added first-class GPT-OSS causal support and pilot Ministral 3 8B/14B
  presets, calibration configs, and support-matrix/catalog coverage.
- Added a CUDA-capable container runtime image path for GPU hosts, smoke-sized
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

- Bumped workflow and dev-security dependencies including `cryptography` to
  `46.0.7`, `ruff` to `0.15.9`, `katex` to `0.16.45`, and refreshed the pinned
  CodeQL action state and Dependabot handling.
- Added a `cu128` runtime-image lockfile for the CUDA container runtime path.

- Refreshed maintainer and user docs around profile-driven token floors, smoke
  strategy, calibration surfaces, and the current host / container
  operating model.

### Fixed
- Fixed `quant_rtn` and report-generation fail-closed behavior so noop edits,
  failed subject runs, malformed primary-metric outputs, and invalid baseline
  pairing states no longer emit misleading downstream artifacts.
- Fixed delegated config execution by routing runtime delegation and
  calibration through a package-internal config-runner module instead of a
  hidden public CLI command, and aligned container vs host tiny-smoke
  semantics by forwarding and resolving `tiny_relax` provenance consistently.
- Fixed host and container live-demo paths across GPT-2 and 14B model
  flows, including CUDA runtime selection, HF cold-cache fallback handling,
  non-GPT-2 layer-count reporting, regenerated report runtime manifests, and
  primary-metric acceptance handling.
- Fixed developer-path regressions around Python interpreter selection, the
  mypy gate, Dependabot/CodeQL updater stability, ClusterFuzzLite Docker
  inputs, and repo gate coverage expectations.
- Reduced repo-wide static debt by removing remaining source `type: ignore`
  suppressions, narrowing broad exception fallbacks, and hardening
  observability, eval, calibration, and adapter boundary paths.

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
- Unified the old assurance UX under explicit host/container runtime modes
  across `evaluate`, `verify`, and `report verify`, replacing the earlier
  split between `--mode local` and the explicit unverified-provenance verify
  bypass.
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

- Bumped workflow `aiohttp` from `3.13.3` to `3.13.4` and pinned runtime/fuzz
  builder inputs more strictly for deterministic post-release smoke and
  packaging behavior.
- Updated Hugging Face runtime requirements and locks to `transformers==5.5.0`
  for Gemma 4 support.

- Refreshed CLI/reference/user-guide pages, shipped preset comments, and
  notebooks to teach the new assurance UX and the current host verify
  pattern consistently.
- Documented the proof-pack reviewer-summary surface and the writable dataset
  cache fallback path in the relevant reference and user-guide pages.
- Updated support and dataset docs to document the Gemma 4 E2B pilot lane and
  the new `vision_text` image-text evaluation flow.

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
- Fixed host evaluation and verification ergonomics so local host runs,
  report verification, and docs/notebook examples all use the same explicit
  assurance vocabulary.

## [0.5.1] - 2026-04-02

### Added
- Added a lightweight container push smoke lane built around
  `sshleifer/tiny-gpt2`, a local JSONL fixture, and the new `Tiny Container
  Smoke` workflow.
- Added a heavier GPT-2 canary preset and workflow for scheduled and manually
  dispatched end-to-end provenance checks.
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
  critical evaluation, runtime provenance, proof-pack verification,
  determinism, registry, invariants, run orchestration, and reporting flows.
- Continued the shell/core split so CLI shells hand policy and owner logic to
  typed core and reporting helpers instead of owning fallback decisions.
- Decomposed the largest owner modules across runtime security, run
  orchestration, run execution, report building, verification checks, and
  proof-pack handling into smaller implementation files with stronger guardrail
  coverage.
- Converged runtime-manifest verification onto a single package-native Python
  path so product provenance, `advanced runtime-verify`, and
  `make runtime-verify` all exercise the same verifier implementation.
- Reworked proof-pack signing and verification around the same package-native
  Ed25519 manifest-signature contract used by the installed CLI and shell
  harnesses.
- Hardened container smoke and tiny-matrix flows so they rebuild the local
  runtime image when needed, prefer the repo-selected interpreter, bootstrap
  the CPU-only Hugging Face stack deterministically, and keep local and CI
  runtime behavior aligned.
- Ratcheted refactored split owners to stricter 95% and 100% per-file coverage
  thresholds where the current suite supports it.

- Patched vulnerable workflow locks and tightened smoke-workflow dependency and
  asset caching behavior for more deterministic CI execution.
- Updated verification and coverage gates so the packaged verifier and the
  newly split owner modules are exercised directly in local and CI runs.
- Bumped workflow and release security pins including `cryptography` to
  `46.0.6`, `pygments` to `2.20.0`, and the Sigstore GitHub Action used by the
  release workflow.
- Bumped `aiohttp` from `3.13.3` to `3.13.4` in workflow requirement locks and
  landed the corresponding Dependabot-equivalent fix on `staging/next`.

- Refreshed docs to match the post-`v0.5.0` architecture and operations model,
  including the shell/core redesign, current evaluate contract, and updated
  report-artifact guidance.
- Updated maintainer smoke notes to distinguish the push-gated tiny container
  smoke from the heavier GPT-2 canary workflow.
- Documented the Python-only runtime-verifier contract and removed the obsolete
  external-verifier environment-variable guidance.
- Updated the architecture/security references so runtime provenance
  ownership now explicitly points at the package-native verifier instead of an
  external-binary model.

### Removed
- Removed remaining compatibility surfaces that no longer fit the stabilized
  architecture, including legacy command shims, reporting facades, owner-layer
  patch-sync wrappers, the retired legacy RMT module, stale lazy export
  placeholders, and other shell-leaking or test-only indirections that had
  survived earlier migrations.
- Removed the repo-local Rust runtime verifier crate and the
  `INVARLOCK_RUNTIME_VERIFIER` product override so runtime provenance now has
  a single package-native verifier path.
- Removed the proof-pack `gpg` signing and verification path in favor of the
  package-native Ed25519 manifest-signature flow.

### Fixed
- Delegated and containerized evaluation reports now emit container execution
  provenance into their runtime manifests.
- Runtime provenance and proof-pack verification now fail closed by default on
  artifacts without verified provenance, mutable runtime-image refs without digests, and
  unsigned or unverifiable proof-pack manifests unless the explicit unverified-provenance
  override is set.
- Runtime provenance now uses the packaged Python runtime-manifest verifier
  directly, removing path-dependent behavior from product verification.
- Tiny container smoke exports now write to host-writable paths, and unsigned
  proof-pack smoke runs use an explicit unverified-provenance override instead of
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
- Fixed the runtime image and smoke bootstrap paths so container-backed Linux smoke
  runs install the CPU-only torch stack deterministically, reuse writable HF
  caches, and no longer depend on stale local runtime images or host `PATH`
  quirks.
- Restored 100% proof-pack shell-harness coverage and fixed warning-path shell
  helpers that had been swallowing finalize, evaluate, or verify failures.

## [0.5.0] - 2026-03-25

### Added
- Added an offline release-verification bundle generator and reference docs for
  auditing release artifacts without network access.
- Added public model-family and runtime-manifest contracts, packaged contract
  artifacts in wheels, and contract-sync automation for shipped distributions.
- Added stronger proof-pack manifest and provenance tooling, package-native
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

- Bumped `ruff` from `0.15.6` to `0.15.7`.
- Bumped `actions/cache` from `5.0.3` to `5.0.4`.
- Bumped `actions/download-artifact` from `7` to `8`.
- Bumped `actions/upload-artifact` from `5` to `7`.
- Bumped `katex` from `0.16.28` to `0.16.38`.
- Bumped `flatted` from `3.4.1` to `3.4.2`.

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

### Removed
- Removed the `QwQ-32B` model lane from the repo, including its maintained
  catalog/support references and its shipped preset and calibration configs.

### Fixed
- Hardened CLI backend, doctor, plugin, and verification checks, including
  safer remote-code defaults, plugin catalog/install surfaces, and
  release-profile overhead enforcement.
- Fixed the CLI runtime-verifier test shim to use the active test interpreter,
  which keeps nested verify/proof-pack provenance tests aligned with the
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

- Added docs spellcheck tooling and pinned repo formatter/build tooling for
  reproducible local and CI verification.
- Bumped GitHub `actions/cache` to v5.

- Renamed and tightened assurance notes, narrowed the public claim surface, and
  expanded reference docs for contracts, calibration, proof packs, and policy
  provenance.
- Refreshed README and test/example wording to match the stabilized
  evaluate/report/verify contract and current repo structure.
- Updated public docs to describe the canonical five-stage guard chain,
  including the terminal invariants pass shown by current CLI output.

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

## [0.3.12] - 2026-02-27

### Added
- Coverage thresholds now enforce split-module branch floors for critical CLI/reporting paths.

### Changed
- Refactored CLI run/report builder flows into smaller modules and injected explicit run-command dependencies.
- Tightened exception-hygiene handling across `run`, `report`, and `doctor` command paths.
- Repository housekeeping now excludes research pipeline artifacts from tracked source files.

- Bumped `katex` from `0.16.27` to `0.16.28`.
- Bumped `markdownlint-cli2` from `0.20.0` to `0.21.0`.

- Replaced remaining legacy assurance-label wording with evaluation terminology
  in docs.
- Clarified calibration policy/preset guidance and aligned ASCII diagram connector formatting.

### Fixed
- Hardened config include resolution and plugin subprocess path handling in CLI flows.
- Normalized doctor/plugin command exit semantics for stable profile-specific failure behavior.
- Strengthened reporting fail-closed schema behavior with network refcounting and schema patch hardening.
- Hardened overhead/tiny-relax guard handling and config/profile gate-control enforcement.
- Made observability alerting import-safe when `requests` is unavailable.
- Hardened docs command runner security checks and enforced pip-audit execution.

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

## [0.3.10] - 2026-02-08

### Added
- Proof packs: new guard showcase suite and expanded scenario coverage (scenario filtering/errors-only mode, suite-scoped scenarios, and model override support).
- Proof packs: new demo/probing artifacts (verdict tables generator, VE `ve_probe` sidecar, and additional RMT/spectral/variance showcase injections).
- CI: add Python 3.12 smoke and scheduled weekly verification.

### Changed
- CI: make release/CI verification more reproducible (deterministic `verify-full`) and improve local `act` ergonomics.
- Docs CI: allow on-demand runs via `workflow_dispatch`.
- Proof packs: strengthen “evidence signal” outputs and tighten fail-closed behavior for verdict/task failures.

- Proof packs: harden dependency preflight and net-enabled install behavior (require `huggingface_hub` where needed; ensure `accelerate` is available).

- Docs: fix markdown link fragments.
- Proof packs: clarify evidence vs proof-grade posture and document new artifacts (intervention summary + VE probe sidecar).

### Fixed
- Guards/variance and VE: improve Mixture-of-Experts compatibility (fused expert weight layouts, broader VE layer discovery, and Mixtral `block_sparse_moe` support) and harden variance defaults/probes.
- Proof packs: improve reliability and determinism of demos (retuned injections/detectors, more robust packaging of probe sidecars, and safer behavior when reports exist but evaluation exits nonzero).
- Assurance: close verification/baseline evidence gaps and tighten audit coverage.
- CLI/eval/tests: stabilize CI help-smoke output, accept extra `load_dataset` kwargs, and allow warn-only determinism.

## [0.3.9] - 2026-02-03

### Changed
- README: refresh above-the-fold header layout, including a banner-sized logo lockup and centered badges.
- Branding: make the README logo lockup more logomark-dominant and add a dark-mode logo variant.
- Branding: logomark-only avatar asset (`docs/assets/invarlock-mark.svg`) for GitHub profile usage.
- Documentation CI can now be run on demand via `workflow_dispatch`.

### Fixed
- CI: update workflow test paths after the report artifact rename.
- Tests: apply ruff-format to warning suppression coverage test.
- CLI: `invarlock report explain` drift gate now prints the resolved drift band (no hard-coded threshold).
- CLI: align `invarlock report` “ARTIFACTS” block so artifact paths start in the same column.
- Observability: CPU health check no longer fails when platform CPU count is unavailable.
- Proof packs: config generator can emit configs to stdout without relying on `/dev/stdout`.
- Tests: stabilize the end-to-end pipeline memory management integration test with a PyTorch warm-up.
- Tests: build-wheel packaging test uses `build --no-isolation` to avoid network in offline environments.
- Tests: import-safety venv integration test skips cleanly when network is unavailable.

## [0.3.8] - 2026-02-02

### Added
- CLI: `--version` / `-V` flag (alias of `invarlock version`) to print the InvarLock version (plus report schema version when available).
- `invarlock evaluate` summary now includes total runtime and confidence interval.
- Proof packs: `verify_pack.sh --strict` (or `PACK_STRICT_MODE=1`) to fail closed on missing/invalid GPG signatures and unexpected pack contents.

### Changed
- **Breaking:** Rename legacy evaluation artifacts to “report” across artifacts,
  docs, scripts, notebooks, and Python API surfaces.
- **Breaking:** CLI terminology unified on `evaluate`.
- Config: reject legacy HF v4 load keys `model.torch_dtype`, `model.load_in_8bit`, and `model.load_in_4bit`; use `model.dtype` and/or `model.quantization_config`.
- Evaluation report bundle filenames updated (JSON: `evaluation.report.json`, Markdown: `evaluation_report.md`).
- Presets: bump default WikiText-2 dataset seed for the causal LM preset from `42` → `43`.
- Proof packs: `manifest.json` records `checksums_sha256_digest` (sha256 of `checksums.sha256`) and may record `signing_key_fingerprint` when signed.

- Require `transformers>=5.0.0` and `huggingface_hub>=1.0.0`.

- Update guides and notebooks for evaluation reports and renamed commands/pages.
- README: add logo, community links, citation snippet, limitations, and quickstart output excerpt.
- Drop legacy Transformers v4 config key documentation and fix minor formatting/typos.

### Fixed
- HuggingFace/Transformers v5 compatibility: migrate load contracts and use `dtype=` where required.
- Reduce noisy HuggingFace/Transformers warnings in `ci`/`release` CLI output.
- Adapters: snapshot config serialization no longer emits deprecated attributes.
- Scripts: CLI example validator ignores internal tool dirs and supports external paths.
- CLI: keep `invarlock calibrate` import-safe so docs/example validation can run without torch installed.
- Proof packs: fix `verify_pack.sh` legacy report discovery for nested
  `evaluation.report.json` files.
- Proof packs: close a tamper-evidence gap by binding `checksums.sha256` to the signed manifest (and enforcing “no extra files” in strict verification).

## [0.3.7] - 2026-01-22

### Added
- Role-based HuggingFace adapters with updated auto-routing (replaces model-name adapters).
- Proof packs: v2 pack layout, scenarios manifest, and assurance verdict generation.
- CLI flags: `invarlock run --edit-label` and baseline-report reuse on the
  retired evaluation command.
- CI notebook smoke runner (`scripts/docs/verify_notebooks_smoke.py`).
- Task-metric overrides, richer telemetry snapshots/reports, and CLI
  progress/NO_COLOR output refinements for longer-running evaluation flows.

### Changed
- Proof-pack workflows hardened: baseline-report reuse, calibrate-only behavior, tuned-params hygiene, and improved task sizing/memory planning.
- Legacy evaluation artifact rendering was refreshed with revamped Markdown
  output, richer HTML/glossary support, and updated report terminology.
- Presets/overlays updated for new adapter roles and additional model families.
- CI: bump `actions/download-artifact` to v7; remove the legacy B200 backend validation harness.

- Expanded and consolidated guides across CLI, configs, datasets, guards, proof packs, and notebooks.

### Fixed
- Adapters: Mixtral support, improved auto-detection, and hardened causal describe/weight tying.
- Proof packs: enforce CI floor constraints, mitigate OOM/missing-tensors cases, and make verification more resilient.
- Reporting/eval: avoid duplicate synthetic samples and preserve primary-metric drift band handling.

## [0.3.6] - 2026-01-13

### Added
- Measurement contracts for guard estimators (approximation-only, GPU/MPS-first) recorded in reports and enforced by `invarlock verify --profile ci|release`.
- Evidence pack suite workflow split: `scripts/evidence_packs/run_suite.sh --calibrate-only` (stop after preset generation) and `--run-only` (resume remaining tasks).
- Evidence pack suite knob for controlled experiments: `PACK_GUARDS_ORDER`.
- Added the primary-metric tail gate end to end, including runner evaluation,
  report rendering, and `explain-gates` visibility.

### Changed
- Runtime configuration was made canonical and expanded beyond `ci` / `release`
  presets so profile-driven runs, overlays, and preset paths stay aligned.
- B200 calibration configs now default to `guards.order: [invariants, variance, invariants]` (drops spectral/rmt) to avoid CPU-bound SVD (`torch.linalg.svdvals` / MKL `sgesdd`) dominating wall time and making GPUs appear idle during calibration.
- B200 calibrated presets now include `guards.order`, and only include `guards.spectral` / `guards.rmt` sections when those guards are enabled (run a smaller follow-up calibration pass if you need spectral caps or an RMT ε).
- B200 bootstrap defaults HuggingFace caches under `${OUTPUT_DIR}/.hf` (override with `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE`) to avoid small root-disk partitions on GPU nodes.
- `invarlock evaluate` now honors `guards.order` when provided by `--preset` (instead of always forcing `["invariants", "spectral", "rmt", "variance", "invariants"]`), so evaluate matches the calibration preset’s intended guard set.

- Bump katex from 0.16.25 to 0.16.27.
- Bump markdownlint-cli2 from 0.19.1 to 0.20.0.

### Fixed
- Fixed calibration and drift-stat edge cases so single-sample statistics and
  no-overlap calibration runs no longer crash under narrow data conditions.
- Fixed B200 scheduling and evaluation reliability on constrained GPU hosts,
  including single-GPU runs, restored edit evaluation flow, and corrected
  metric parsing for generated reports.
- Fixed degraded primary-metric and legacy report handling so pairing
  mismatches, non-finite metrics, and degraded verdicts surface correctly
  through the generated artifacts.

## [0.3.5] - 2026-01-02

### Added
- Evidence pack bash test suite (`scripts/evidence_packs/tests/*`, `scripts/evidence_packs/tests/run.sh`) with deterministic command mocks and optional branch/line coverage checks.
- Evidence pack runtime helpers (`scripts/evidence_packs/lib/core/runtime.sh`) plus pack build/verify helpers (`scripts/evidence_packs/run_pack.sh`, `scripts/evidence_packs/verify_pack.sh`) to capture artifacts during long runs.
- Perplexity token-id sanitization to mask out-of-range IDs (and ignore them in labels) instead of triggering device-side asserts.

### Changed
- WikiText-2 window stratification now uses a deterministic offline byte-level n-gram scorer (replaces the GPT‑2 scorer) to keep window selection stable across model families and avoid implicit model downloads.
- B200 validation suite is dynamic-scheduling only; dependency promotion is centralized to reduce queue lock contention and improve throughput.
- B200 generated configs default to `guards.order: [invariants, rmt, variance]` to avoid slow CPU SVD during calibration; spectral caps are not produced unless you re-enable spectral calibration separately.
- B200 bootstrap defaults HuggingFace caches under `${WORK_DIR}/hf_home` to avoid small root-disk partitions on GPU nodes.

- Updated CLI/dataset/env-var references for the new difficulty scorer and removal of `INVARLOCK_SCORES_BATCH_SIZE`.

### Removed
- `INVARLOCK_SCORES_BATCH_SIZE` (the WikiText‑2 difficulty scorer no longer batches on device).

### Fixed
- B200 harness: treat 30B+ models as “large” for overhead-skip heuristics to avoid double-loading stalls.

## [0.3.4] - 2025-12-28

### Added
- Chunked snapshot/restore support for HF adapters to reduce peak memory during retries.
- Evidence pack workflow helpers (run_suite + scheduler/queue utilities + model creation tooling).

### Changed
- CI/Release baseline pairing is fail-closed: `invarlock run --baseline ...` now requires valid `evaluation_windows` evidence and enforces dataset/tokenizer/masking parity.
- CI/Release report generation now requires `paired_windows` evidence and rejects non-perfect window pairing.

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
- Guard metric impact reporting introduced paired bare/guarded primary-metric
  measurements; primary metric `display_ci` is aligned with log-space CI for
  perplexity metrics.
- B200 validation workflow upgraded to v2.1.0 with dynamic scheduling, GPU lock management,
  and expanded task orchestration scripts.

- Expanded B200 validation guide with v2.1.0 workflow details and scheduler/queue notes.
- Assurance docs, CLI guidance, and environment variable references refreshed for new behavior.

### Fixed
- Calibration data slicing now supports iterables with optional materialization and clearer errors.
- Sequence hashing now includes per-sequence lengths to avoid ambiguous digests.
- Variance guard predictive gating improves min-effect and regression reasoning.

## [0.3.2] - 2025-12-14

### Added
- Calibration CLI (`invarlock calibrate`) and runtime modules for policy and guard tuning.
- Determinism utilities and CLI flows to exercise repeatable runs and presets.
- Bench policy regression harness and additional regression tests for guards and reports.
- Benchmark policy regression golden `bench-golden-2025-12-13` (`ae8094204c998fc51bf51052d7d1457d3cdc17bab9bc4785e88c4f07d0234ad3`) tracks guard-effect quality impact, runtime overhead, and memory overhead against silent gate/output shifts.

### Changed
- Guard policies and tier runtime configuration updated to support calibration and determinism flows.
- CLI commands (`run`, `verify`, `doctor`, `explain-gates`) extended with calibration and reporting surfaces.

- Expanded assurance docs for calibration, guard contracts, determinism, and BCA/bootstrap methods.

### Fixed
- Additional edge cases in report reporting, policy utilities, and guard analysis covered and hardened via new tests.

## [0.3.1] - 2025-12-10

### Added
- **INVARLOCK_SKIP_GUARD_METRIC_IMPACT_CHECK env var** - Skip guard metric impact measurement even with ci/release profiles for large models.
- **Configurable PM acceptance range** - Set via preset config or `INVARLOCK_PM_ACCEPTANCE_MIN/MAX` environment variables.
- **Comprehensive evidence pack guide** - New documentation at `docs/user-guide/evidence-packs.md`.

### Changed
- B200 validation scripts updated to v2.0.1 with improved cleanup traps and progress monitoring.

### Deprecated
- `INVARLOCK_TINY_RELAX` for PM acceptance - prefer `INVARLOCK_PM_ACCEPTANCE_MAX` and presets instead.

### Fixed
- **Memory leak in run.py reload fallback** - GPU memory is now freed before reloading models, preventing OOM on 70B+ runs.
- **B200 validation script bugs** - Fixed preset path resolution, model size detection, and error propagation in dynamic scheduling workers.

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
  - Auto-routes to quantized HF adapters based on checkpoint metadata
- **Comprehensive adapter test coverage** (46 new tests)
  - `test_capabilities.py` - QuantizationMethod, QuantizationConfig, ModelCapabilities
  - `test_safe_device.py` - Safe device movement and capability detection
- **Observability module test coverage** (230 new tests across 6 files)
- **Test documentation** - README files for `tests/guards/` and `tests/observability/`

### Changed
- `hf_causal.py`: Uses `_safe_to_device()` instead of direct `model.to()` call
- `invarlock.plugins` AWQ adapter: Uses `_safe_to_device()` with AWQ capabilities
- `invarlock.plugins` GPTQ adapter: Uses `_safe_to_device()` with GPTQ capabilities

- Added quantized adapter section to `docs/reference/model-adapters.md`
  - BNB adapter usage and pre-quantized detection
  - AWQ adapter (Python 3.12 compatible)
  - GPTQ adapter (requires Python 3.10/3.11)
  - Quantization auto-detection flow

### Fixed
- BNB 8-bit model loading error when subject is a saved quantized checkpoint
- Empty sample handling in variance guard (`_safe_mean()` helper)

## Pre-public and import history

The published release history begins at `v0.2.0`, but the repository does not contain a tagged `v0.1.0`. Git history shows two separate `feat: initial public import` roots on December 1, 2025: `v0.2.0` is a standalone one-commit public snapshot, while the continuing release line that leads to `v0.3.0` and later starts from a separate public-import root. Earlier internal development therefore exists only as pre-public foundation and is not represented here as a published semver release.

## [0.2.0] - 2025-12-01

First public GitHub and PyPI release snapshot.

### Added
- Core compare & evaluate pipeline and guard chain for edit‑agnostic robustness reports.
- Evaluation report schema v1 and CLI entry points (including `invarlock evaluate`).
- Torch‑optional core install with optional extras (e.g., `invarlock[hf]`, `invarlock[adapters]`).
- Initial documentation set: quickstart, user guides, and CLI reference.

### Changed
- Until 1.0.0, **minor** releases (0.x.y → 0.(x+1).0) may include breaking changes. Refer to the README and CLI help for the current surface and behavior.
