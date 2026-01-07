# Environment Variables

This page lists InvarLock environment variables and their purpose. Unless noted,
unset variables default to “off”.

## Network & Data

- `INVARLOCK_ALLOW_NETWORK=1`
  - Enable outbound network access for commands that fetch models/datasets.
  - Default: disabled (offline), set per command to opt in.
- `HF_DATASETS_OFFLINE=1` (from Hugging Face)
  - Make datasets operate fully offline using local cache.

## Evaluation & Pairing

- `INVARLOCK_BOOTSTRAP_BCA=1`
  - Prefer BCa bootstrap for paired Δlog confidence intervals (when sample size allows).
- `INVARLOCK_EVAL_STRICT=0`
  - Allow evaluation failures to soft-fail and emit `metrics.eval_error` instead of aborting.
  - Default: strict (fail-closed).
- `INVARLOCK_TINY_RELAX=1`
  - Relax some gates/thresholds in tiny dev demos; used by `doctor` and certificate heuristics.
- `INVARLOCK_EVAL_DEVICE=<device>`
  - Override evaluation device for the main model.
    Accepts `cpu`, `cuda`, `cuda:0`, or `mps`. When unset, evaluation uses the model’s
    loaded device (resolved from `model.device` / `--device`).
- `INVARLOCK_STORE_EVAL_WINDOWS=0`
  - Disable storing token/attention windows in reports (smaller artifacts; can reduce memory/time on long runs).
  - Default: enabled (stores windows).
- `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1`
  - Allow iterables without `__len__`/slicing to be materialized for evaluation.
  - Default: disabled to preserve deterministic, indexable windowing.

## Dataset Preparation

- `INVARLOCK_CAPACITY_FAST=1`
  - Faster capacity estimation in dataset provider (approximate).
- `INVARLOCK_DEDUP_TEXTS=1`
  - Exact-text deduplication before tokenization to reduce duplicates.

## Determinism & Performance

- `INVARLOCK_OMP_THREADS=<int>`
  - Controls the CPU thread caps used by the determinism preset (CI/Release), including `torch.set_num_threads` and common BLAS/OMP thread env vars.
  - Default: `1`.
- `INVARLOCK_DEBUG_TRACE=1`
  - Emit verbose debug traces from dataset/evaluation code paths (high volume).
  - Default: disabled.
- `INVARLOCK_LIGHT_IMPORT=1`
  - Lightweight import mode for docs/tests; avoids heavy optional imports and some side effects during CLI import.
  - Default: disabled.

## Checkpointing & Snapshots

- `INVARLOCK_SNAPSHOT_MODE={auto|bytes|chunked}`
  - Control snapshot mode (used by CLI `run` and checkpoint utilities).
  - If `bytes` snapshotting fails, the CLI will attempt `chunked` snapshotting when supported; otherwise it falls back to reload-per-attempt.
- `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION=<float>`
  - Tune `auto` mode RAM fraction (default 0.4).
- `INVARLOCK_SNAPSHOT_THRESHOLD_MB=<int>`
  - Minimum model size (MiB) before chunked snapshotting kicks in (default 768).

## Model Export

- `INVARLOCK_EXPORT_MODEL=1`
  - In `invarlock run`, export a Hugging Face-loadable model directory (if the adapter supports it).
  - Default: disabled.
- `INVARLOCK_EXPORT_DIR=<path>`
  - Export destination (absolute or relative to the run directory). Used when export is enabled.

## Guarding & Evidence

- `INVARLOCK_ASSERT_GUARDS=1`
  - Enable lightweight guard runtime assertions (`guard_assert`).
- `INVARLOCK_GUARD_PREPARE_STRICT=0`
  - Allow guard `prepare()` failures to log and continue instead of aborting.
  - Default: strict (fail-closed).
- `INVARLOCK_EVIDENCE_DEBUG=1`
  - Emit a small `guards_evidence.json` next to reports for audit.

## Guard Overhead & Primary Metric

- `INVARLOCK_SKIP_OVERHEAD_CHECK=1`
  - Skip guard-overhead measurement even in `ci`/`release` profiles to avoid double-loading large models.
- `INVARLOCK_PM_ACCEPTANCE_MAX=<float>`
  - Upper bound for primary-metric acceptance (default 1.10). Set to 1.15 for slight drift allowance.
- `INVARLOCK_PM_ACCEPTANCE_MIN=<float>`
  - Lower bound for primary-metric acceptance (default 0.95) when enforcing a symmetric ratio band.

## Config Loading

- `INVARLOCK_CONFIG_ROOT=<path>`
  - Override packaged runtime data (`runtime/tiers.yaml`, `runtime/profiles/*.yaml`, etc.). When set, InvarLock checks `$INVARLOCK_CONFIG_ROOT/runtime/...` before the packaged `invarlock._data.runtime` resources.
- `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1`
  - Allow YAML `!include` to reference files outside the config directory.
  - Default: disabled to prevent accidental reads of unrelated local files.

## Reporting

- `INVARLOCK_TELEMETRY=1`
  - Print a one-line summary telemetry line (also stored in certificates under `telemetry.summary_line`).

## Plugins

- `INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1`
  - Disable plugin discovery in `invarlock plugins` and `invarlock doctor` (useful for docs/tests or minimal environments).
- `INVARLOCK_MINIMAL=1`
  - Show a minimal plugin list (hide core/built-in adapters) in `invarlock plugins`.
- `INVARLOCK_PLUGINS_DRY_RUN=1`
  - Force `invarlock plugins-install` / `invarlock plugins-uninstall` to behave as `--dry-run` even when `--apply` is set.

## Documentation Build

- `INVARLOCK_DOCS_MERMAID=1`
  - Enable Mermaid diagrams (mermaid2 plugin). Default is disabled to avoid
    network checks during strict/offline builds.
- `INVARLOCK_DOCS_EXTRA_JS='["<url>", "<url>"]'`
  - Provide a YAML/JSON list of JavaScript URLs to inject (e.g., MathJax,
    Polyfill). By default, no CDN scripts are included. Example:

    ```bash
    INVARLOCK_DOCS_MERMAID=1 \
    INVARLOCK_DOCS_EXTRA_JS='["https://polyfill.io/v3/polyfill.min.js?features=es6","https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"]' \
    mkdocs build --strict
    ```

## Notes

- All network-related features in docs are opt-in by environment variable.
- The CLI remains offline by default; explicitly set `INVARLOCK_ALLOW_NETWORK=1`
  per command when you need downloads.
