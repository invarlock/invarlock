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
- `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1`
  - Allow iterables without `__len__`/slicing to be materialized for evaluation.
  - Default: disabled to preserve deterministic, indexable windowing.

## Dataset Preparation

- `INVARLOCK_CAPACITY_FAST=1`
  - Faster capacity estimation in dataset provider (approximate).
- `INVARLOCK_DEDUP_TEXTS=1`
  - Exact-text deduplication before tokenization to reduce duplicates.

## Checkpointing & Snapshots

- `INVARLOCK_SNAPSHOT_MODE={auto|bytes|chunked}`
  - Control snapshot mode (used by CLI `run` and checkpoint utilities).
  - If `bytes` snapshotting fails, the CLI will attempt `chunked` snapshotting when supported; otherwise it falls back to reload-per-attempt.
- `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION=<float>`
  - Tune `auto` mode RAM fraction (default 0.4).
- `INVARLOCK_SNAPSHOT_THRESHOLD_MB=<int>`
  - Minimum model size (MiB) before chunked snapshotting kicks in (default 768).

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

- `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE=1`
  - Allow YAML `!include` to reference files outside the config directory.
  - Default: disabled to prevent accidental reads of unrelated local files.

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
