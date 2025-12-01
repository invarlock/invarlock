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
- `INVARLOCK_TINY_RELAX=1`
  - Relax some gates/thresholds in tiny dev demos; used by `doctor` and certificate heuristics.
- `INVARLOCK_EVAL_DEVICE=<device>`
  - Override evaluation device for the main model and the WikiText‑2 difficulty scorer.
    Accepts `cpu`, `cuda`, `cuda:0`, or `mps`. When unset, evaluation uses the model’s
    loaded device (resolved from `model.device` / `--device`).

## Dataset Preparation

- `INVARLOCK_CAPACITY_FAST=1`
  - Faster capacity estimation in dataset provider (approximate).
- `INVARLOCK_DEDUP_TEXTS=1`
  - Exact-text deduplication before tokenization to reduce duplicates.
- `INVARLOCK_SCORES_BATCH_SIZE=<int>`
  - Bound candidate scoring batch size for capacity probing.

## Checkpointing & Snapshots

- `INVARLOCK_SNAPSHOT_MODE={auto|bytes|chunked}`
  - Control snapshot mode (used by CLI `run` and checkpoint utilities).
- `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION=<float>`
  - Tune `auto` mode RAM fraction (default 0.4).
- `INVARLOCK_SNAPSHOT_THRESHOLD_MB=<int>`
  - Minimum model size (MiB) before chunked snapshotting kicks in (default 768).

## Guarding & Evidence

- `INVARLOCK_ASSERT_GUARDS=1`
  - Enable lightweight guard runtime assertions (`guard_assert`).
- `INVARLOCK_EVIDENCE_DEBUG=1`
  - Emit a small `guards_evidence.json` next to reports for audit.
- `INVARLOCK_VALIDATE_LEGACY=1`
  - Enable legacy validation pathways (back-compat tolerance handling).

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
