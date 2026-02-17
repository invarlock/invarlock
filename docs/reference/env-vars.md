# Environment Variables

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Environment-level toggles for network access, evaluation, snapshots, and docs tooling. |
| **Audience** | CLI users and operators tuning runtime behavior. |
| **Scope** | CLI commands and programmatic runs; config values override env when both are set. |
| **Network** | Offline by default; network must be explicitly enabled. |
| **Source of truth** | `docs/reference/env-vars.md`, `src/invarlock/cli/commands/*`, `src/invarlock/core/runner.py`. |

## Quick Start

```bash
# Allow model + dataset downloads for a single command
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate --baseline gpt2 --subject gpt2

# Force evaluation device for a one-off run
INVARLOCK_EVAL_DEVICE=cpu invarlock run -c <config>.yaml --out runs/cpu_smoke
```

## Concepts

- **Offline-first**: all network access is opt-in and must be explicitly enabled.
- **Precedence**: when a setting exists in both env + config/CLI, the winner is
  setting-specific (see the matrix below).
- **Auditability**: selected env flags are recorded in `report.meta.env_flags` for
  traceability.

**Precedence (conflict cases)**

1. CLI/config values for assurance-critical policy (strictness, drift/acceptance bands, overhead skip, tiny relax).
2. Env overrides only for explicitly env-scoped toggles (for example, downloads and calibration materialization).
3. Packaged defaults when no explicit setting exists.

### Key override matrix

| Setting | Env var | Config/CLI | Winner rule | How to confirm |
| --- | --- | --- | --- | --- |
| Calibration materialize | `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE` | `context.eval.materialize_calibration` / `context.eval.allow_iterable_calibration` | Env wins. | Config shows in `report.context`; env is not recorded. |
| Network downloads | `INVARLOCK_ALLOW_NETWORK` | — | Env-only toggle. | Not recorded; rely on env. |
| Offline datasets | `HF_DATASETS_OFFLINE` | — | Env-only toggle. | Not recorded; rely on env. |

### Conflict examples

| Scenario | Result | Fix |
| --- | --- | --- |
| `context.run.skip_overhead_check: true` in `--profile release` | Overhead check is skipped and recorded in `guard_overhead.source`. | Set `context.run.skip_overhead_check: false` for full overhead enforcement. |
| `context.run.tiny_relax: true` | Tiny-relax gating is enabled from config and recorded in `auto.tiny_relax`. | Remove or set to `false` for full policy strictness. |

## Reference

### Network & data

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_ALLOW_NETWORK` | unset | Enable outbound downloads for models/datasets. |
| `HF_DATASETS_OFFLINE` | unset | Force Hugging Face datasets to use local cache only. |

### Model loading

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_TRUST_REMOTE_CODE` | unset | Allow HF adapters to set `trust_remote_code=true`. |

HF adapters also honor `TRUST_REMOTE_CODE_BOOL` and `ALLOW_REMOTE_CODE` for compatibility.

### Evaluation & pairing

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_BOOTSTRAP_BCA` | unset | Prefer BCa bootstrap CIs when sample size allows. |
| `INVARLOCK_TINY_RELAX` | unset | Doctor-only hint for tiny local demos (does not drive assurance gates). |
| `INVARLOCK_EVAL_DEVICE` | unset | Force evaluation device (`cpu`, `cuda`, `mps`). |
| `INVARLOCK_STORE_EVAL_WINDOWS` | `1` | Store token windows in reports (set `0` to disable). |
| `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE` | unset | Allow materializing iterables lacking `__len__`. |

### Dataset preparation

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_CAPACITY_FAST` | unset | Approximate capacity estimation for quick runs. |
| `INVARLOCK_DEDUP_TEXTS` | unset | Exact-text dedupe before tokenization. |

### Determinism & performance

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_OMP_THREADS` | `1` | Thread caps for determinism preset. |
| `INVARLOCK_DEBUG_TRACE` | unset | Verbose debug traces for data/eval paths. |
| `INVARLOCK_LIGHT_IMPORT` | unset | Avoid heavy imports for docs/tests. |

### Checkpointing & snapshots

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_SNAPSHOT_MODE` | `auto` | `auto`, `bytes`, or `chunked` snapshot strategy. |
| `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION` | `0.4` | RAM fraction threshold for `auto` mode. |
| `INVARLOCK_SNAPSHOT_THRESHOLD_MB` | `768` | Size threshold for chunked snapshots. |

### Model export

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_EXPORT_MODEL` | unset | Enable HF export during `invarlock run`. |
| `INVARLOCK_EXPORT_DIR` | unset | Target directory for model export. |

### Guarding & evidence

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_ASSERT_GUARDS` | unset | Enable guard runtime assertions. |
| `INVARLOCK_EVIDENCE_DEBUG` | unset | Emit `guards_evidence.json` for audit. |

Primary-metric gate bounds are profile/config settings (`primary_metric.acceptance_range`
and `primary_metric.drift_band`), not environment overrides.
Strictness/tiny-relax/overhead-skip are also config/profile policy:
`context.eval.strict` / `context.eval.strict_errors`, `context.run.strict_guard_prepare`,
`context.run.tiny_relax`, `context.run.skip_overhead_check`.

### Config loading

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_CONFIG_ROOT` | unset | Override packaged `runtime/` data. |
| `INVARLOCK_ALLOW_CONFIG_INCLUDE_OUTSIDE` | unset | Allow YAML `!include` outside config dir. |

### Reporting & telemetry

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_TELEMETRY` | unset | Emit single-line telemetry summary. |

### Plugins

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_DISABLE_PLUGIN_DISCOVERY` | unset | Disable plugin discovery in `plugins` and `doctor`. |
| `INVARLOCK_MINIMAL` | unset | Show minimal plugin list in `invarlock plugins`. |
| `INVARLOCK_PLUGINS_DRY_RUN` | unset | Force plugin install/uninstall dry run. |

### Docs build

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_DOCS_MERMAID` | unset | Enable Mermaid diagrams in MkDocs. |
| `INVARLOCK_DOCS_EXTRA_JS` | unset | Extra JavaScript URLs for docs build. |

## Troubleshooting

- **Downloads blocked**: set `INVARLOCK_ALLOW_NETWORK=1` and retry.
- **Calibration iterables fail**: use `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1`.
- **Plugin list empty**: unset `INVARLOCK_DISABLE_PLUGIN_DISCOVERY` or `INVARLOCK_MINIMAL`.

## Observability

- `report.meta.env_flags` records selected env toggles.
- reports capture telemetry and policy digests derived from these flags.

## Related Documentation

- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Guards](guards.md)
