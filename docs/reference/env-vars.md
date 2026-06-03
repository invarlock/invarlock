# Environment Variables

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Environment-level toggles for network access, evaluation, snapshots, and docs tooling. |
| **Audience** | CLI users and operators tuning runtime behavior. |
| **Scope** | CLI commands and programmatic runs; precedence is setting-specific when env and config/CLI both exist. |
| **Network** | Offline by default; network must be explicitly enabled. |
| **Source of truth** | `docs/reference/env-vars.md`, `src/invarlock/cli/commands/*`, `src/invarlock/core/plugins_inventory.py`, `src/invarlock/runtime_security.py`, `src/invarlock/core/runner.py`. |

## Quick Start

```bash
# Allow model + dataset downloads for a single command
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate --baseline gpt2 --subject gpt2

# Force evaluation device for a one-off compare/evaluate run
INVARLOCK_EVAL_DEVICE=cpu INVARLOCK_ALLOW_NETWORK=1 \
  invarlock evaluate --baseline gpt2 --subject gpt2 --device cpu
```

## Concepts

- **Offline-first**: all network access is opt-in and must be explicitly enabled.
- **Precedence**: when a setting exists in both env + config/CLI, the winner is
  setting-specific (see the matrix below).
- **Auditability**: selected env flags are recorded in `report.meta.env_flags` for
  traceability.

**Precedence (conflict cases)**

1. CLI/config values for assurance-critical policy (strictness, drift/acceptance bands, overhead skip).
2. Env overrides only for explicitly env-scoped toggles (for example, downloads, calibration materialization, and tiny-relax smoke behavior).
3. Packaged defaults when no explicit setting exists.

### Key override matrix

| Setting | Env var | Config/CLI | Winner rule | How to confirm |
| --- | --- | --- | --- | --- |
| Calibration materialize | `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE` | `context.eval.materialize_calibration` / `context.eval.allow_iterable_calibration` | Env wins. | Config shows in `report.context`; env is not recorded. |
| Tiny relax | `INVARLOCK_TINY_RELAX` | `context.run.tiny_relax` | Either opt-in enables tiny-relax run/report policy. | Recorded in run context and surfaced through report validation. |
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
| `INVARLOCK_ALLOW_REMOTE_CODE` | unset | Explicitly allow remote model code execution. |

`INVARLOCK_ALLOW_REMOTE_CODE` is the public environment gate for remote model
code execution. Use `INVARLOCK_ALLOW_REMOTE_CODE=1` for `invarlock evaluate`
when remote code is required; `--allow-remote-code` is exposed on advanced
calibration/config-runner commands that load models directly.

### Evaluation & pairing

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_TINY_RELAX` | unset | Dev/demo compare-evaluate override for tiny smoke runs; records tiny-relax provenance and applies tiny-relax report-policy semantics. Do not use for production assurance. |
| `INVARLOCK_EVAL_DEVICE` | unset | Force evaluation device (`cpu`, `cuda`, `mps`). |
| `INVARLOCK_STORE_EVAL_WINDOWS` | `1` | Store token windows in reports (set `0` to disable). |
| `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE` | unset | Allow materializing iterables lacking `__len__`. |

Bootstrap CI method and replicate counts are controlled by runtime profile,
tier policy, and report policy. There is no public env var that forces BCa.

### Dataset preparation

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_CAPACITY_FAST` | unset | Approximate capacity estimation for quick runs. |
| `INVARLOCK_DEDUP_TEXTS` | unset | Exact-text dedupe before tokenization. |
| `INVARLOCK_HF_DATASETS_CACHE` | unset | Override the writable fallback cache used when HF dataset loads hit a shared-cache lock/permission error. |

### Determinism & performance

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_OMP_THREADS` | `1` | Thread caps for determinism preset. |
| `INVARLOCK_DEBUG_TRACE` | unset | Verbose debug traces for data/eval paths. |
| `INVARLOCK_LIGHT_IMPORT` | unset | Avoid heavy imports for docs/tests. |
| `PACK_DEFER_REPORT_RENDERING` | unset (`1` under `run_pack.sh --release-review`) | Evidence-pack wrapper toggle that skips optional markdown/reviewer rendering during evaluation. |
| `PACK_DEFER_OPTIONAL_REPORT_RENDERING` | unset | Alias for `PACK_DEFER_REPORT_RENDERING`. |

Evidence-pack evaluation-loop toggles are repo-wrapper controls, not public
`invarlock evaluate` defaults. Required JSON reports and sidecars are still
written; `PACK_DEFER_REPORT_RENDERING=1` skips optional rendered review files in
the evaluation hot path.

### Checkpointing & snapshots

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_SNAPSHOT_MODE` | `auto` | `auto`, `bytes`, or `chunked` snapshot strategy. |
| `INVARLOCK_SNAPSHOT_AUTO_RAM_FRACTION` | `0.4` | RAM fraction threshold for `auto` mode. |
| `INVARLOCK_SNAPSHOT_THRESHOLD_MB` | `768` | Size threshold for chunked snapshots. |

### Model export

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_EXPORT_MODEL` | unset | Enable HF export during model-export capable CLI flows. |
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
| `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` | unset | Enable third-party plugin discovery. |
| `INVARLOCK_MINIMAL` | unset | Show minimal plugin list in `invarlock advanced plugins`. |

### Runtime enforcement

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_ALLOW_HOST_EXECUTION` | unset | Advanced/internal host-execution override. Prefer `invarlock evaluate --execution-mode host` for the public compare/evaluate path. |
| `INVARLOCK_CONTAINER_EXECUTION` | unset | Internal recursion guard marking runtime-container execution. |
| `INVARLOCK_CONTAINER_ENGINE` | unset | Force the OCI engine used for default runtime-container execution (`podman` or `docker`). |
| `INVARLOCK_RUNTIME_IMAGE` | unset | Override the OCI image used for containerized model execution. |
| `INVARLOCK_RUNTIME_IMAGE_DIGEST` | unset | Supply the immutable digest recorded into `runtime.manifest.json`. |
| `PACK_RUNTIME_IMAGE_FLAVOR` | `default` | Remote evidence-pack setup helper image selector. Use `quant` on CUDA hosts to build/use `invarlock-runtime:cuda-quant` for containerized `hf_bnb`, `hf_gptq`, `hf_awq`, `hf_torchao`, `hf_hqq`, `hf_quanto`, and `hf_ct` evidence. The quant image uses a pinned CUDA devel base so GPTQModel can JIT-compile kernels with `nvcc`; strict custom-image evidence still requires `INVARLOCK_RUNTIME_IMAGE_DIGEST`. |

### Docs build

| Variable | Default | Purpose |
| --- | --- | --- |
| `INVARLOCK_DOCS_MERMAID` | unset | Enable Mermaid diagrams in MkDocs. |

## Troubleshooting

- **Downloads blocked**: set `INVARLOCK_ALLOW_NETWORK=1` and retry.
- **Multiple container engines installed**: set `INVARLOCK_CONTAINER_ENGINE=podman` or `INVARLOCK_CONTAINER_ENGINE=docker`.
- **HF dataset cache lock/permission errors on local reruns**: set `INVARLOCK_HF_DATASETS_CACHE=/path/to/writable/cache` or let InvarLock retry under its own writable cache.
- **Calibration iterables fail**: use `INVARLOCK_ALLOW_CALIBRATION_MATERIALIZE=1`.
- **Third-party plugins missing**: set `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=1`;
  advanced plugin/calibration commands also expose `--allow-third-party-plugins`.

## Observability

- `report.meta.env_flags` records selected env toggles.
- reports capture telemetry and policy digests derived from these flags.

## Related Documentation

- [CLI Reference](cli.md)
- [Configuration Schema](config-schema.md)
- [Dataset Providers](datasets.md)
- [Guards](guards.md)
