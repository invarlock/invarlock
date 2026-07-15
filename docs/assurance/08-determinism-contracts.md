# Determinism Contracts

> **Plain language:** If we fix the seed bundle, record dataset/tokenizer
> hashes, reuse baseline IDs for the corresponding subject arms, and keep
> preview/final slices disjoint, evaluation runs should be reproducible within
> float tolerance under the stated backend/version preconditions—and we
> surface those checks in the report.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | State the determinism preconditions and report evidence required for reproducible paired evaluation. |
| **Audience** | Evaluation maintainers, CI/release approvers, and operators comparing run evidence. |
| **Contract scope** | Seed bundle, dataset/tokenizer hashes, paired schedules, backend flags, and drift boundaries. |
| **Source of truth** | `src/invarlock/core/determinism_policy.py`, run/report provenance code, and determinism contract tests. |

## Claim

Fixed seeds, dataset/tokenizer hashes, exact baseline/subject ID reuse,
disjoint preview/final slices, non-overlapping token windows, and a pinned
backend stack reduce avoidable variability and make discrepancies easier to
investigate. They do not guarantee bitwise or universal floating-point
reproducibility, even on nominally similar hardware. Cross-backend and
cross-version results are comparison evidence, not strict reproducibility claims.

## Derivation (sketch)

Evaluation stays deterministic when the following preconditions hold—each item
ties the runtime contract back to reproducible maths:

1. **Seed bundle**: record `{python, numpy, torch}` (plus bootstrap seed under
   `dataset.windows.stats.bootstrap.seed` when bootstrap is used); set framework determinism flags.
2. **Dataset/tokenizer provenance**: store `dataset_hash`, `tokenizer_hash`,
   tokenizer name/version, vocab size, BOS/EOS policy.
3. **Schedule reuse**: edited runs reuse the corresponding baseline
   `window_ids`; enforce `window_match_fraction=1.0`, configured
   `window_overlap_fraction=0.0`, equal preview/final schedule counts, and
   disjoint preview/final ID sets. Equal slice counts do not make those slices
   paired.
4. **Environment flags** (GPU/CI):
   - `torch.use_deterministic_algorithms(True)`
   - `torch.backends.cudnn.benchmark = False`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.set_num_threads(INVARLOCK_OMP_THREADS or 1)` plus matching CPU
     thread-cap environment variables; seed Python, NumPy, and Torch RNGs from
     the same configured seed
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8` (fallback `:16:8` on smaller GPUs)
   - disable TF32: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`
   - `TOKENIZERS_PARALLELISM=false`
   Prefer single-thread CPU for CI or debugging, but allow release scripts to opt into higher thread counts via `INVARLOCK_OMP_THREADS`.

## Runtime Contract

- CI/Release runs hard-fail if a baseline pairing context exists and baseline
  matching is incomplete, the configured stride makes token windows overlap,
  or schedule counts differ. Strict report verification also rejects
  intersecting preview/final ID sets.
- report contains seeds/hashes, pairing metrics, coverage floors, bootstrap
  metadata, and policy tier/digest.

## Observability

- `meta.seeds.{python,numpy,torch}`, `meta.env_flags`, and `meta.determinism`
  (determinism preset + TF32/determinism flags). `provenance.env_flags` records
  backend/library versions for auditability.
- `meta.tokenizer_hash` and `provenance.provider_digest` for dataset/tokenizer provenance.
- `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`
  plus raw `evaluation_windows.{preview,final}.window_ids`.
- `primary_metric.{ratio_vs_baseline,display_ci}` and `dataset.windows.stats.coverage` for counts.
- `artifacts.report_path`, `provenance.{baseline,edited}.report_path`, and `policy_provenance.policy_digest` — reproducibility breadcrumbs.

## Assumptions & Scope

- Applies to inference-only evaluation loops; training/edit algorithms may
  introduce additional nondeterminism governed by their own evidence surfaces.
- Identical seeds, configs, inputs, and backend settings should preserve
  pairings and input/policy digests. Numeric evidence must still be compared
  under a justified tolerance; raw report files can also differ in generated-time
  metadata and timestamped run directories.
- Determinism is best-effort on some backends. The paired-CI identity test checks
  arithmetic consistency inside one report; it is not an empirical same-backend
  rerun tolerance study.
- [Cross-Device Drift Bands](12-device-drift-bands.md) documents configurable
  review thresholds used by `scripts/smoke/check_device_drift.py`; those defaults
  are not proven hardware-wide drift bounds.
- Some hardware backends (e.g., GPUs without deterministic kernels) may exceed
  float tolerances despite the flags; document deviations in the report
  metadata.

## References

- PyTorch. “Reproducibility.” <https://docs.pytorch.org/docs/2.12/notes/randomness.html>
