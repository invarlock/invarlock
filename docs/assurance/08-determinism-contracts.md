# Determinism Contracts

> **Plain language:** If we fix the seed bundle, record dataset/tokenizer
> hashes, and keep the paired window schedule stable, evaluation runs should be
> reproducible within float tolerance under the stated backend/version
> preconditions—and we surface those checks in the
> report.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | State the determinism preconditions and report evidence required for reproducible paired evaluation. |
| **Audience** | Evaluation maintainers, CI/release reviewers, and operators comparing run evidence. |
| **Contract scope** | Seed bundle, dataset/tokenizer hashes, paired schedules, backend flags, and drift boundaries. |
| **Source of truth** | `src/invarlock/core/determinism_policy.py`, run/report provenance code, and determinism contract tests. |

## Claim

With fixed seeds, dataset/tokenizer hashes, a paired non-overlapping schedule,
and a pinned backend stack, evaluation should be reproducible within float
tolerance on the same backend. Cross-backend and cross-version results are
empirical drift checks, not strict reproducibility claims.

## Derivation (sketch)

Evaluation stays deterministic when the following preconditions hold—each item
ties the runtime contract back to reproducible maths:

1. **Seed bundle**: record `{python, numpy, torch}` (plus bootstrap seed under
   `dataset.windows.stats.bootstrap.seed` when bootstrap is used); set framework determinism flags.
2. **Dataset/tokenizer provenance**: store `dataset_hash`, `tokenizer_hash`,
   tokenizer name/version, vocab size, BOS/EOS policy.
3. **Schedule reuse**: edited runs reuse baseline `window_ids`; enforce
   `window_match_fraction=1.0`, `window_overlap_fraction=0.0`, equal counts.
4. **Environment flags** (GPU/CI):
   - `torch.use_deterministic_algorithms(True)`
   - `torch.backends.cudnn.benchmark = False`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.set_num_threads(INVARLOCK_OMP_THREADS or 1)` and mirror to NumPy /
     Python RNG
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8` (fallback `:16:8` on smaller GPUs)
   - disable TF32: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`
   - `TOKENIZERS_PARALLELISM=false`
   Prefer single-thread CPU for CI or debugging, but allow release scripts to opt into higher thread counts via `INVARLOCK_OMP_THREADS`.

## Runtime Contract

- CI/Release runs hard-fail if a baseline pairing context exists and pairing is
  incomplete, windows overlap, or counts differ.
- report contains seeds/hashes, pairing metrics, coverage floors, bootstrap
  metadata, and policy tier/digest.

## Observability

- `meta.seeds.{python,numpy,torch}`, `meta.env_flags`, and `meta.determinism`
  (determinism preset + TF32/determinism flags). `provenance.env_flags` records
  backend/library versions for auditability.
- `meta.tokenizer_hash` and `provenance.provider_digest` for dataset/tokenizer provenance.
- `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`.
- `primary_metric.{ratio_vs_baseline,display_ci}` and `dataset.windows.stats.coverage` for counts.
- `artifacts.report_path`, `provenance.{baseline,edited}.report_path`, and `policy_provenance.policy_digest` — reproducibility breadcrumbs.

## Assumptions & Scope

- Applies to inference-only evaluation loops; training/edit algorithms may
  introduce additional nondeterminism governed by their own evidence surfaces.
- Identical seeds, configs, and backend should yield identical numeric evidence,
  pairings, hashes, and policy/provenance digests after normalizing volatile
  artifact paths and timestamps. Raw report files can differ in generated-time
  metadata and timestamped run directories.
- Determinism is best-effort on some backends; enforce `|Δ ratio| ≤ 1e-6` when
  regenerating reports on the **same backend** (see
  `tests/reporting/policy/test_report_paired_ci_identity.py::test_paired_ci_identity_holds`).
- Cross-device drift must stay within the bands listed in
  [Cross-Device Drift Bands](12-device-drift-bands.md); use
  `scripts/smoke/check_device_drift.py` in CI to guard the limit.
- Some hardware backends (e.g., GPUs without deterministic kernels) may exceed
  float tolerances despite the flags; document deviations in the report
  metadata.

## References

- PyTorch. “Reproducibility.” <https://docs.pytorch.org/docs/2.12/notes/randomness.html>
