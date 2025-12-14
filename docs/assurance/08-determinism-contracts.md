# Determinism Contracts

> **Plain language:** If we fix the seed bundle, record dataset/tokenizer
> hashes, and keep the paired window schedule stable, every evaluation run is
> reproducible within float tolerance—and we surface those checks in the
> certificate.

## Claim

With fixed seeds, dataset/tokenizer hashes, and a paired, non‑overlapping
schedule, evaluation is reproducible (within float tolerance) and certificates
are stable.

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

- Runs **abort** for CI/Release if pairing < 1.0, overlap > 0.0, or counts differ.
- Certificate contains seeds/hashes, pairing metrics, and policy tier/digest.

## Observability

- `meta.seeds.{python,numpy,torch}` and `provenance.env_flags`.
- `meta.tokenizer_hash` and `provenance.provider_digest` for dataset/tokenizer provenance.
- `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`.
- `primary_metric.{ratio_vs_baseline,display_ci}` and `dataset.windows.stats.coverage` for counts.
- `artifacts.report_path`, `provenance.{baseline,edited}.report_path`, and `policy_provenance.policy_digest` — reproducibility breadcrumbs.

## Assumptions & Scope

- Applies to inference-only evaluation loops; training/edit algorithms may
  introduce additional nondeterminism not covered here.
- Identical seeds, configs, and backend should yield bit-for-bit identical
  certificates; any divergence is a bug to report.
- Determinism is best-effort on some backends; enforce `|Δ ratio| ≤ 1e-6` when
  regenerating certificates on the **same backend** (see
  `tests/eval/test_certificate.py::test_certificate_ratio_matches_weighted_log_delta`).
- Cross-device drift must stay within the bands listed in
  `docs/assurance/04-guard-contracts.md`; use `scripts/check_device_drift.py` in
  CI to guard the limit.
- Some hardware backends (e.g., GPUs without deterministic kernels) may exceed
  float tolerances despite the flags; document deviations in the certificate
  metadata.
