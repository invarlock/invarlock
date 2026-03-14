# Assurance Case Overview (v1.0)

> **TL;DR:** InvarLock evaluates whether **weight edits** (quantization, pruning, etc.) regress a model beyond defined bounds. It does **not** evaluate content safety, alignment, or deployment security. The assurance case covers: (1) paired primary metrics with bootstrap CIs, (2) the canonical five-stage guard chain (`invariants` pre, `spectral`, `RMT`, `variance`, `invariants` post), (3) deterministic evaluation with full provenance. Each claim has tests and report evidence.

> **Plain language:** This overview lists every assurance claim, the evidence we ship with the repo, and the runtime contracts that enforce each claim in production.

This note enumerates the explicit **assurance claims** the toolkit makes, the
**evidence** shipped in-tree, and the **runtime contracts** that enforce each
claim. Each claim must have:

If you need definitions for guard terms (kappa threshold, epsilon band, window
pairing), see the [Glossary](glossary.md).

1) a short argument/derivation (“Evidence”), and
2) a **test or contract** that fails fast when assumptions are violated
   (“Runtime enforcement”).

We also list **observability**—the report fields that let reviewers verify
the claim.

## Scope, assumptions, and non‑goals

InvarLock’s assurance case is intentionally narrow. It is about **regression risk
from weight edits relative to a chosen baseline under a specific configuration**,
not about global model safety.

### In scope

- Structured or quantization‑style **weight edits** applied to an existing model
  (baseline vs edited subject).
- **Paired primary metrics** (ppl/accuracy) on calibrated evaluation windows,
  with log‑space pairing and BCa bootstrap CIs.
- **GuardChain** behavior: invariants, spectral, RMT, and variance guards that
  detect structural breakage, unstable weights, outlier growth, and harmful
  variance shifts introduced by the edit.
- **Determinism and provenance** for the evaluation run: seeds, datasets,
  tokenizers, pairing schedules, and policy configuration reflected in the
  report.
- Execution on **Linux/macOS** environments using the pinned HF/PyTorch stack
  and profiles documented in the configs and docs.

### Out of scope (non‑goals)

- Preventing or detecting **content harms** (toxicity, bias, jailbreaks),
  prompt‑level attacks, or alignment failures in general use.
- Guaranteeing safety for **unrelated training changes**, new datasets, or new
  architectures that fall outside the calibrated families and tiers.
- Enforcing infrastructure or deployment hardening (authz, data governance,
  access control); these live outside the InvarLock runtime.
- Guaranteeing correctness on environments outside the stated support matrix
  (e.g., native Windows, custom CUDA stacks, arbitrary dependency versions).

The table below should be read with this scope in mind: each row is a claim
about **paired evaluation and guard behavior for weight edits** under the
documented tiers and environments, not a universal guarantee about model safety.

> For the end-to-end validation protocol (Step-0 through Step-8 reproducibility and guard overhead checks), see the methodology overview in the docs.

| Claim | Evidence | Runtime enforcement | Observability (report v1.0) | Assumptions & scope |
|------|----------|---------------------|----------------------------------|---------------------|
| Paired ratios are computed in **log space**, **token‑weighted**, then re‑exponentiated. | `docs/assurance/01-eval-math-derivation.md` | The report pairs windows and enforces `ratio_ci == exp(logloss_delta_ci)` within tolerance; see tests `tests/eval/test_report_builder.py::TestMakeEvaluationReport::test_pm_preview_final_ratio_identity` and `tests/core/test_bootstrap.py::test_compute_paired_delta_and_ratio_ci_consistency`. | `primary_metric.{ratio_vs_baseline,display_ci}`, `dataset.windows.stats.{paired_windows,window_match_fraction,window_overlap_fraction}`. | Windows are **paired**, **non‑overlapping**; token counts are known. BCa bootstrap used on paired ΔlogNLL; if all windows equal length, weighting reduces to simple mean. |
| Tier-specific **primary metric** gates keep edits within acceptance bands (Balanced ≤ 1.10×, Conservative ≤ 1.05× for ppl‑like). | `docs/assurance/04-guard-contracts.md` | `make_report` applies tier thresholds; see `tests/eval/test_assurance_contracts.py::test_ppl_ratio_gate_enforced`. | `validation.primary_metric_acceptable`, `primary_metric.{ratio_vs_baseline,display_ci}`, `auto.tier`. | Baseline/reference pairing intact; CLI tier selection propagated. |
| Spectral family caps achieve the documented **false positive rate** (FPR). | `docs/assurance/05-spectral-fpr-derivation.md` | Property test `tests/eval/test_assurance_contracts.py::test_spectral_fpr_matches_tail_probabilities`. | `spectral.family_caps[*].kappa`, `spectral.families[*].kappa`, `spectral.multiple_testing` | z‑scores approx Gaussian under null; per‑run FPR set via Bonferroni/BH. |
| RMT ε‑rule enforces the declared **acceptance band** on activation edge‑risk growth. | `docs/assurance/06-rmt-epsilon-rule.md` | `tests/eval/test_assurance_contracts.py::test_rmt_epsilon_rule_acceptance_band`. | `rmt.{edge_risk_by_family_base,edge_risk_by_family,epsilon_default,epsilon_by_family,epsilon_violations,stable,status}`, `rmt.families.*.{edge_base,edge_cur,delta}` | ε calibrated on **null** runs and stored in `tiers.yaml`. |
| Variance Equalization (VE) **enables only** when the **predictive** paired ΔlogNLL CI upper bound ≤ −`min_effect_lognll` **and** mean Δ ≤ −`min_effect_lognll` (tier‑specific sidedness for CI width). | `docs/assurance/07-ve-gate-power.md` | report validates predictive A/B provenance & CI; see `tests/eval/test_assurance_contracts.py::test_predictive_gate_respects_min_effect`. | `variance.{enabled,predictive_gate,ab_test,scope,proposed_scales}`, `resolved_policy.variance.{min_effect_lognll,predictive_one_sided}` | Balanced = **one‑sided** improvement; Conservative = **two‑sided** CI with improvement‑only gating (CI entirely above +`min_effect_lognll` is treated as regression). Calibrated on same windows. |
| Model invariants hold before evaluation (no NaNs, correct shapes, weight‑tying, tokenizer alignment). | `docs/assurance/04-guard-contracts.md` | `invarlock.guards.invariants` aborts before eval on violation; `tests/guards/test_invariants_guard.py::test_invariants_guard_detects_non_finite_weights`. | `validation.invariants_pass`, `meta.tokenizer_hash`, `provenance.provider_digest`, `policy_digest` | Invariants checked pre‑eval; violations abort to avoid undefined behavior. |
| Bootstrap sanity holds (paired windows, zero overlap, sufficient replicates). | `docs/assurance/04-guard-contracts.md` | report builder enforces pairing/overlap/replicate counts; see `tests/core/test_runner_more_edges.py` and `tests/eval/test_assurance_contracts.py::test_seed_bundle_contract`. | `dataset.windows.stats.{paired_windows,window_match_fraction,window_overlap_fraction,coverage,bootstrap}` | Abort evaluation when pairing < 1.0, overlap > 0, or replicates below tier minimum (CI/Release profiles). |
| Deterministic evaluation requires **seed bundle**, dataset/tokenizer hashes, and **perfect pairing**. | `docs/assurance/08-determinism-contracts.md` | Seed propagation + pairing checks; `tests/eval/test_assurance_contracts.py::test_seed_bundle_contract`. | `meta.seeds`, `meta.tokenizer_hash`, `provenance.provider_digest`, `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows,coverage}`, `policy_digest` | Deterministic flags set; equal preview/final counts; reuse baseline window IDs. |

| Guard Overhead stays within budget (≤ +1.0% PM). | `docs/assurance/10-guard-overhead-method.md` | report gate `validation.guard_overhead_acceptable`; bare vs guarded measured on same windows/seeds with single toggle and snapshot/restore. | `guard_overhead.{bare_ppl,guarded_ppl,overhead_ratio,overhead_percent,overhead_threshold}`, `validation.guard_overhead_acceptable` | Same schedule and seeds; bare control is guard‑free; snapshot/restore or deterministic reload. |

**Summary**

- Every assurance-critical guard links to a short assurance note and an automated test.
- The report verifier enforces **log‑space math** and **pairing** at runtime.
- Observability fields make the assurance case auditable in reports and proof packs.

> Tier scope: Balanced and Conservative are the supported published assurance tiers. The Aggressive tier is research‑oriented and not covered by this assurance case. The `none` tier is provided only for dev/demo flows (loosest gates) and is **explicitly outside** the assurance case.

> 🔍 **Verify on your machine**
>
> ```bash
> OMP_NUM_THREADS=1 conda run -n invarlock pytest -q
> OMP_NUM_THREADS=1 conda run -n invarlock python scripts/check_docs_links.py
> OMP_NUM_THREADS=1 conda run -n invarlock mkdocs build --strict
> ```
>
> Running the suite above mirrors the CI guardrails: it replays the assurance tests, regenerates tier tables, validates doc links, and ensures the MkDocs build stays clean.
