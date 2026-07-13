# Assurance Case Overview (v1.0)

> **Plain language:** This overview lists the principal public assurance claims,
> their in-repository evidence, and the runtime contracts that enforce them in
> CI/release review.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Index the assurance claims, evidence notes, and runtime enforcement paths for weight-edit regression review. |
| **Audience** | Maintainers, release approvers, auditors, and contributors changing assurance-critical behavior. |
| **Contract scope** | Assurance case v1.0 for one configured baseline, subject, dataset window plan, tier, profile, and runtime policy. |
| **Source of truth** | This claim table, the linked assurance notes, `src/invarlock/core/assurance_contract.py`, and `src/invarlock/reporting/verify_contract.py`. |

> **TL;DR:** InvarLock evaluates whether **weight edits** (quantization,
> pruning, etc.) regress a model beyond configured bounds. The assurance case
> covers paired primary metrics with bootstrap CIs, the canonical five-stage
> guard chain (`invariants` pre, `spectral`, `RMT`, `variance`, `invariants`
> post), and reproducibility metadata and artifact-binding checks. Content
> safety, alignment, deployment security, representative sampling, and
> independent execution attestation remain separate review domains. Tests
> establish implemented contracts; they do not establish population
> calibration or field effectiveness.

> **Assurance boundary:** The assurance case supports a configured weight-edit
> regression review for one baseline, subject, dataset window plan, tier,
> profile, and runtime policy.

This note enumerates the principal **assurance claims** the toolkit makes, the
**evidence** included in-tree, and the **runtime contracts** that enforce each
claim. Each claim must have:

If you need definitions for guard terms (kappa threshold, epsilon band, window
pairing), see the [Glossary](glossary.md).

1) a short argument/derivation (“Evidence”), and
2) a **test or contract** that fails fast when assumptions are violated
   (“Runtime enforcement”).

We also list **observability**—the report fields that let readers verify
the claim.

## Scope, Assumptions, and Adjacent Domains

InvarLock’s assurance case is intentionally narrow. It is focused on
**regression risk from weight edits relative to a chosen baseline under a
specific configuration**. Content safety, alignment, and deployment security
belong to separate review domains.

### In scope

- Structured or quantization‑style **weight edits** applied to an existing model
  (baseline vs edited subject).
- **Paired primary metrics** (ppl/accuracy) on configured evaluation windows,
  with log‑space pairing and BCa bootstrap CIs.
- **GuardChain** behavior: invariants, spectral, RMT, and variance guards that
  detect structural breakage, unstable weights, outlier growth, and harmful
  variance shifts introduced by the edit.
- **Reproducibility metadata and provenance bindings** for the evaluation run: seeds, datasets,
  tokenizers, pairing schedules, and policy configuration reflected in the
  report.
- Execution metadata for the documented runtime profiles. Platform support and
  dependency qualification remain governed by the support matrix and runtime
  image, not by a report-local verifier claim.

### Separate Review Domains

- **Content-harm review** for toxicity, bias, jailbreak behavior, prompt-level
  attacks, and alignment behavior in general use.
- **Model-change review** for unrelated training changes, new datasets, or new
  architectures outside documented support families and tiers.
- **Infrastructure and deployment review** for authz, data governance, access
  control, and runtime hardening outside the InvarLock evaluation runtime.
- **Platform qualification** for environments outside the stated support matrix
  such as native Windows, custom CUDA stacks, or arbitrary dependency versions.

The table below should be read with this scope in mind: each row is a claim
about **paired evaluation and guard behavior for weight edits** under the
documented tiers and environments.

> For the end-to-end report lifecycle, see [One Run Lifecycle](../reference/one-run-lifecycle.md). Guard metric impact evidence is detailed in [Guard Metric Impact Method](10-guard-metric-impact-method.md).

| Claim | Evidence | Runtime enforcement | Observability (report v1.0) | Assumptions & scope |
|------|----------|---------------------|----------------------------------|---------------------|
| Baseline/subject ratios are computed in **log space**, **token‑weighted**, then re‑exponentiated. | `docs/assurance/01-eval-math-derivation.md` | The report pairs identical-ID final windows and enforces `ratio_ci == exp(logloss_delta_ci)` within tolerance; see tests `tests/reporting/policy/test_report_paired_ci_identity.py::test_paired_ci_identity_holds` and `tests/core/test_bootstrap.py::test_compute_paired_delta_and_ratio_ci_consistency`. | `primary_metric.{ratio_vs_baseline,display_ci}`, `dataset.windows.stats.{paired_windows,window_match_fraction,window_overlap_fraction}`, `evaluation_windows.{preview,final}.window_ids`. | Baseline and subject final windows are **paired**. Preview and final ID sets are **disjoint**, not paired. `window_overlap_fraction` separately describes sliding-window token overlap. Token counts are known. BCa is used only on paired baseline/subject ΔlogNLL; if all paired windows have equal length, weighting reduces to a simple mean. |
| Tier-specific **primary metric** gates keep edits within acceptance bands (Balanced base ≤ 1.10×, Conservative base ≤ 1.05× for ppl‑like; effective acceptance adds the published `hysteresis_ratio`). | `docs/assurance/04-guard-contracts.md` | `make_report` applies tier thresholds and hysteresis; see `tests/eval/test_assurance_contracts.py::test_ppl_ratio_gate_enforced` and `tests/reporting/contracts/test_report_policy_edges.py::test_ppl_hysteresis_applied_near_threshold`. | `validation.primary_metric_acceptable`, `validation.hysteresis_applied`, `primary_metric.{ratio_vs_baseline,display_ci}`, `resolved_policy.metrics.pm_ratio`, `auto.tier`. | Baseline/reference pairing intact; CLI tier selection propagated. |
| Spectral family caps and the configured BH/Bonferroni-named candidate selector are applied as documented. | `docs/assurance/05-spectral-fpr-derivation.md` | Property and decision-parity tests exercise candidate selection, family caps, cap budgets, and blocking behavior. | `spectral.family_caps[*].kappa`, `spectral.families[*].kappa`, `spectral.multiple_testing` | In exploratory local-baseline mode, selected violations may remain within `max_caps`; in the current external-baseline strict path, any selected spectral violation blocks. Formal FDR/FWER control and Gaussian null calibration are **not established** by the repository. |
| RMT ε‑rule enforces the declared **acceptance band** on activation edge‑risk growth. | `docs/assurance/06-rmt-epsilon-rule.md` | `tests/eval/test_assurance_contracts.py::test_rmt_epsilon_rule_acceptance_band`. | `rmt.{edge_risk_by_family_base,edge_risk_by_family,epsilon_default,epsilon_by_family,epsilon_violations,stable,status}`, `rmt.families.*.{edge_base,edge_cur,delta}` | The current runtime blocks ε-band violations and strict verification requires stable, violation-free evidence. ε is an operational default, not a demonstrated cross-family/device false-alarm threshold. |
| Variance Equalization (VE) **enables only** when the predictive paired ΔlogNLL CI and mean are negative and also meet −`min_effect_lognll`, using the tier-specific interval sidedness. | `docs/assurance/07-ve-gate-power.md` | Report verifier validates enabled-VE predictive A/B provenance and CI; see `tests/eval/test_assurance_contracts.py::test_predictive_gate_respects_min_effect` and `tests/reporting/contracts/test_reporting_variance_enablement.py::test_validate_variance_enablement_rejects_missing_gate_provenance`. | `variance.{enabled,predictive_gate,ab_test,scope,proposed_scales}`, `resolved_policy.variance.{min_effect_lognll,predictive_one_sided}` | With zero min-effect, both the upper bound and mean must be strictly negative. Current strict verification also requires `predictive_gate.passed=true`, even when VE remains disabled. Shipped values are operational defaults, not demonstrated power calibration. |
| Model invariants are checked before and after the edit path. | `docs/assurance/04-guard-contracts.md` | Separate pre/post invariant stages report either-stage failure. Strict assurance forces invariant `strict_mode=true` and `on_fail=block`; outside strict mode, non-finite/tokenizer failures are fatal while other structural findings follow configured policy. | `validation.invariants_pass`, pre/post invariant stage results, `meta.tokenizer_hash`, `provenance.provider_digest`, `policy_digest` | A passing invariant scan covers implemented checks only; it is not general semantic or security validation. |
| Bootstrap contract sanity distinguishes paired baseline evidence from independent slice drift. | `docs/assurance/03-bca-bootstrap.md` | The report builder and strict verifier check identical-ID baseline/subject pairs, disjoint preview/final slices, configured replicate floors, and the declared method for each interval. | `dataset.windows.stats.{paired_windows,window_match_fraction,window_overlap_fraction,coverage,bootstrap,preview_final_slice_delta_summary}` | Paired baseline/subject windows are resampled as clusters. Preview and final arms are resampled independently. Every replicate recomputes the applicable token-weighted mean. Nominal coverage under arbitrary dependence or selection is not established. |
| Deterministic evaluation requires **seed bundle**, dataset/tokenizer hashes, and **perfect baseline/subject pairing**. | `docs/assurance/08-determinism-contracts.md` | Seed propagation + pairing checks; `tests/eval/test_assurance_contracts.py::test_seed_bundle_contract`. | `meta.seeds`, `meta.tokenizer_hash`, `provenance.provider_digest`, `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows,coverage}`, `policy_digest` | Deterministic flags set; baseline/subject final IDs are identical; preview/final IDs remain disjoint. Equal preview/final counts are a schedule constraint, not pairing evidence. |
| Strict metric reports bind preview/final evidence and the kind-specific baseline comparison to one internally consistent sample basis. | `docs/assurance/15-strict-assurance-checklist.md` | PPL verification recomputes both weighted log-loss arms, displayed points, `ratio_vs_baseline`, and interval identities. Accuracy verification recomputes both count arms and `delta_vs_baseline_pp`; the PPL ratio field is forbidden for accuracy. | `primary_metric.{preview,final,n_preview,n_final,counts_source,estimated}` plus kind-specific `ratio_vs_baseline` or `delta_vs_baseline_pp`, raw windows, classification counts, and coverage fields. | The checks detect internal forks; they do not prove examples, labels, baseline selection, or submitted values were honestly produced. |
| Runtime report/manifest binding is distinguished from caller-supplied runtime-image policy. | `docs/assurance/14-trust-model.md` | Strict verification requires `--expected-runtime-image-digest` and rejects a missing or mismatched digest. Report/off modes may accept `manifest_bound` status. | `runtime_provenance.{binding_verified,expected_digest_matched,trust_status,declared_image_digest}` | The machine field proves only equality. A matching value identifies the expected image only if the verifier caller obtained it independently; it is not remote attestation that the declared image actually executed. |
| Guard primary-metric degradation stays within the configured `0.01` policy limit when evaluated (1% relative increase for PPL; one percentage-point absolute drop for accuracy). | `docs/assurance/10-guard-metric-impact-method.md` | Report gate `validation.guard_metric_impact_acceptable`; current strict assurance requires measured, evaluated evidence, recomputes the kind-specific degradation, and rejects an explicit skip. | `guard_metric_impact.{metric_kind,direction,degradation_basis,bare_value,guarded_value,degradation,degradation_limit}`, `validation.guard_metric_impact_acceptable` | This is a paired bare-vs-guarded **model-quality difference**, not an elapsed-time or compute measurement. The `0.01` value is a policy default, not a general empirical bound. |

**Summary**

- Each claim listed above links to its detailed note and applicable test or
  runtime evidence.
- The report verifier enforces **log‑space math**, count consistency, pairing,
  and artifact bindings; it cannot establish evaluation integrity or representative sampling.
- Observability fields make reports reviewable. They do not turn self-declared
  provenance or uncalibrated thresholds into independent assurance.

> Tier scope: Balanced and Conservative are the supported published assurance tiers. The Aggressive tier is research‑oriented and outside the current assurance case. `none` and unrecognized tier names are invalid; packaged-policy resolution fails explicitly instead of selecting Balanced.

> 🔍 **Verify on your machine**
>
> ```bash
> make verify
> make docs-check-build
> make docs-lint-strict
> ```
>
> Running the suite above exercises the repo-native verification and docs
> guardrails: tests, runtime verifier, lint/format checks, strict docs build,
> link checks, and strict docs lint.
