# Evaluation Math: Paired Baseline Ratios and Independent Slice Drift

> **Plain language:** The baseline/subject perplexity ratio uses identical-ID
> final windows and a paired bootstrap. Preview and final are deliberately
> disjoint slices: their drift is a difference of two independently estimated
> means, never an index-paired Δ distribution.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Derive the paired baseline/subject ratio and distinguish it from independent preview/final drift. |
| **Audience** | Report verifier maintainers, statistics auditors, and contributors changing paired metric code. |
| **Contract scope** | PPL-like metrics with identical-ID baseline/subject final windows plus disjoint preview/final slices. |
| **Source of truth** | `src/invarlock/core/bootstrap.py`, report pairing logic, and paired-CI contract tests. |

## Claim

For ppl-like metrics on identical-ID paired final windows `i = 1..n` with token
counts `t_i`, the reported **ratio** between baseline A and subject B
satisfies

$$
\text{ratio} = \exp\!\Big(\overline{\Delta \ell}_{\text{w}}\Big),\quad
\Delta \ell_i = \ell^{(B)}_i - \ell^{(A)}_i
$$

where $\ell_i$ is the **per‑token** log‑loss on window $i$, and the **weighted** mean is

$$
\overline{\Delta \ell}_{\text{w}} = \frac{\sum_i t_i \, \Delta \ell_i}{\sum_i t_i}.
$$

The **ratio confidence interval** is obtained by exponentiating the paired
ΔlogNLL CI computed on the **same** windows with BCa bootstrap (paired,
token‑weighted).

This identity does not make preview and final windows paired. Their ID sets are
disjoint. For preview/final drift, InvarLock computes

$$
\Delta_{\mathrm{slice}} = \bar{\ell}_{\mathrm{final},w}
- \bar{\ell}_{\mathrm{preview},w}
$$

and resamples the preview and final arms independently. It does not subtract
array element `i` in preview from element `i` in final, and it does not apply a
paired BCa interval to those disjoint arrays.

## Visual Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│               PAIRED EVALUATION MATH (log-space, token-weighted)        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   WINDOW PAIR i    ┌─────────────────────────────────────────────────┐  │
│   ────────────────▶│  Arm A (baseline)    Arm B (subject)            │  │
│                    │  ────────────────    ────────────────           │  │
│                    │  ℓᵢ⁽ᴬ⁾ = log-loss    ℓᵢ⁽ᴮ⁾ = log-loss           │  │
│                    │  tᵢ   = token count  tᵢ   = token count         │  │
│                    └──────────────────────┬──────────────────────────┘  │
│                                           │                             │
│                                           ▼                             │
│                    ┌─────────────────────────────────────────────────┐  │
│                    │  Δℓᵢ = ℓᵢ⁽ᴮ⁾ − ℓᵢ⁽ᴬ⁾   (per-window Δlog-loss)   │  │
│                    └──────────────────────┬──────────────────────────┘  │
│                                           │                             │
│   FOR ALL WINDOWS i=1..n                  ▼                             │
│                    ┌─────────────────────────────────────────────────┐  │
│                    │      Σᵢ tᵢ · Δℓᵢ                                │  │
│                    │  Δℓ̄ₓ = ─────────────   (token-weighted mean)    │  │
│                    │         Σᵢ tᵢ                                   │  │
│                    └──────────────────────┬──────────────────────────┘  │
│                                           │                             │
│                                           ▼                             │
│            ┌──────────────────────────────┴─────────────┐               │
│            │                                            │               │
│            ▼                                            ▼               │
│   ┌─────────────────┐                       ┌────────────────────────┐  │
│   │     RATIO       │                       │   BCa BOOTSTRAP (CI)   │  │
│   │ ────────────────│                       │ ────────────────────── │  │
│   │ exp(Δℓ̄ₓ)        │                       │ Resample windows      │  │
│   │ = PPL⁽ᴮ⁾/PPL⁽ᴬ⁾ │                       │ uniformly; recompute  │  │
│   │                 │                       │ weighted mean → [L,U]│  │
│   └────────┬────────┘                       └───────────┬────────────┘  │
│            │                                            │               │
│            └────────────────────┬───────────────────────┘               │
│                                 ▼                                       │
│            ┌─────────────────────────────────────────────┐              │
│            │                   report                    │              │
│            │  ratio_vs_baseline = exp(Δℓ̄ₓ)               │              │
│            │  display_ci       = [exp(L), exp(U)]        │              │
│            └─────────────────────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Derivation (sketch)

For ppl-like primary metrics (perplexity):

$$
\text{PPL} = \exp(\bar{\ell}),
\qquad
\bar{\ell} = \frac{\sum_i t_i \ell_i}{\sum_i t_i}
$$

Thus the ratio:

$$
\frac{\text{PM}^{(B)}}{\text{PM}^{(A)}} \quad \text{(ratio in display space for ppl-like metrics)}
= \exp\Big(\bar{\ell}^{(B)} - \bar{\ell}^{(A)}\Big)
= \exp\Big(\overline{\Delta \ell}_{\text{w}}\Big).
$$

BCa treats each paired window as a resampling cluster. Each replicate samples
`n` window indices uniformly with replacement, carries each selected window's
token count with it, and recomputes
$\sum_j t_j\Delta\ell_j/\sum_j t_j$ on that replicate. This preserves the
window-level target statistic; sampling windows with probability proportional to token
count would define a different procedure. Exponentiating interval $[L,U]$
gives $[\exp(L),\exp(U)]$.

### Finite-schedule identity in log space

Let $q_i^{(A)} = \exp(-\ell_i^{(A)})$ and $q_i^{(B)} =
\exp(-\ell_i^{(B)})$ denote each window's geometric-mean assigned token
probability. For the observed windows,

$$
\overline{\Delta \ell}_{\text{w}}
= \frac{\sum_i t_i\,(\ell_i^{(B)}-\ell_i^{(A)})}{\sum_i t_i}
= \log\Bigg[\prod_i
\Bigg(\frac{q_i^{(A)}}{q_i^{(B)}}\Bigg)^{t_i/\sum_j t_j}
\Bigg].
$$

Thus, for the selected finite schedule, the statistic is the log of a
token-weighted geometric likelihood ratio. Generalization to a population
requires an independently justified sampling design and dependence
assumptions; deterministic selection, non-overlap, and a bootstrap seed do not
by themselves establish that claim.

### Jensen inequality note

Let

$$
r_i = \exp(\Delta \ell_i) =
\frac{\mathrm{PPL}^{(B)}_i}{\mathrm{PPL}^{(A)}_i}
$$

Then

$$
\exp\big(\overline{\Delta \ell}_{\text{w}}\big)
$$

is the weighted geometric mean of $r_i$. By AM-GM (equivalently Jensen on
$\log$), the weighted geometric mean is $\le$ the weighted arithmetic mean of
$r_i$. The ratio of mean perplexities is a different quantity and can be larger
or smaller; see the counter-example below.

## Why log‑space vs ratio of means (counter‑example)

The naive ratio of mean perplexities can be biased toward high‑perplexity
windows. A simple two‑window example shows the pitfall:

```python
from math import exp, log

weights = [512, 256]
baseline = [40.0, 220.0]
subject = [38.0, 260.0]  # high-perplexity window regresses strongly

ratio_log = exp(
    sum(w * (log(b) - log(a)) for w, a, b in zip(weights, baseline, subject))
    / sum(weights)
)

ratio_means = (
    sum(w * b for w, b in zip(weights, subject))
    / sum(w * a for w, a in zip(weights, baseline))
)

print(ratio_log, ratio_means)  # 1.0217..., 1.12
```

InvarLock uses the exponential of the token‑weighted mean ΔlogNLL
(`exp(weighted_mean(Δlog))`), which respects pairing and avoids the bias.

## Runtime Contract

- reports must satisfy:
  - `primary_metric.display_ci == exp(primary_metric.ci)` (paired baseline path; ppl-like kinds).
  - `dataset.windows.stats.preview_final_slice_delta_summary` records the
    independent-slice `{mean,ci,basis,paired,ci_method}` contract; `basis` is
    `independent_disjoint_slices` and `paired` is `false`.
  - baseline pairing evidence has `dataset.windows.stats.window_match_fraction
    == 1.0`; configured sliding windows have
    `dataset.windows.stats.window_overlap_fraction == 0.0`; and the raw
    preview/final `evaluation_windows.*.window_ids` sets are disjoint.

- Runs hard-fail in CI/Release profiles when a baseline pairing context exists
  and preview/final counts differ, baseline matching is incomplete, or the
  configured stride creates overlapping token windows. Verification also
  rejects invalid baseline pairing and intersecting preview/final ID sets in
  generated reports.

## Observability

- `primary_metric.{preview,final}` — supports preview→final drift checks for ppl-like kinds.
- `primary_metric.display_ci` and `primary_metric.ci` — paired baseline/subject
  final-window ΔlogNLL interval (check both log and exponentiated views).
- `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`.
- `dataset.windows.stats.preview_final_slice_delta_summary.{mean,ci,basis,paired,ci_method,preview_windows,final_windows}`.
- `dataset.windows.stats.bootstrap.{replicates,seed,preview_final_delta_method,preview_final_delta_seed}`.
- `dataset.windows.stats.coverage.{preview,final}` — confirms both arms honour window/coverage minima.

## Edge cases & safeguards

- If all `t_i` equal, weighting reduces to simple mean: implementation can short‑circuit.
- A collapsed independent-slice interval is recorded as `degenerate=true` but
  does not by itself make the primary metric invalid.
- A constant paired baseline/subject Δ may collapse the paired CI to `[μ, μ]`.
- Label alignment & padding must not contribute to `t_i` (masked tokens excluded).
- The repository tests arithmetic, resampling, fallbacks, and report identities.
  It does not establish nominal BCa coverage for arbitrary serial dependence,
  heterogeneous window weights, adaptive data selection, or cherry-picked runs.

## References

- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft), chapters on language modeling and perplexity. <https://web.stanford.edu/~jurafsky/slp3/>
- Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing.* MIT Press.
- Hugging Face Transformers. “Perplexity of fixed-length models.”
  <https://huggingface.co/docs/transformers/perplexity>
