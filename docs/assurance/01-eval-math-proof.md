# Paired Evaluation Math (log-space, token-weighted)

> **Plain language:** The reported perplexity ratio is just the exponential of
> the token-weighted mean Δlog-loss, and the confidence interval comes from
> exponentiating the same paired bootstrap—we prove both facts here.

## Claim

For paired evaluation windows `i = 1..n` with token counts `t_i`, the reported
**ratio** between two arms A and B (e.g., preview/final or edited/baseline)
satisfies

$$
\text{ratio} = \exp\!\Big(\overline{\Delta \ell}_{\text{w}}\Big),\quad
\Delta \ell_i = \ell^{(B)}_i - \ell^{(A)}_i,
$$

where $\ell_i$ is the **per‑token** log‑loss on window $i$, and the **weighted** mean is

$$
\overline{\Delta \ell}_{\text{w}} = \frac{\sum_i t_i \, \Delta \ell_i}{\sum_i t_i}.
$$

The **ratio confidence interval** is obtained by exponentiating the paired
ΔlogNLL CI computed on the **same** windows with BCa bootstrap (paired,
token‑weighted).

## Visual Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│               PAIRED EVALUATION MATH (log-space, token-weighted)        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   WINDOW PAIR i    ┌─────────────────────────────────────────────────┐ │
│   ────────────────▶│  Arm A (baseline)    Arm B (subject)            │ │
│                    │  ────────────────    ────────────────           │ │
│                    │  ℓᵢ⁽ᴬ⁾ = log-loss    ℓᵢ⁽ᴮ⁾ = log-loss           │ │
│                    │  tᵢ   = token count  tᵢ   = token count         │ │
│                    └──────────────────────┬──────────────────────────┘ │
│                                           │                            │
│                                           ▼                            │
│                    ┌─────────────────────────────────────────────────┐ │
│                    │  Δℓᵢ = ℓᵢ⁽ᴮ⁾ − ℓᵢ⁽ᴬ⁾   (per-window Δlog-loss)  │ │
│                    └──────────────────────┬──────────────────────────┘ │
│                                           │                            │
│   FOR ALL WINDOWS i=1..n                  ▼                            │
│                    ┌─────────────────────────────────────────────────┐ │
│                    │      Σᵢ tᵢ · Δℓᵢ                                │ │
│                    │  Δℓ̄ₓ = ─────────────   (token-weighted mean)     │ │
│                    │         Σᵢ tᵢ                                   │ │
│                    └──────────────────────┬──────────────────────────┘ │
│                                           │                            │
│                                           ▼                            │
│            ┌──────────────────────────────┴──────────────────────────┐ │
│            │                          │                              │ │
│            ▼                          ▼                              │ │
│   ┌─────────────────┐        ┌────────────────────────┐              │ │
│   │     RATIO       │        │   BCa BOOTSTRAP (CI)   │              │ │
│   │ ────────────────│        │ ──────────────────────  │              │ │
│   │ exp(Δℓ̄ₓ)        │        │ Resample {Δℓᵢ} with   │              │ │
│   │ = PPL⁽ᴮ⁾/PPL⁽ᴬ⁾│        │ weights ∝ tᵢ → [L,U]  │              │ │
│   │                 │        │ CI = [exp(L), exp(U)]  │              │ │
│   └────────┬────────┘        └───────────┬────────────┘              │ │
│            │                             │                           │ │
│            └──────────────┬──────────────┘                           │ │
│                           ▼                                          │ │
│            ┌─────────────────────────────────────────────┐           │ │
│            │               CERTIFICATE                   │           │ │
│            │  ratio_vs_baseline = exp(Δℓ̄ₓ)               │           │ │
│            │  display_ci       = [exp(L), exp(U)]        │           │ │
│            └─────────────────────────────────────────────┘           │ │
│                                                                      │ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Derivation (sketch)

For ppl-like primary metrics (perplexity), $\text{PPL} = \exp(\bar{\ell})$ where $\bar{\ell} = \sum t_i \ell_i / \sum t_i$.
Thus the ratio:

$$
\frac{\text{PM}^{(B)}}{\text{PM}^{(A)}} \quad \text{(ratio in display space for ppl-like metrics)}
= \exp\Big(\bar{\ell}^{(B)} - \bar{\ell}^{(A)}\Big)
= \exp\Big(\overline{\Delta \ell}_{\text{w}}\Big).
$$

BCa applied to the paired vector $\{\Delta \ell_i\}$ (resampled with weights
proportional to $t_i$) yields CI $[L, U]$; exponentiate to obtain
$[\exp(L), \exp(U)]$.

### Unbiasedness in log space (lemma)

Let the token‑weighted mean be $\overline{\Delta \ell}_{\text{w}} = \sum_i t_i\,\Delta \ell_i / \sum_i t_i$. By linearity of expectation,

$$
\mathbb{E}\big[\overline{\Delta \ell}_{\text{w}}\big]
= \frac{\sum_i t_i\, \mathbb{E}[\Delta \ell_i]}{\sum_i t_i}
= \log\Bigg(\prod_i \Big(\tfrac{p_i^{(B)}}{p_i^{(A)}}\Big)^{\,t_i/\sum_j t_j}\Bigg),
$$

so the estimator is unbiased for the log of the (token‑weighted) ratio. Under mild assumptions (ergodicity across windows), the point estimator converges to the population log‑ratio.

### Jensen inequality note

Let $r_i = \exp(\Delta \ell_i) = \mathrm{PPL}^{(B)}_i / \mathrm{PPL}^{(A)}_i$. Then
$\exp\big(\overline{\Delta \ell}_{\text{w}}\big)$ is the weighted geometric mean
of $r_i$. By AM-GM (equivalently Jensen on $\log$), the weighted geometric mean
is $\le$ the weighted arithmetic mean of $r_i$. The ratio of mean perplexities
is a different quantity and can be larger or smaller; see the counter-example
below.

## Why log‑space vs ratio of means (counter‑example)

The naive ratio of mean perplexities can be biased toward high‑perplexity
windows. A simple two‑window example shows the pitfall:

```python
from math import exp, log

weights = [512, 256]
preview = [40.0, 220.0]
final = [38.0, 260.0]  # high-perplexity window regresses strongly

ratio_log = exp(
    sum(w * (log(b) - log(a)) for w, a, b in zip(weights, preview, final))
    / sum(weights)
)

ratio_means = (
    sum(w * b for w, b in zip(weights, final))
    / sum(w * a for w, a in zip(weights, preview))
)

print(ratio_log, ratio_means)  # 1.0217..., 1.12
```

InvarLock uses the exponential of the token‑weighted mean ΔlogNLL
(`exp(weighted_mean(Δlog))`), which respects pairing and avoids the bias.

## Runtime Contract

- Certificates must satisfy:
  - `primary_metric.display_ci == exp(primary_metric.ci)` (paired baseline path; ppl-like kinds).
  - `dataset.windows.stats.paired_delta_summary` records `{mean,std,degenerate}` for the paired Δ distribution.
  - `dataset.windows.stats.window_match_fraction == 1.0` and `dataset.windows.stats.window_overlap_fraction == 0.0`.

- Runs **abort** in CI/Release profiles if preview/final counts differ or pairing < 1.0.

## Observability

- `primary_metric.{preview,final}` — supports preview→final drift checks for ppl-like kinds.
- `primary_metric.display_ci` and `primary_metric.ci` — paired ΔlogNLL interval (check both log and exponentiated views).
- `dataset.windows.stats.{window_match_fraction,window_overlap_fraction,paired_windows}`.
- `dataset.windows.stats.paired_delta_summary.{mean,std,degenerate}` and `dataset.windows.stats.bootstrap.{replicates,seed}`.
- `dataset.windows.stats.coverage.{preview,final}` — confirms both arms honour window/coverage minima.

## Edge cases & safeguards

- If all `t_i` equal, weighting reduces to simple mean: implementation can short‑circuit.
- Degenerate Δ (all equal): mark `degenerate=true` and collapse the CI to `[μ, μ]` with `μ = mean(Δ)`; certificate records the fallback.
- Label alignment & padding must not contribute to `t_i` (masked tokens excluded).

## References

- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft), chapters on language modeling and perplexity. <https://web.stanford.edu/~jurafsky/slp3/>
- Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing.* MIT Press.
