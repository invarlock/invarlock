# Primary Metric — Tiny Smoke Examples

Small, deterministic snippets for each Primary Metric kind. These use only in‑memory “windows” and do not require models, datasets, or network.

## Setup

```python
from __future__ import annotations
import math
from invarlock.eval.primary_metric import get_metric, compute_primary_metric_from_report
```

## ppl_causal (causal LM perplexity)

Point from windows (ppl = exp(weighted mean logloss)):

```python
m = get_metric("ppl_causal")
windows = {"logloss": [math.log(2.0), math.log(4.0)], "token_counts": [1, 1]}
print(m.point_from_windows(windows=windows))  # ~2.828427 (sqrt(8))
```

From a minimal report (with preview/final) + baseline ratio:

```python
report = {
  "evaluation_windows": {
    "preview": {"logloss": [math.log(2.0)], "token_counts": [10]},
    "final":   {"logloss": [math.log(3.0)], "token_counts": [10]},
  }
}
baseline = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 2.0}}}
pm = compute_primary_metric_from_report(
    report, kind="ppl_causal", baseline=baseline
)
print(pm)  # preview=2.0, final=3.0, ratio_vs_baseline=1.5
```

## ppl_mlm (masked LM perplexity)

Prefers `masked_token_counts` when present:

```python
m = get_metric("ppl_mlm")
windows = {
  "logloss": [math.log(5.0)],
  "token_counts": [100],            # ignored in favor of masked counts
  "masked_token_counts": [10],
}
print(m.point_from_windows(windows=windows))  # 5.0
```

## ppl_seq2seq (decoder perplexity)

Same token‑aggregated formula over decoder labels:

```python
m = get_metric("ppl_seq2seq")
windows = {"logloss": [math.log(7.0)], "token_counts": [7]}
print(m.point_from_windows(windows=windows))  # 7.0
```

## accuracy (example‑aggregated, 0..1)

Per‑example flags:

```python
acc = get_metric("accuracy")
print(acc.point_from_windows({"example_correct": [1, 0, 1, 1]}))  # 0.75
```

Aggregate counts (with optional abstain/tie policy):

```python
print(acc.point_from_windows({"correct_total": 9, "total": 10}))  # 0.9
```

From a minimal report with preview/final + baseline delta:

```python
report = {"metrics": {"classification": {
  "preview": {"correct_total": 8,  "total": 10},
  "final":   {"correct_total": 18, "total": 20},
}}}
baseline = {"metrics": {"accuracy": 0.85}}
pm = compute_primary_metric_from_report(report, kind="accuracy", baseline=baseline)
print(pm)  # preview=0.80, final=0.90, ratio_vs_baseline=0.05 (delta)
```

## Alias (multimodal)

One convenience alias delegates to the base metric:

- `vqa_accuracy` → `accuracy`

```python
rep = {"metrics": {"classification": {
  "preview": {"correct_total": 80,  "total": 100},
  "final":   {"correct_total": 190, "total": 200},
}}}
base = {"metrics": {"accuracy": 0.90}}
pm_vqa = compute_primary_metric_from_report(rep, kind="vqa_accuracy", baseline=base)
pm_acc = compute_primary_metric_from_report(rep, kind="accuracy",     baseline=base)
assert pm_vqa["ratio_vs_baseline"] == pm_acc["ratio_vs_baseline"]
```

---

Tip: For CI or smoke demos, these snippets run offline and avoid heavy imports. They exercise the exact paths used by reporting and certificate generation.
