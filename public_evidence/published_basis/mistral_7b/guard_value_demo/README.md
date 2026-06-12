# Mistral 7B Guard-Value Demo

This artifact packages a real scenario subset run for `mistralai/Mistral-7B-v0.1` on the remote CUDA runner `root@86.38.238.232`.

The FP8 comparison is retained as historical scenario evidence: the primary metric alone accepts `fp8_e5m2_stress` (`validation.primary_metric_acceptable = true`, `ratio_vs_baseline = 1.0248910150012365`), and the original scenario contract recorded `spectral.caps_applied = 2`. A baseline comparison shows those two caps are the same Mistral attention outliers already present in the noop basis, so this FP8 run is not counted as current baseline-relative guard-value proof.

Scope note: this is a real scenario pack and a useful sentinel, not a strict report-failure demo and not a current flagship guard-value proof. The same run also includes `scale_explosion` and `rank_collapse` expected-failure reports that show stronger spectral detection paths, but those faults also fail the primary metric.
