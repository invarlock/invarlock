# Mistral 7B Guard-Value Demo

This artifact packages a real scenario subset run for `mistralai/Mistral-7B-v0.1` on the remote CUDA runner `root@86.38.238.232`.

The flagship comparison is `fp8_e5m2_stress`: the primary metric alone accepts the edit (`validation.primary_metric_acceptable = true`, `ratio_vs_baseline = 1.0248910150012365`), while the scenario contract records the spectral guard intervention (`spectral.caps_applied = 2`, strictness `must_detect`). The same run also includes `scale_explosion` and `rank_collapse` expected-failure reports to show the spectral detection path on stronger injected faults.

Scope note: this is a guard-intervention demo, not a strict report-failure demo. The FP8 stress report keeps `validation.spectral_stable = true` because the spectral cap budget is not exceeded; the evidence-pack verdict gates on the required primary-guard hit instead.
