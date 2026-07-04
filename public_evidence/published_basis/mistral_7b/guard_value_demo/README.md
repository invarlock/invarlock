# Mistral 7B Guard-Value Demo

This artifact packages a real, no-calibration guard-value probe sweep for `mistralai/Mistral-7B-v0.1` on a self-hosted CUDA runner.

The active all-guard comparison is recorded in `artifact_package/reports/guard_value_all_guard_probe_sweep.json`. It publishes three PM-pass, baseline-relative guard-value cases:

- `spectral_moderate_scale_mlp_l31_up_s112`: a targeted 1.12x scale of `model.layers.31.mlp.up_proj`; PM accepts (`ratio_vs_baseline = 1.0076338080085065`) while the baseline-relative spectral detector records one new FFN cap.
- `rmt_norm_noise_l31_ffn_up_b030`: a targeted RMT anisotropy edit of `model.layers.31.mlp.up_proj`; PM accepts (`ratio_vs_baseline = 1.0027430699936888`) while `rmt_probe.json` records an FFN edge-risk epsilon violation relative to baseline.
- `ve_mlp_scale_skew_l31_down_s090`: a targeted 0.90x scale of `model.layers.31.mlp.down_proj`; PM accepts (`ratio_vs_baseline = 1.0002479838633067`) while `ve_probe.json` records a positive VE A/B signal. The package also ships `artifact_package/reports/analysis/ve_baseline_probe.json`, where the same baseline self-probe has no signal.

The package also includes `spectral_moderate_scale_attn_l31_o_s105` as the selected attention negative control: a 1.05x scale on `model.layers.31.self_attn.o_proj` passes PM (`ratio_vs_baseline = 0.9994204418834356`), adds no new baseline-relative spectral cap, and keeps the edited target below the documented stock attention cap (`z = 2.7987064430328767`, stock cap `3.018`). The earlier `spectral_moderate_scale_attn_l31_o_s112` record is retained as margin-policy context only: that 1.12x attention edit passes under the packaged policy but is boundary-adjacent (`z = 3.033867912045445`), above the stock-style cap and below the packaged report kappa (`3.068`). The compact spectral sweep summary records adjacent scale points showing the attention target starts triggering at 1.18 and the FFN target remains PM-accepted through 1.20.

Provenance note: manual task logs from probe selection were not retained in this published artifact. The shipped evidence consists of the full evaluation reports, runtime manifests, sidecar probe outputs, compact summaries, manifest hashes, and `artifact_package/logs/run_pack.log`.

Scope note: this is an evidence-pack guard-value demonstration, not a claim that every stock runtime cap or sidecar signal is an automatic release failure. Invariants are structural checks rather than near-threshold statistical guards; they are required to pass in these PM-pass guard-value cases, but this package does not claim a PM-pass invariant-fail flagship case.
