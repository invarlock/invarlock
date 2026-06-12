# Mistral 7B Guard-Value Demo

This artifact packages a real, no-calibration guard-value probe sweep for `mistralai/Mistral-7B-v0.1` on the remote CUDA runner `root@86.38.238.232`.

The active flagship comparison is `spectral_moderate_scale_mlp_l31_up_s112`: a targeted 1.12x scale of `model.layers.31.mlp.up_proj`, selected because the published noop basis showed it was the closest non-baseline FFN module to its spectral cap. PM-only accepts the run (`validation.primary_metric_acceptable = true`, `ratio_vs_baseline = 1.0076338080085065`), while the evidence-pack PM+guards comparison records one new baseline-relative FFN cap.

The package also includes `spectral_moderate_scale_attn_l31_o_s112` as a negative control: the same 1.12x scale on the closest non-baseline attention module passes PM but does not add a new cap. The compact sweep summary records adjacent scale points showing the attention target starts triggering at 1.18 and the FFN target remains PM-accepted through 1.20.

Scope note: this is an evidence-pack guard-value demonstration, not a claim that every stock runtime spectral cap is an automatic release failure. The older FP8, scale-explosion, and rank-collapse reports remain as historical/detection-path context; the current baseline-relative proof is the lower-dose FFN probe.
