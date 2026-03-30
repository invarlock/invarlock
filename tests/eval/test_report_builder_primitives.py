# ruff: noqa: F405
from __future__ import annotations

from tests.eval.report_builder_support import *  # noqa: F401,F403,F405


class TestEvaluationReportHelpers:
    """Direct tests for low-level helper utilities."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (True, 1),
            (2, 2),
            (2.0, 2),
            ("5", 5),
            (3.7, None),
            ("not-int", None),
        ],
    )
    def test_coerce_int(self, value, expected):
        assert _coerce_int(value) == expected

    def test_sanitize_seed_bundle_preserves_known_entries(self):
        bundle = {"python": "11", "numpy": None, "torch": 17}
        sanitized = _sanitize_seed_bundle(bundle, fallback=42)
        assert sanitized == {"python": 11, "numpy": None, "torch": 17}

    @pytest.mark.parametrize(
        ("modules", "expected"),
        [
            (["layer.attn.c_proj"], "attn"),
            (["mlp.c_fc"], "ffn"),
            (["embed.wte", "mlp.c_fc"], "embed+ffn"),
            ([], "unknown"),
        ],
    )
    def test_infer_scope_from_modules(self, modules, expected):
        result = _infer_scope_from_modules(modules)
        if not modules:
            assert result == expected
        else:
            # Results are sorted by family name when multiple families detected.
            assert result == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ((1, 2), (1.0, 2.0)),
            ("(3.5, 4.5)", (3.5, 4.5)),
            ("invalid", (math.nan, math.nan)),
            ([1, "bad"], (math.nan, math.nan)),
        ],
    )
    def test_coerce_interval(self, value, expected):
        lower, upper = _coerce_interval(value)
        exp_lower, exp_upper = expected
        if math.isnan(exp_lower):
            assert math.isnan(lower) and math.isnan(upper)
        else:
            assert lower == pytest.approx(exp_lower)
            assert upper == pytest.approx(exp_upper)

    def test_pair_logloss_windows_pairs_by_id(self):
        run_windows = {
            "window_ids": [1, 2, 3],
            "logloss": [0.2, 0.3, 0.4],
        }
        baseline_windows = {
            "window_ids": [3, 1, 2],
            "logloss": [0.41, 0.19, 0.29],
        }
        paired = _pair_logloss_windows(run_windows, baseline_windows)
        assert paired is not None
        run_vals, base_vals = paired
        assert run_vals == [0.2, 0.3, 0.4]
        assert base_vals == [0.19, 0.29, 0.41]

    def test_pair_logloss_windows_requires_matching_lengths(self):
        run_windows = {"window_ids": [1], "logloss": [0.2]}
        baseline_windows = {"window_ids": [2], "logloss": [0.3]}
        assert _pair_logloss_windows(run_windows, baseline_windows) is None

    def test_prepare_guard_overhead_with_reports(self):
        bare = {"metrics": {"primary_metric": {"kind": "ppl_causal", "final": 100.0}}}
        guarded = {
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 101.0}}
        }
        payload = {
            "bare_report": bare,
            "guarded_report": guarded,
            "overhead_threshold": 0.05,
            "source": "regression",
        }
        section, passed = _prepare_guard_overhead_section(payload)
        assert passed is True
        assert section["evaluated"] is True
        assert section["source"] == "regression"
        assert section["overhead_percent"] == pytest.approx(1.0)

    def test_prepare_guard_overhead_ratio_fallback(self):
        raw = {
            "bare_ppl": 100,
            "guarded_ppl": 103,
            "overhead_threshold": 0.02,
        }
        section, passed = _prepare_guard_overhead_section(raw)
        assert passed is False
        assert section["overhead_ratio"] == pytest.approx(1.03)
        assert section["threshold_percent"] == pytest.approx(2.0)

    def test_prepare_guard_overhead_missing_ratio_records_error(self):
        raw = {"bare_ppl": "nan", "guarded_ppl": None}
        section, passed = _prepare_guard_overhead_section(raw)
        # Missing/invalid inputs: mark not evaluated and soft-pass
        assert passed is True
        assert section["diagnostics"]
        assert section["diagnostics"][0]["severity"] == "warning"
        assert section["evaluated"] is False

    def test_prepare_guard_overhead_structured_reports(self):
        bare_report = {
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 98.0}}
        }
        guarded_report = {
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 99.0}}
        }

        class FakeResult:
            def __init__(self):
                self.metrics = {
                    "overhead_ratio": 1.015,
                    "overhead_percent": 1.5,
                    "bare_ppl": 98.0,
                    "guarded_ppl": 99.0,
                }
                self.diagnostics = [
                    {
                        "kind": "guard_overhead_info",
                        "severity": "info",
                        "message": "ok",
                        "details": {},
                    }
                ]
                self.checks = {"ppl": True}
                self.passed = True

        with patch(
            "invarlock.reporting.validate.validate_guard_overhead",
            return_value=FakeResult(),
        ):
            section, passed = _prepare_guard_overhead_section(
                {
                    "bare_report": bare_report,
                    "guarded_report": guarded_report,
                    "overhead_threshold": 0.02,
                    "source": "structured",
                }
            )

        assert passed is True
        assert section["evaluated"] is True
        assert section["overhead_ratio"] == pytest.approx(1.015)
        assert section["source"] == "structured"
        assert section["checks"]["ppl"] is True
        assert section["diagnostics"][0]["message"] == "ok"

    def test_pair_logloss_windows_invalid_inputs(self):
        assert _pair_logloss_windows(None, {}) is None
        run_windows = {"window_ids": ["a"], "logloss": [0.1]}
        baseline_windows = {"window_ids": [1], "logloss": [0.2]}
        assert _pair_logloss_windows(run_windows, baseline_windows) is None

    def test_iter_guard_entries_handles_dict_mapping(self):
        report = {
            "guards": {
                "spectral": {"policy": {"sigma_quantile": 0.95}},
                "variance": {"policy": {"predictive_one_sided": True}},
            }
        }
        entries = _iter_guard_entries(report)
        assert {entry["name"] for entry in entries} == {"spectral", "variance"}

    def test_compute_variance_policy_digest_handles_keys(self):
        variance_policy = {
            "deadband": 0.1,
            "min_abs_adjust": 0.02,
            "max_scale_step": 0.5,
            "min_effect_lognll": 9e-4,
            "predictive_one_sided": True,
            "topk_backstop": 4,
            "max_adjusted_modules": 1,
            "irrelevant": 123,
        }
        expected_digest = _compute_variance_policy_digest(variance_policy)
        assert len(expected_digest) == 16
        assert expected_digest == _compute_variance_policy_digest(variance_policy)

        assert _compute_variance_policy_digest({"unknown": 1}) == ""

    def test_format_family_caps_and_epsilon_map(self):
        caps = {"ffn": {"kappa": 2.5}, "attn": 2.8, "invalid": {"kappa": "bad"}}
        formatted_caps = _format_family_caps(caps)
        assert formatted_caps["ffn"]["kappa"] == pytest.approx(2.5)
        assert formatted_caps["attn"]["kappa"] == pytest.approx(2.8)
        assert "invalid" not in formatted_caps

        epsilon = {"ffn": 0.1, "attn": 0.08, "bad": "x"}
        formatted_eps = _format_epsilon_map(epsilon)
        assert formatted_eps["ffn"] == pytest.approx(0.1)
        assert formatted_eps["attn"] == pytest.approx(0.08)
        assert "bad" not in formatted_eps

    def test_extract_policy_overrides_deduplicates_entries(self):
        report = {
            "meta": {
                "policy_overrides": ["configs/overrides/spectral.yaml"],
                "overrides": "configs/overrides/variance.yaml",
                "auto": {"overrides": ["configs/overrides/rmt.yaml", None]},
            },
            "config": {"overrides": ["configs/overrides/variance.yaml", "local.yaml"]},
        }

        overrides = _extract_policy_overrides(report)
        assert overrides == [
            "configs/overrides/spectral.yaml",
            "configs/overrides/variance.yaml",
            "configs/overrides/rmt.yaml",
            "local.yaml",
        ]

    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            ({"auto": {"tier": "balanced"}}, "balanced"),
            ({"policy_tier": "aggressive"}, "aggressive"),
            ({}, "balanced"),
        ],
    )
    def test_resolve_policy_tier_variants(self, meta, expected):
        report = {"meta": meta}
        assert _resolve_policy_tier(report) == expected

    def test_resolve_policy_tier_from_context(self):
        report = {
            "meta": {},
            "context": {"auto": {"tier": "conservative"}},
        }
        assert _resolve_policy_tier(report) == "conservative"

    def test_normalize_baseline_handles_baseline_v1(self):
        baseline = {
            "schema_version": "baseline-v1",
            "meta": {"model_id": "m", "commit_sha": "sha"},
            "metrics": {"ppl_final": 11.0, "ppl_preview": 10.5},
        }
        normalized = _normalize_baseline(baseline)
        assert normalized["model_id"] == "m"
        assert normalized["ppl_final"] == 11.0
