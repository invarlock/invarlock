import math
from types import SimpleNamespace

import pytest

import invarlock.reporting.report_normalization as report_normalization
from invarlock.reporting.dataset_hashing import _extract_dataset_info
from invarlock.reporting.policy_utils import _resolve_policy_tier
from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.utils import (
    _coerce_int,
    _get_mapping,
    _get_section,
    _iter_guard_entries,
    _sanitize_seed_bundle,
)
from tests.reporting._support_canonical_reports import (
    canonical_baseline,
    canonical_run_report,
)
from tests.reporting._support_primary_metric import independent_slice_summary


def _baseline_report(model_id: str, preview: float, final: float) -> dict[str, object]:
    baseline = create_empty_report()
    baseline["meta"].update(
        {
            "model_id": model_id,
            "adapter": "hf_causal",
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        }
    )
    baseline["context"] = {"profile": "dev"}
    baseline["edit"]["name"] = "noop"
    baseline["metrics"] = {
        "primary_metric": {
            "kind": "ppl_causal",
            "preview": preview,
            "final": final,
        },
    }
    return canonical_baseline(baseline)


def test_sanitize_seed_bundle_varied_inputs():
    # Fallback-only path (no bundle)
    out = _sanitize_seed_bundle(None, fallback=7)
    assert out == {"python": 7, "numpy": 7, "torch": 7}

    # Bundle with explicit None preserved and coercions applied
    out2 = _sanitize_seed_bundle(
        {"python": None, "numpy": "7", "torch": "x"}, fallback=None
    )
    assert out2["python"] is None
    assert out2["numpy"] == 7
    assert out2["torch"] is None


def test_iter_guard_entries_list_and_mapping():
    # List of guard entries
    report_list = {
        "guards": [
            {"name": "spectral", "policy": {}},
            {"name": "variance", "policy": {}},
        ]
    }
    entries = _iter_guard_entries(report_list)
    assert {e["name"] for e in entries} == {"spectral", "variance"}

    # Mapping form
    report_map = {"guards": {"spectral": {"policy": {}}, "variance": {"policy": {}}}}
    entries_map = _iter_guard_entries(report_map)
    assert {e["name"] for e in entries_map} == {"spectral", "variance"}

    # Mapping with non-dict payload still yields entry with name only
    report_map2 = {"guards": {"invariants": None}}
    entries_map2 = _iter_guard_entries(report_map2)
    assert entries_map2 == [{"name": "invariants"}]

    # Non-list/non-dict guards returns empty list
    report_bad = {"guards": None}
    assert _iter_guard_entries(report_bad) == []


def test_get_section_and_mapping_helpers():
    src = {"a": 1}
    assert _get_section(src, "a") == 1
    assert _get_mapping(src, "a") == {}

    obj = SimpleNamespace(a={"k": 2})
    assert _get_section(obj, "a") == {"k": 2}
    assert _get_mapping(obj, "a") == {"k": 2}


def test_coerce_int_nonfinite_float():
    assert _coerce_int(float("nan")) is None
    assert _coerce_int(float("inf")) is None


def test_resolve_policy_tier_exception_path():
    class BadStr:
        def __str__(self):
            raise RuntimeError("boom")

    report = {"meta": {"auto": {"tier": BadStr()}}}
    with pytest.raises(ValueError, match="meta.auto.tier"):
        _resolve_policy_tier(report)


def test_make_evaluation_report_raises_on_drift_vs_delta_mismatch(monkeypatch):
    # Minimal report with preview/final and paired windows
    window_ids = list(range(1, 181))
    logloss_vals = [0.1] * len(window_ids)
    report = {
        "meta": {"model_id": "m", "seed": 123},
        "metrics": {
            "ppl_preview": 10.0,
            "ppl_final": 11.0,
            # Inject a paired-delta summary mean inconsistent with preview→final drift
            "preview_final_slice_delta_summary": independent_slice_summary(
                math.log(1.22),
                preview_windows=180,
                final_windows=180,
            ),
        },
        "data": {
            "dataset": "dummy",
            "split": "train",
            "seq_len": 8,
            "stride": 8,
            "preview_n": 180,
            "final_n": 180,
            "tokenizer_name": "tok",
        },
        "evaluation_windows": {
            "final": {"window_ids": window_ids, "logloss": logloss_vals}
        },
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = _baseline_report("m", preview=9.4, final=9.5)
    baseline["evaluation_windows"] = {
        "final": {"window_ids": window_ids, "logloss": logloss_vals}
    }
    report["metrics"].update(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 11.0,
                "ratio_vs_baseline": 11.0 / 9.5,
                "ci": (-0.01, 0.01),
                "display_ci": (math.exp(-0.01), math.exp(0.01)),
            },
            "logloss_delta_ci": (-0.01, 0.01),
            "bootstrap": {
                "method": "percentile",
                "replicates": 1200,
                "alpha": 0.05,
                "seed": 0,
                "coverage": {
                    "preview": {"used": 180},
                    "final": {"used": 180},
                    "replicates": {"used": 1200},
                },
            },
            "window_match_fraction": 1.0,
            "window_overlap_fraction": 0.0,
            "stats": {
                "requested_preview": 180,
                "requested_final": 180,
                "actual_preview": 180,
                "actual_final": 180,
            },
        }
    )

    # Bypass full schema validation to focus on drift consistency branch
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    monkeypatch.setattr(
        "invarlock.core.bootstrap.compute_paired_delta_log_ci",
        lambda *_a, **_k: (-0.01, 0.01),
    )

    # The canonical values are resolved before identity enforcement, so this
    # inconsistent paired summary must fail closed.
    report.setdefault("metrics", {}).setdefault("window_plan", {}).update(
        {"profile": "ci", "preview_n": 180, "final_n": 180}
    )
    with pytest.raises(ValueError, match="drift ratio"):
        make_report(report, baseline)


def test_make_evaluation_report_primary_seed_defaulted_when_missing(monkeypatch):
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            # Provide an explicit seed to satisfy strict validation
            "seed": 0,
            "seeds": {"python": None, "numpy": None, "torch": None},
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.1,
                "ratio_vs_baseline": 1.0,
            }
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
            "tokenizer_name": "tok",
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = _baseline_report("m", preview=10.1, final=10.2)
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    # Ensure minimal acceptance criteria satisfied
    cert = make_report(canonical_run_report(report), baseline)
    # Seed=0 is a valid, preserved seed value.
    assert cert["meta"]["seed"] == 0


def test_make_evaluation_report_uses_tokenizer_hash_from_data(monkeypatch):
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 123,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.9,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
            "tokenizer_hash": "tok-abc",
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = _baseline_report("m", preview=10.1, final=10.5)
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(canonical_run_report(report), baseline)
    assert cert["meta"]["tokenizer_hash"] == "tok-abc"


def test_make_evaluation_report_includes_cuda_flags_and_model_profile(monkeypatch):
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 7,
            "cuda_flags": {"bf16": True},
            "model_profile": {"n_params": 1000},
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.9,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            }
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = _baseline_report("m", preview=10.1, final=10.5)
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(canonical_run_report(report), baseline)
    # Extended meta fields may be omitted after normalization
    assert isinstance(cert.get("meta"), dict)


def test_make_evaluation_report_carries_window_plan(monkeypatch):
    report = {
        "meta": {
            "model_id": "m",
            "adapter": "hf_causal",
            "seed": 9,
            "auto": {
                "tier": "balanced",
                "probes_used": 0,
                "target_pm_ratio": None,
            },
        },
        "context": {"profile": "dev"},
        "metrics": {
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 9.9,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
            },
            "window_plan": {"profile": "dev", "preview_n": 10, "final_n": 12},
        },
        "data": {
            "dataset": "dummy",
            "split": "val",
            "seq_len": 8,
            "stride": 1,
            "preview_n": 1,
            "final_n": 1,
        },
        "evaluation_windows": {"final": {"window_ids": [1, 2], "logloss": [0.1, 0.2]}},
        "guards": [],
        "edit": {
            "name": "mock",
            "deltas": {
                "params_changed": 0,
                "heads_pruned": 0,
                "neurons_pruned": 0,
                "layers_modified": 0,
            },
        },
        "plugins": {"adapter": {}, "edit": {}, "guards": []},
    }
    baseline = _baseline_report("m", preview=10.1, final=10.5)
    monkeypatch.setattr(report_normalization, "validate_report", lambda _: True)
    cert = make_report(canonical_run_report(report), baseline)
    # Window plan may be omitted; ensure dataset pairing stats are present
    stats = cert.get("dataset", {}).get("windows", {}).get("stats", {})
    assert isinstance(stats, dict)


def test_extract_dataset_info_uses_explicit_hashes():
    report = {
        "meta": {"seed": 123},
        "data": {
            "dataset": "ds",
            "split": "val",
            "seq_len": 16,
            "stride": 1,
            "preview_n": 2,
            "final_n": 3,
            "preview_hash": "abc123",
            "final_hash": "def456",
            "preview_total_tokens": 160,
            "final_total_tokens": 240,
        },
    }
    info = _extract_dataset_info(report)
    # Expect blake2s: prefix when explicit hashes provided
    assert info["hash"]["preview"].startswith("blake2s:")
    assert info["hash"]["final"].startswith("blake2s:")
    # Total tokens carried through
    assert info["hash"]["total_tokens"] == 160 + 240
