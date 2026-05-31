from __future__ import annotations

import pytest

from invarlock.cli import run_pairing as pairing_helpers_mod
from invarlock.cli.run_overhead import plan_release_windows
from invarlock.cli.run_pairing import (
    _canonical_dataset_id,
    compute_provider_digest,
    resolve_metric_and_provider,
)
from invarlock.reporting.report_overhead import normalize_guard_overhead_result
from invarlock.reporting.run_report_metrics_contract import format_debug_metric_diffs


class _CfgObj:
    def __init__(self, dataset_provider=None, metric=None):
        class D:
            def __init__(self, provider):
                self.provider = provider

        class E:
            def __init__(self, metric):
                self.metric = metric

        self.dataset = D(dataset_provider)
        self.eval = E(metric)


class _Profile:
    def __init__(self, default_provider=None, default_metric=None):
        self.default_provider = default_provider
        self.default_metric = default_metric


def test_format_debug_metric_diffs_happy_and_degenerate() -> None:
    pm = {"preview": 9.0, "final": 10.0, "ratio_vs_baseline": 2.0}
    metrics = {
        "primary_metric": {"preview": 8.0, "final": 10.0, "ratio_vs_baseline": 2.0}
    }
    base = {"metrics": {"primary_metric": {"final": 5.0}}}
    rendered = format_debug_metric_diffs(pm, metrics, base)
    assert "final: v1-v1" in rendered
    assert "Δlog(final)" in rendered or "log" in rendered
    assert "ratio_vs_baseline" in rendered
    assert format_debug_metric_diffs(None, None, None) == ""


def test_normalize_overhead_result_handles_non_finite() -> None:
    out = normalize_guard_overhead_result({})
    assert out["evaluated"] is False and out["passed"] is True
    ok = {"overhead_ratio": 0.005}
    out2 = normalize_guard_overhead_result(ok)
    assert "evaluated" not in out2 and "passed" not in out2


def test_resolve_metric_and_provider_various_paths() -> None:
    cfg = _CfgObj(
        dataset_provider="c4", metric={"kind": "ppl_mlm", "reps": 3, "ci_level": 0.95}
    )
    mk, pk, opts = resolve_metric_and_provider(cfg, _Profile())
    assert (
        mk == "ppl_mlm"
        and pk == "c4"
        and opts.get("reps") == 3.0
        and opts.get("ci_level") == 0.95
    )

    cfg2 = _CfgObj(dataset_provider=None, metric={"kind": "auto"})
    mk2, pk2, _ = resolve_metric_and_provider(
        cfg2, _Profile(default_provider="wt103", default_metric="ppl_seq2seq")
    )
    assert mk2 == "ppl_seq2seq" and pk2 == "wt103"

    cfg3 = _CfgObj(dataset_provider=None, metric=None)
    mk3, pk3, _ = resolve_metric_and_provider(
        cfg3, _Profile(), resolved_loss_type="mlm"
    )
    assert mk3 == "ppl_mlm" and isinstance(pk3, str) and pk3
    mk4, _, _ = resolve_metric_and_provider(
        cfg3, _Profile(), resolved_loss_type="seq2seq"
    )
    assert mk4 == "ppl_seq2seq"


def test_resolve_metric_and_provider_attr_metric_and_bad_values() -> None:
    class M:
        def __init__(self):
            self.kind = "ppl_causal"
            self.reps = "2"
            self.ci_level = "bad"

    class E:
        def __init__(self):
            self.metric = M()

    class D:
        def __init__(self):
            self.provider = None

    class Cfg:
        def __init__(self):
            self.dataset = D()
            self.eval = E()

    mk, pk, opts = resolve_metric_and_provider(
        Cfg(), _Profile(default_provider="wikitext2")
    )
    assert (
        mk == "ppl_causal"
        and pk == "wikitext2"
        and opts.get("reps") == 2.0
        and "ci_level" not in opts
    )


def test_compute_provider_digest_none_paths() -> None:
    assert compute_provider_digest({}) is None
    assert compute_provider_digest({"evaluation_windows": {}}) is None


def test_canonical_dataset_id_and_provider_digest_fallbacks(monkeypatch) -> None:
    class _ItemsThenAttr:
        def __init__(self) -> None:
            self.provider = "  wt103  "

        def items(self):  # noqa: D401, ANN001
            raise RuntimeError("boom")

    class _AttrOnly:
        def __init__(self) -> None:
            self.dataset = "  c4  "

    class _BadStr:
        def __str__(self) -> str:
            raise RuntimeError("bad str")

    assert _canonical_dataset_id(None) is None
    assert _canonical_dataset_id("   ") is None
    assert _canonical_dataset_id({"kind": "  wikitext2  "}) == "wikitext2"
    assert _canonical_dataset_id(_ItemsThenAttr()) == "wt103"
    assert _canonical_dataset_id(_AttrOnly()) == "c4"
    assert _canonical_dataset_id(_BadStr()) is None

    monkeypatch.setattr(
        pairing_helpers_mod,
        "_compute_mask_positions_digest",
        lambda windows: "masksha",  # noqa: ARG005
    )

    numeric_report = {
        "evaluation_windows": {
            "preview": {"window_ids": [1, 2]},
            "final": {"window_ids": [3]},
        },
        "meta": {"tokenizer_hash": "tok-meta"},
    }
    numeric_digest = compute_provider_digest(numeric_report)
    assert numeric_digest == {
        "ids_sha256": numeric_digest["ids_sha256"],
        "tokenizer_sha256": "tok-meta",
        "masking_sha256": "masksha",
    }

    string_report = {
        "evaluation_windows": {
            "preview": {"window_ids": [1, "x"]},
            "final": {"window_ids": ["y"]},
        },
        "data": {"tokenizer_hash": "tok-data"},
    }
    string_digest = compute_provider_digest(string_report)
    assert string_digest == {
        "ids_sha256": string_digest["ids_sha256"],
        "tokenizer_sha256": "tok-data",
        "masking_sha256": "masksha",
    }


def test_canonical_dataset_id_and_provider_digest_edge_paths() -> None:
    class _AttrNoneThenString:
        provider = None

        def __str__(self) -> str:
            return "  fallback-dataset  "

    class _BadItems:
        def items(self):
            return [("dataset", "wt103")]

    assert _canonical_dataset_id(_AttrNoneThenString()) == "fallback-dataset"
    assert _canonical_dataset_id(_BadItems()) == "wt103"

    digest = compute_provider_digest(
        {
            "evaluation_windows": {
                "preview": {"window_ids": []},
                "final": {"window_ids": [3]},
            },
            "meta": {"tokenizer_hash": ""},
        },
        compute_mask_positions_digest_fn=lambda _windows: "",
    )
    assert digest == {"ids_sha256": digest["ids_sha256"]}


def test_compute_provider_digest_reraises_unexpected_window_id_errors() -> None:
    class _BadWindowId:
        def __int__(self) -> int:
            raise RuntimeError("boom")

    report = {
        "evaluation_windows": {
            "preview": {"window_ids": [_BadWindowId()]},
            "final": {"window_ids": []},
        }
    }

    with pytest.raises(RuntimeError, match="boom"):
        compute_provider_digest(report)


def test_compute_provider_digest_supports_multimodal_example_ids() -> None:
    digest = compute_provider_digest(
        {
            "evaluation_windows": {
                "preview": {
                    "example_ids": ["ex-2", "ex-1"],
                    "processor_sha256": "proc-123",
                },
                "final": {
                    "records": [{"id": "ex-3"}],
                },
            }
        },
        compute_mask_positions_digest_fn=lambda _windows: "",
    )

    assert digest == {
        "ids_sha256": digest["ids_sha256"],
        "processor_sha256": "proc-123",
    }


def test_compute_provider_digest_skips_non_dict_sections_and_uses_meta_fallbacks() -> (
    None
):
    digest = compute_provider_digest(
        {
            "evaluation_windows": {
                "preview": "ignore-me",
                "final": {"window_ids": [7]},
            },
            "meta": {
                "tokenizer_hash": "",
                "processor_sha256": "proc-meta",
            },
            "data": {
                "tokenizer_hash": "tok-data",
                "processor_sha256": "proc-data",
            },
        },
        compute_mask_positions_digest_fn=lambda _windows: "masksha",
    )

    assert digest == {
        "ids_sha256": digest["ids_sha256"],
        "tokenizer_sha256": "tok-data",
        "processor_sha256": "proc-meta",
        "masking_sha256": "masksha",
    }


def test_compute_provider_digest_uses_data_processor_sha_fallback() -> None:
    digest = compute_provider_digest(
        {
            "evaluation_windows": {
                "preview": {"window_ids": [1]},
                "final": {"window_ids": [2]},
            },
            "data": {
                "tokenizer_hash": "",
                "processor_sha256": "proc-data",
            },
        },
        compute_mask_positions_digest_fn=lambda _windows: "",
    )

    assert digest == {
        "ids_sha256": digest["ids_sha256"],
        "processor_sha256": "proc-data",
    }


def test_plan_release_windows_console_adjustment_message(capsys) -> None:
    class _Console:
        def print(self, *args, **kwargs):
            print(*args)

    plan = plan_release_windows(
        {
            "available_unique": 10000,
            "available_nonoverlap": 10000,
            "total_tokens": 10000,
            "dedupe_rate": 0.0,
        },
        requested_preview=5000,
        requested_final=5000,
        max_calibration=100,
        console=_Console(),
        event_fn=lambda console, tag, msg, **kwargs: console.print(f"{tag}: {msg}"),
    )
    captured = capsys.readouterr()
    assert plan["actual_preview"] == plan["actual_final"]
    assert "METRIC:" in captured.out
