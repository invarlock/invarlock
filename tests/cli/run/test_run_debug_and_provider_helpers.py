from __future__ import annotations

from invarlock.cli.run_overhead import plan_release_windows
from invarlock.cli.run_pairing import (
    compute_provider_digest,
    resolve_metric_and_provider,
)
from invarlock.core.run_guard_overhead_policy import normalize_guard_overhead_result
from invarlock.reporting.run_metric_utils import format_debug_metric_diffs


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
