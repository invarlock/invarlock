from __future__ import annotations

import importlib
import sys
import types


def _import_run_module():
    # tiny transformers stub to avoid heavy deps during import
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _Tok:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def get_vocab(self):
                return {"<pad>": 0, "<eos>": 1}

        class _Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                return _Tok()

        class _GPT2(_Auto):
            pass

        tr.AutoTokenizer = _Auto  # type: ignore[attr-defined]
        tr.GPT2Tokenizer = _GPT2  # type: ignore[attr-defined]
        sys.modules["transformers"] = tr
        sub = types.ModuleType("transformers.tokenization_utils_base")
        sub.PreTrainedTokenizerBase = object  # type: ignore[attr-defined]
        sys.modules["transformers.tokenization_utils_base"] = sub
    return importlib.import_module("invarlock.cli.commands.run")


class _CfgObj:
    def __init__(self, dataset_provider=None, metric=None):
        class D:  # simple namespace-like
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
    run_mod = _import_run_module()
    pm = {"preview": 9.0, "final": 10.0, "ratio_vs_baseline": 2.0}
    metrics = {
        "primary_metric": {"preview": 8.0, "final": 10.0, "ratio_vs_baseline": 2.0}
    }
    base = {"metrics": {"primary_metric": {"final": 5.0}}}
    s = run_mod._format_debug_metric_diffs(pm, metrics, base)
    assert "final: v1-v1" in s
    assert "Δlog(final)" in s or "log" in s  # allow math domain
    assert "ratio_vs_baseline" in s

    # Degenerate: missing pieces yields empty string
    assert run_mod._format_debug_metric_diffs(None, None, None) == ""


def test_normalize_overhead_result_handles_non_finite() -> None:
    run_mod = _import_run_module()
    payload = {}
    out = run_mod._normalize_overhead_result(payload)
    assert out["evaluated"] is False and out["passed"] is True
    # Finite value leaves payload as-is (no evaluated/pass injected)
    ok = {"overhead_ratio": 0.005}
    out2 = run_mod._normalize_overhead_result(ok)
    assert "evaluated" not in out2 and "passed" not in out2


def test_resolve_metric_and_provider_various_paths() -> None:
    run_mod = _import_run_module()
    # Direct provider string and explicit metric dict
    cfg = _CfgObj(
        dataset_provider="c4", metric={"kind": "ppl_mlm", "reps": 3, "ci_level": 0.95}
    )
    mk, pk, opts = run_mod._resolve_metric_and_provider(cfg, _Profile())
    assert (
        mk == "ppl_mlm"
        and pk == "c4"
        and opts.get("reps") == 3.0
        and opts.get("ci_level") == 0.95
    )

    # 'auto' kind falls back to profile default metric
    cfg2 = _CfgObj(dataset_provider=None, metric={"kind": "auto"})
    mk2, pk2, _ = run_mod._resolve_metric_and_provider(
        cfg2, _Profile(default_provider="wt103", default_metric="ppl_seq2seq")
    )
    assert mk2 == "ppl_seq2seq" and pk2 == "wt103"

    # Legacy loss-type mapping and provider fallback
    cfg3 = _CfgObj(dataset_provider=None, metric=None)
    mk3, pk3, _ = run_mod._resolve_metric_and_provider(
        cfg3, _Profile(), resolved_loss_type="mlm"
    )
    assert mk3 == "ppl_mlm" and isinstance(pk3, str) and pk3
    mk4, _, _ = run_mod._resolve_metric_and_provider(
        cfg3, _Profile(), resolved_loss_type="seq2seq"
    )
    assert mk4 == "ppl_seq2seq"


def test_resolve_metric_and_provider_attr_metric_and_bad_values() -> None:
    run_mod = _import_run_module()

    class M:
        def __init__(self):
            self.kind = "ppl_causal"
            self.reps = "2"  # coercible
            self.ci_level = "bad"  # not coercible

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

    mk, pk, opts = run_mod._resolve_metric_and_provider(
        Cfg(), _Profile(default_provider="wikitext2")
    )
    assert (
        mk == "ppl_causal"
        and pk == "wikitext2"
        and opts.get("reps") == 2.0
        and "ci_level" not in opts
    )


def test_compute_provider_digest_none_paths() -> None:
    run_mod = _import_run_module()
    assert run_mod._compute_provider_digest({}) is None
    assert run_mod._compute_provider_digest({"evaluation_windows": {}}) is None


def test_plan_release_windows_console_adjustment_message(capsys) -> None:
    run_mod = _import_run_module()

    class _Console:
        def print(self, *args, **kwargs):
            # simple print passthrough for capture
            print(*args)

    # capacity where target_per_arm (300) exceeds usable raw so adjustment triggers
    capacity = {
        "available_unique": 1000,
        "available_nonoverlap": 1000,
        "total_tokens": 10_000,
        "dedupe_rate": 0.1,
        "candidate_unique": 500,  # reduce effective_unique but sufficient
        "candidate_limit": 500,
    }
    plan = run_mod._plan_release_windows(
        capacity,
        requested_preview=300,
        requested_final=300,
        max_calibration=200,
        console=_Console(),
    )
    out = capsys.readouterr().out
    assert "Release window capacity:" in out and "Adjusted per-arm" in out
    assert plan["coverage_ok"] is True
