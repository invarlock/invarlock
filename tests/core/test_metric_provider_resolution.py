from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core import metric_provider_resolution as mpr


class _MetricViaGet:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get(self, key: str) -> object:
        return self._values.get(key)


def test_resolve_metric_and_provider_uses_section_callable_and_metric_get(monkeypatch):
    monkeypatch.setattr(
        mpr, "resolve_provider_kind_and_kwargs", lambda _value: ("", {})
    )

    class _Cfg:
        def __init__(self) -> None:
            self.dataset = SimpleNamespace(provider=None)

        def section(self, name: str):  # noqa: ANN001
            if name == "eval":
                return {
                    "metric": _MetricViaGet(
                        {"kind": "auto", "reps": "4", "ci_level": "0.9"}
                    )
                }
            return {}

    profile = SimpleNamespace(
        default_provider="profile-provider", default_metric="ppl_seq2seq"
    )

    kind, provider, opts = mpr.resolve_metric_and_provider(
        _Cfg(),
        profile,
        resolved_loss_type="classification",
        metric_kind_override="auto",
    )
    assert kind == "ppl_seq2seq"
    assert provider == "profile-provider"
    assert opts == {"reps": 4.0, "ci_level": 0.9}


def test_resolve_metric_and_provider_override_and_loss_fallback(monkeypatch):
    monkeypatch.setattr(
        mpr, "resolve_provider_kind_and_kwargs", lambda _value: ("", {})
    )

    cfg = SimpleNamespace(dataset=SimpleNamespace(provider=None))
    profile = SimpleNamespace(default_provider=None, default_metric=None)

    kind, provider, opts = mpr.resolve_metric_and_provider(
        cfg,
        profile,
        resolved_loss_type="classification",
        metric_kind_override="accuracy",
    )
    assert kind == "accuracy"
    assert provider == "wikitext2"
    assert opts == {}

    kind2, provider2, opts2 = mpr.resolve_metric_and_provider(
        cfg,
        profile,
        resolved_loss_type=None,
    )
    assert kind2 == "ppl_causal"
    assert provider2 == "wikitext2"
    assert opts2 == {}


def test_resolve_metric_and_provider_reraises_unexpected_metric_lookup_errors(
    monkeypatch,
):
    monkeypatch.setattr(
        mpr, "resolve_provider_kind_and_kwargs", lambda _value: ("", {})
    )

    class _MetricBoom:
        def get(self, _key: str) -> object:
            raise RuntimeError("boom")

    class _Cfg:
        dataset = SimpleNamespace(provider=None)

        def section(self, name: str):  # noqa: ANN001
            if name == "eval":
                return {"metric": _MetricBoom()}
            return {}

    profile = SimpleNamespace(default_provider=None, default_metric=None)

    with pytest.raises(RuntimeError, match="boom"):
        mpr.resolve_metric_and_provider(_Cfg(), profile, resolved_loss_type="mlm")
