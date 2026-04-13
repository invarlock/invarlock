from __future__ import annotations

import pytest

from invarlock.core import metric_kind_contract as metric_kind_mod


def test_load_metric_kind_catalog_raises_when_contract_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_kind_mod.load_metric_kind_catalog.cache_clear()
    monkeypatch.setattr(
        metric_kind_mod,
        "load_json_contract",
        lambda _filename: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(
        metric_kind_mod.MetricKindContractError,
        match="Failed to load metric kind contract",
    ):
        metric_kind_mod.load_metric_kind_catalog()

    metric_kind_mod.load_metric_kind_catalog.cache_clear()


def test_load_metric_kind_catalog_rejects_non_list_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_kind_mod.load_metric_kind_catalog.cache_clear()
    monkeypatch.setattr(
        metric_kind_mod,
        "load_json_contract",
        lambda _filename: {"accuracy": True},
    )

    with pytest.raises(
        metric_kind_mod.MetricKindContractError,
        match="non-empty JSON array of strings",
    ):
        metric_kind_mod.load_metric_kind_catalog()

    metric_kind_mod.load_metric_kind_catalog.cache_clear()


def test_load_metric_kind_catalog_rejects_empty_concrete_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_kind_mod.load_metric_kind_catalog.cache_clear()
    monkeypatch.setattr(
        metric_kind_mod,
        "load_json_contract",
        lambda _filename: ["", "   ", None],
    )

    with pytest.raises(
        metric_kind_mod.MetricKindContractError,
        match="at least one concrete metric kind",
    ):
        metric_kind_mod.load_metric_kind_catalog()

    metric_kind_mod.load_metric_kind_catalog.cache_clear()


def test_metric_kind_helpers_fail_closed_on_normalization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args, **_kwargs):
        raise ValueError("bad kind")

    monkeypatch.setattr(metric_kind_mod, "normalize_metric_kind", _boom)

    assert metric_kind_mod.is_known_metric_kind("accuracy") is False
    assert metric_kind_mod.is_ppl_metric_kind("ppl_causal") is False
