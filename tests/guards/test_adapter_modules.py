from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.guards.adapter_modules import (
    adapter_layer_count,
    iter_adapter_layer_modules,
    iter_named_adapter_scoped_modules,
)


class _Wrapped:
    def __init__(self, inner) -> None:
        self.module = inner


class _ConfigRaises:
    @property
    def n_layer(self):
        raise RuntimeError("bad config")


def test_adapter_layer_count_prefers_describe_then_direct_then_config() -> None:
    events: list[str] = []

    class DescribeAdapter:
        def describe(self, _model):
            return {"n_layer": 3}

    class BrokenDescribeAdapter:
        def describe(self, _model):
            raise RuntimeError("describe failed")

    assert adapter_layer_count(object(), DescribeAdapter()) == 3
    assert (
        adapter_layer_count(
            object(),
            BrokenDescribeAdapter(),
            direct_layer_count=lambda: 2,
            log_event=lambda event, **_details: events.append(event),
        )
        == 2
    )
    assert "adapter_describe_error" in events

    wrapped = _Wrapped(SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4)))
    assert adapter_layer_count(wrapped, adapter=None) == 4
    assert adapter_layer_count(SimpleNamespace(config=_ConfigRaises()), None) == 0


def test_iter_adapter_layer_modules_handles_absent_layers_and_layer_errors() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    layer_errors: list[tuple[int, str]] = []

    assert list(iter_adapter_layer_modules(object(), adapter=None)) == []
    assert list(iter_adapter_layer_modules(object(), adapter=object())) == []

    class NoLayersAdapter:
        def get_layer_modules(self, _model, _index):
            return {}

    assert (
        list(
            iter_adapter_layer_modules(
                object(),
                NoLayersAdapter(),
                log_event=lambda event, **details: events.append((event, details)),
            )
        )
        == []
    )
    assert events[-1][0] == "adapter_fallback_no_layers"

    class ErrorAdapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            raise RuntimeError("layer failed")

    assert (
        list(
            iter_adapter_layer_modules(
                object(),
                ErrorAdapter(),
                log_event=lambda event, **details: events.append((event, details)),
                on_layer_error=lambda index, exc: layer_errors.append(
                    (index, str(exc))
                ),
            )
        )
        == []
    )
    assert layer_errors == [(0, "layer failed")]
    assert events[-1][0] == "adapter_layer_modules_error"


def test_iter_adapter_layer_modules_accepts_items_objects_and_filters_keys() -> None:
    class ItemsObject:
        def items(self):
            return [(1, object()), ("ok", "module")]

    class ItemsAdapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            return ItemsObject()

    yielded = list(iter_adapter_layer_modules(object(), ItemsAdapter()))

    assert [(item.layer_index, item.key, item.module) for item in yielded] == [
        (0, "ok", "module")
    ]

    class NotIterableAdapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            return object()

    assert list(iter_adapter_layer_modules(object(), NotIterableAdapter())) == []


def test_iter_named_adapter_scoped_modules_applies_inclusion_predicate() -> None:
    class Adapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            return {"keep": object(), "drop": object()}

    names = list(
        iter_named_adapter_scoped_modules(
            object(),
            Adapter(),
            should_include=lambda name, _module: name.endswith(".keep"),
        )
    )

    assert len(names) == 1
    assert names[0][0] == "adapter.layers.0.keep"


def test_iter_adapter_layer_modules_propagates_broken_items() -> None:
    class BrokenItems:
        def items(self):
            raise RuntimeError("broken items")

    class Adapter:
        def describe(self, _model):
            return {"n_layer": 1}

        def get_layer_modules(self, _model, _index):
            return BrokenItems()

    with pytest.raises(RuntimeError, match="broken items"):
        list(iter_adapter_layer_modules(object(), Adapter()))
