from __future__ import annotations

from types import SimpleNamespace

from invarlock.guards.spectral import select_budgeted_violations


def test_select_budgeted_violations_handles_unknown_method_and_default_selection() -> (
    None
):
    guard = SimpleNamespace(
        multiple_testing={"method": "unknown", "alpha": "bad", "m": "bad"},
        module_family_map={"known": "ffn"},
    )
    violations = [
        {"module": "known", "z_score": 3.0},
        {"module": object(), "z_score": 2.0},
        {"family": "preset", "z_score": "bad"},
    ]

    selected, metrics = select_budgeted_violations(guard, violations)

    assert metrics["method"] == "bonferroni"
    assert metrics["default_selected_without_pvalue"] == 1
    assert "other" in metrics["families_tested"]
    assert guard.multiple_testing["m"] == "bad"
    assert metrics["m"] == 2
    assert {item["family"] for item in selected} >= {"ffn", "preset"}


def test_select_budgeted_violations_supports_non_dict_multiple_testing() -> None:
    guard = SimpleNamespace(
        multiple_testing="disabled",
        module_family_map={"m": "attn"},
    )
    violations = [
        {"family": "custom", "z_score": float("nan")},
        {"module": "m", "z_score": 2.5},
    ]

    selected, metrics = select_budgeted_violations(guard, violations)

    assert metrics["method"] == "bh"
    assert metrics["default_selected_without_pvalue"] == 1
    assert any(item.get("module") == "m" for item in selected)
    assert any(
        item.get("family") == "custom" and item["selected"] for item in violations
    )


def test_select_budgeted_violations_falls_back_to_other_when_family_map_is_empty() -> (
    None
):
    guard = SimpleNamespace(
        multiple_testing={"method": "bh", "alpha": 0.05, "m": 1},
        module_family_map={"m": ""},
    )
    violations = [{"module": "m", "z_score": 2.0}]

    selected, metrics = select_budgeted_violations(guard, violations)

    assert selected[0]["family"] == "other"
    assert metrics["families_tested"] == ["other"]
