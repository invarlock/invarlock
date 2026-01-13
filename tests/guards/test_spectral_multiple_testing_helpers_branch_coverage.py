from invarlock.guards import spectral as sp


class _BadFloat:
    def __float__(self):
        raise TypeError("nope")


def test_z_to_two_sided_pvalue_branches() -> None:
    assert sp._z_to_two_sided_pvalue(0) == 1.0
    assert sp._z_to_two_sided_pvalue(float("inf")) == 1.0
    assert sp._z_to_two_sided_pvalue(_BadFloat()) == 1.0
    assert 0.0 < sp._z_to_two_sided_pvalue(2.0) < sp._z_to_two_sided_pvalue(1.0) < 1.0


def test_finite01_branches() -> None:
    assert sp._finite01(0.0) is True
    assert sp._finite01(0.5) is True
    assert sp._finite01(1.0) is True
    assert sp._finite01(-0.1) is False
    assert sp._finite01(1.1) is False
    assert sp._finite01(float("nan")) is False
    assert sp._finite01(_BadFloat()) is False


def test_bh_reject_families_branches() -> None:
    assert sp._bh_reject_families({}, alpha=0.05, m=3) == set()

    family_pvals = {"a": 0.01, "b": 0.04, "c": 0.5}
    assert sp._bh_reject_families(family_pvals, alpha=0.05, m=3) == {"a"}

    # Boundary: p == cutoff should be selected (guards against < vs <= mutants).
    family_pvals_eq = {"a": 0.01, "b": 0.05}
    assert sp._bh_reject_families(family_pvals_eq, alpha=0.05, m=2) == {"a", "b"}

    # Exercise alpha parsing failure + non-int m path.
    family_pvals2 = {"a": 0.01, "b": 0.02, "c": 0.03}
    assert sp._bh_reject_families(family_pvals2, alpha="bad", m="bad") == {
        "a",
        "b",
        "c",
    }

    assert sp._bh_reject_families({"a": 0.01}, alpha=0.0, m=1) == set()
    assert (
        sp._bh_reject_families({"a": 1.2, "b": float("nan")}, alpha=0.05, m=2) == set()
    )


def test_bonferroni_reject_families_branches() -> None:
    assert sp._bonferroni_reject_families({}, alpha=0.05, m=3) == set()

    family_pvals = {"a": 0.01, "b": 0.03}
    assert sp._bonferroni_reject_families(family_pvals, alpha=0.05, m=2) == {"a"}

    family_pvals_eq = {"a": 0.025, "b": 0.03}
    assert sp._bonferroni_reject_families(family_pvals_eq, alpha=0.05, m=2) == {"a"}

    assert sp._bonferroni_reject_families(family_pvals, alpha="bad", m="bad") == {"a"}
    assert sp._bonferroni_reject_families({"a": 0.01}, alpha=-1.0, m=1) == set()
    assert (
        sp._bonferroni_reject_families({"a": 1.2, "b": float("nan")}, alpha=0.05, m=2)
        == set()
    )


def test_select_budgeted_violations_bh_and_fail_closed() -> None:
    guard = sp.SpectralGuard(multiple_testing={"method": "bh", "alpha": 0.05, "m": 2})
    guard.module_family_map = {"m1": "ffn", "m2": "attn", "m3": "attn"}

    budgeted = [
        {"module": "m1", "z_score": 3.0},
        {"module": "m2", "z_score": 0.0},
        {"module": "m3", "z_score": "bad"},
    ]

    selected, metrics = guard._select_budgeted_violations(budgeted)
    assert {v.get("module") for v in selected} == {"m1", "m3"}

    by_module = {v.get("module"): v for v in budgeted}
    assert by_module["m1"]["selected"] is True
    assert by_module["m2"]["selected"] is False
    assert by_module["m3"]["selected"] is True

    assert metrics["method"] == "bh"
    assert metrics["default_selected_without_pvalue"] == 1
