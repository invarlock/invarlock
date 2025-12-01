from types import SimpleNamespace

from invarlock.guards.variance import VarianceGuard


def test_set_run_context_with_non_dict_context_skips_dataset_and_pairing():
    g = VarianceGuard()
    # context as non-dict triggers else branches for dataset_meta/pairing_baseline
    report = SimpleNamespace(meta={}, context=["not-a-dict"], edit={})
    g.set_run_context(report)
    assert getattr(g, "_dataset_meta", None) is None
    assert g._stats.get("pairing_reference") is None
