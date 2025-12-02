from invarlock.guards.variance import VarianceGuard


def test_extract_window_ids_various_shapes_and_fallback():
    g = VarianceGuard()
    batches = [
        {"window_id": "a"},
        {"window_ids": ["b", "c"]},
        {"metadata": {"window_id": "d"}},
        {},  # no ids → ignored
    ]
    out = g._extract_window_ids(batches)
    assert out == ["a", "b", "c", "d"]

    # Fallback when no ids present: enumerate
    out2 = g._extract_window_ids([{}, {}])
    assert out2 == ["0", "1"]
