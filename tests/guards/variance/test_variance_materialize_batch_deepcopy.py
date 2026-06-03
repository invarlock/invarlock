from invarlock.guards.variance import VarianceGuard


class NoDeepcopy:
    def __deepcopy__(self, memo):
        raise RuntimeError("no deepcopy")


def test_materialize_batch_deepcopy_failure_fallback():
    g = VarianceGuard()
    payload = {"foo": NoDeepcopy()}
    out = g._materialize_batch(payload)
    # Should return original object for fallback path, not raise
    assert isinstance(out, dict) and isinstance(out["foo"], NoDeepcopy)
