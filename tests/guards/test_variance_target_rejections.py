from types import SimpleNamespace

from invarlock.guards.variance import VarianceGuard


class NoProj:
    pass


class BadProj:
    # Has a weight attribute but not suitable shape
    def __init__(self):
        self.weight = None


class RejectionBlocks:
    def __init__(self):
        self.attn = SimpleNamespace(c_proj=None)  # missing_module
        self.mlp = SimpleNamespace(c_proj=BadProj())  # unsupported_type


class TinyRejectModel:
    def __init__(self, layers=1):
        self.transformer = SimpleNamespace(h=[RejectionBlocks() for _ in range(layers)])

    def named_modules(self):  # minimal impl for iteration
        # Yield self and children-like objects
        yield from []


def test_target_resolution_rejections():
    model = TinyRejectModel(layers=2)
    g = VarianceGuard(
        policy={
            "scope": "both",
            "tap": ["transformer.h.*.attn.c_proj"],
            "min_gain": 0.0,
        }
    )
    g._prepared = True
    g.prepare(model, adapter=None, calib=None)
    res = g._stats.get("target_resolution", {})
    rejected = res.get("rejected", {})
    # Expect missing_module and tap_mismatch/unsupported_type rejections
    # For mlp branch, tap_mismatch since tap only matches attn, plus unsupported_type recorded
    assert "missing_module" in rejected
    # tap_mismatch may cover mlp projections due to tap filter
    assert "tap_mismatch" in rejected or "unsupported_type" in rejected
