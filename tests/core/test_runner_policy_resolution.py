from types import SimpleNamespace

from invarlock.core.runner import CoreRunner


def test_resolve_guard_policies_fallback_to_default_tier():
    runner = CoreRunner()
    # Report without auto config in meta; triggers default balanced tier
    report = SimpleNamespace(meta={"config": {}}, edit={})
    policies = runner._resolve_guard_policies(report, auto_config=None)
    # Expect standard guard keys present for balanced tier
    assert isinstance(policies, dict)
    for key in ("spectral", "rmt", "variance"):
        assert key in policies
