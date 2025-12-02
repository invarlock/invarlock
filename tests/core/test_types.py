from invarlock.core.types import GuardOutcome, PolicyConfig, get_worst_action


def test_policy_config_action_resolution():
    # Default on_violation used when requested is 'none'
    cfg = PolicyConfig(
        on_violation="warn", guard_overrides=None, enable_auto_rollback=False
    )
    assert cfg.get_action_for_guard("spectral", "none") == "warn"

    # Requested action other than none takes precedence
    assert cfg.get_action_for_guard("spectral", "rollback") == "rollback"

    # Guard-specific override takes highest precedence
    cfg.guard_overrides = {"spectral": "abort"}
    assert cfg.get_action_for_guard("spectral", "warn") == "abort"


def test_get_worst_action_and_guard_outcome_defaults():
    # Worst action across a set
    actions = ["none", "warn", "rollback", "abort"]
    assert get_worst_action(actions) == "abort"

    # Empty list falls back to none
    assert get_worst_action([]) == "none"

    # Unknown actions retain their label but lowest priority
    assert get_worst_action(["foo"]) == "foo"

    # GuardOutcome defaults populated
    o = GuardOutcome("g", True)
    assert isinstance(o.violations, list) and o.violations == []
    assert isinstance(o.metrics, dict) and o.metrics == {}
