from invarlock.core.auto_tuning import get_tier_summary, validate_tier_config


def test_get_tier_summary_valid_and_invalid():
    ok = get_tier_summary("balanced", edit_name="quant_rtn")
    assert ok["tier"] == "balanced" and "policies" in ok
    bad = get_tier_summary("unknown")
    assert bad.get("error") and "valid_tiers" in bad


def test_validate_tier_config_variants():
    valid, err = validate_tier_config(
        {"tier": "conservative", "enabled": True, "probes": 0}
    )
    assert valid and err is None
    valid, err = validate_tier_config({})
    assert not valid and "Missing 'tier'" in err
    valid, err = validate_tier_config({"tier": "unknown"})
    assert not valid and "Invalid tier" in err
    valid, err = validate_tier_config({"tier": "balanced", "enabled": "yes"})
    assert not valid and "'enabled' must be a boolean" in err
    valid, err = validate_tier_config({"tier": "balanced", "probes": "many"})
    assert not valid and "'probes' must be an integer" in err
