from invarlock.eval.metrics import PerplexityStatus, validate_perplexity


def test_validate_perplexity_invalids_and_statuses():
    # NaN/Inf
    ok, status, _ = validate_perplexity(float("nan"))
    assert ok is False and status == "invalid"
    ok, status, _ = validate_perplexity(float("inf"))
    assert ok is False and status == "invalid"
    # Less than 1.0
    ok, status, _ = validate_perplexity(0.9)
    assert ok is False and status == "invalid"

    # Status buckets
    assert validate_perplexity(10.0)[1] == PerplexityStatus.EXCELLENT
    assert validate_perplexity(75.0)[1] == PerplexityStatus.GOOD
    assert validate_perplexity(150.0)[1] == PerplexityStatus.ACCEPTABLE
    assert validate_perplexity(300.0)[1] == PerplexityStatus.POOR


def test_validate_perplexity_thresholds_and_allow_high():
    # Warning branch (returns True but logs warning) and error branch
    ok, status, msg = validate_perplexity(
        250.0, warn_threshold=200.0, error_threshold=1000.0
    )
    assert (
        ok is True
        and status in {PerplexityStatus.POOR, PerplexityStatus.UNUSABLE}
        and "warning" in msg.lower()
    )

    ok2, status2, msg2 = validate_perplexity(
        5000.0, warn_threshold=200.0, error_threshold=2000.0
    )
    assert ok2 is False and "exceeds error threshold" in msg2

    # allow_high bypasses error failure
    ok3, status3, _ = validate_perplexity(
        5000.0, warn_threshold=200.0, error_threshold=2000.0, allow_high=True
    )
    assert ok3 is True and status3 in {PerplexityStatus.UNUSABLE, PerplexityStatus.POOR}
