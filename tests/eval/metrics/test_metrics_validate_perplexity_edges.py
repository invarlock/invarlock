from invarlock.eval.metrics import validate_perplexity


def test_validate_perplexity_warn_and_error_paths():
    # Very high ppl should trigger error (unless allow_high)
    ok, status, msg = validate_perplexity(
        2500.0,
        vocab_size=None,
        warn_threshold=200.0,
        error_threshold=2000.0,
        allow_high=False,
    )
    assert ok is False and status == "unusable"

    # Mid-range above warn threshold but below error should be POOR and ok unless allow_high=False doesn't affect
    ok2, status2, msg2 = validate_perplexity(
        220.0,
        vocab_size=None,
        warn_threshold=200.0,
        error_threshold=2000.0,
        allow_high=True,
    )
    assert ok2 is True and status2 in {"poor"}
