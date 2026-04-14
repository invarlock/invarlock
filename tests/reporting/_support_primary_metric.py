def pm(cert: dict) -> dict:
    """Return primary_metric with safe defaults for tests.

    Ensures display_ci is present and 2-length for comparisons.
    """
    assert "primary_metric" in cert, "missing primary_metric"
    m = dict(cert["primary_metric"])  # shallow copy
    if "display_ci" not in m and isinstance(m.get("final"), int | float):
        m["display_ci"] = [m["final"], m["final"]]
    return m
