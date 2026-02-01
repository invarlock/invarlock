from invarlock.reporting.guards_analysis import _extract_invariants


def test_invariants_mixed_severity_status_fail():
    report = {
        "metrics": {
            "invariants": {
                # Non-dict false → error failure entry
                "bool_check": False,
                # Dict without explicit violations but passed False
                "dict_check": {"passed": False, "message": "oops"},
            }
        },
        "guards": [
            {
                "name": "invariants",
                "metrics": {
                    "checks_performed": 3,
                    "violations_found": 2,
                    "fatal_violations": 1,
                    "warning_violations": 1,
                },
                "violations": [
                    {"check": "w1", "type": "mismatch", "severity": "warning"},
                    {"check": "f1", "type": "fatal", "severity": "error"},
                ],
            }
        ],
    }
    out = _extract_invariants(report)
    assert out["status"] == "fail"
    assert out["summary"]["fatal_violations"] == 1
    assert any(f.get("severity") == "error" for f in out["failures"]) and any(
        f.get("severity") == "warning" for f in out["failures"]
    )
