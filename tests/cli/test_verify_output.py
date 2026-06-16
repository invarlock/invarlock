from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from invarlock.reporting import verify_output as verify_output


class _BadFloat:
    def __float__(self) -> float:
        raise ValueError("boom")


class _BadFloatSubclass(float):
    def __float__(self) -> float:
        raise ValueError("boom")


class _ExplodingDict(dict[str, Any]):
    def get(self, key: str, default: Any = None) -> Any:
        raise RuntimeError(f"cannot read {key}")


def test_coerce_ci_output_and_metric_family_paths() -> None:
    assert verify_output._coerce_finite_float(_BadFloatSubclass(1.0)) is None
    assert verify_output._coerce_ci_output(None) is None
    assert verify_output._coerce_ci_output([1, 2]) == [1.0, 2.0]
    assert verify_output._coerce_ci_output([_BadFloat(), 2]) is None

    assert verify_output._metric_family("accuracy") == "accuracy"
    assert verify_output._metric_family("accuracy") == "accuracy"
    assert verify_output._metric_family("ppl_causal") == "ppl"
    assert verify_output._metric_family("other") == "other"


def test_build_recompute_summary_accuracy_and_outer_exception() -> None:
    accuracy_report = {
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
    }
    primary_metric = {"final": 0.8}
    assert verify_output._build_recompute_summary(
        accuracy_report,
        kind="accuracy",
        primary_metric=primary_metric,
        tolerance=1e-9,
    ) == {"family": "accuracy", "ok": True, "reason": None}

    skipped = verify_output._build_recompute_summary(
        {"metrics": {"classification": {}}},
        kind="accuracy",
        primary_metric=primary_metric,
        tolerance=1e-9,
    )
    assert skipped == {"family": "accuracy", "ok": True, "reason": "skipped"}

    assert (
        verify_output._build_recompute_summary(
            _ExplodingDict(),
            kind="accuracy",
            primary_metric=primary_metric,
            tolerance=1e-9,
        )
        is None
    )


def test_build_recompute_summary_ppl_paths() -> None:
    ppl_report = {
        "evaluation_windows": {
            "final": {"logloss": [math.log(9.0)], "token_counts": [1]}
        }
    }
    assert verify_output._build_recompute_summary(
        ppl_report,
        kind="ppl_causal",
        primary_metric={"final": 9.0},
        tolerance=1e-9,
    ) == {"family": "ppl", "ok": True, "reason": None}

    mismatch = verify_output._build_recompute_summary(
        ppl_report,
        kind="ppl_causal",
        primary_metric={"final": 10.0},
        tolerance=1e-9,
    )
    assert mismatch == {"family": "ppl", "ok": False, "reason": "mismatch"}

    zero_den = verify_output._build_recompute_summary(
        {"evaluation_windows": {"final": {"logloss": [1.0], "token_counts": [0]}}},
        kind="ppl_causal",
        primary_metric={"final": 10.0},
        tolerance=1e-9,
    )
    assert zero_den == {"family": "ppl", "ok": True, "reason": None}

    skipped = verify_output._build_recompute_summary(
        {"evaluation_windows": {"final": {"logloss": [], "token_counts": []}}},
        kind="ppl_causal",
        primary_metric={"final": 10.0},
        tolerance=1e-9,
    )
    assert skipped == {"family": "ppl", "ok": True, "reason": "skipped"}

    bad_float = verify_output._build_recompute_summary(
        {
            "evaluation_windows": {
                "final": {"logloss": [_BadFloat()], "token_counts": [1]}
            }
        },
        kind="ppl_causal",
        primary_metric={"final": 10.0},
        tolerance=1e-9,
    )
    assert bad_float == {"family": "ppl", "ok": True, "reason": "skipped"}

    assert (
        verify_output._build_recompute_summary(
            {},
            kind="not_ppl",
            primary_metric={"final": 10.0},
            tolerance=1e-9,
        )
        is None
    )


def test_build_verify_json_result_item_and_payload() -> None:
    report = {
        "primary_metric": {
            "kind": "accuracy",
            "final": 0.8,
            "ratio_vs_baseline": float("nan"),
            "display_ci": [0.75, 0.85],
        },
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
    }
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        report,
        ok=False,
        reason="policy_fail",
        tolerance=1e-9,
    )
    assert item["id"] == "report.json"
    assert item["kind"] == "accuracy"
    assert item["ok"] is False
    assert item["reason"] == "policy_fail"
    assert item["ratio_vs_baseline"] is None
    assert item["ci"] == [0.75, 0.85]
    assert item["recompute"] == {"family": "accuracy", "ok": True, "reason": None}

    payload = verify_output.build_verify_json_payload(
        [Path("a.json"), Path("b.json")],
        ok=True,
        reason="ok",
        tolerance=1e-9,
        load_report_fn=lambda path: (
            report
            if path.name == "a.json"
            else (_ for _ in ()).throw(ValueError("bad load"))
        ),
    )
    assert payload["summary"] == {"ok": True, "reason": "ok"}
    assert payload["evaluation_report"] == {"count": 2}
    assert "resolution" not in payload
    assert len(payload["results"]) == 2
    assert payload["results"][0]["kind"] == "accuracy"
    assert payload["results"][1]["kind"] == ""
    assert payload["results"][1]["recompute"] is None


def test_build_verify_json_result_item_counts_guard_warnings_fallback() -> None:
    class _BadWarningCount:
        def __int__(self) -> int:
            raise TypeError("bad warning count")

    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        {
            "guard_warnings": {
                "present": False,
                "warning_count": _BadWarningCount(),
                "warnings": [{"guard": "spectral"}, {"guard": "rmt"}],
            }
        },
        ok=True,
        reason="ok",
        tolerance=1e-9,
    )

    assert item["guard_warnings_present"] is True
    assert item["warning_count"] == 2


def test_build_verify_json_result_item_handles_non_dict_report_object() -> None:
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        object(),  # type: ignore[arg-type]
        ok=True,
        reason="ok",
        tolerance=1e-9,
    )

    assert item["kind"] == ""
    assert item["guard_warnings_present"] is False
    assert item["warning_count"] == 0
    assert item["recompute"] is None


def test_build_verify_json_result_item_ignores_malformed_guard_warnings() -> None:
    item = verify_output.build_verify_json_result_item(
        Path("report.json"),
        {"guard_warnings": "bad"},
        ok=True,
        reason="ok",
        tolerance=1e-9,
    )

    assert item["guard_warnings_present"] is False
    assert item["warning_count"] == 0


def test_build_verify_error_payload_and_success_line() -> None:
    payload = verify_output.build_verify_error_payload(
        Path("cert.json"),
        reason="malformed",
        encoded_error={"code": "E601", "message": "bad schema"},
    )
    assert payload["summary"] == {"ok": False, "reason": "malformed"}
    assert payload["results"][0]["id"] == "cert.json"
    assert payload["error"] == {"code": "E601", "message": "bad schema"}

    full_line = verify_output.build_verify_success_line(
        {
            "primary_metric": {
                "kind": "ppl_causal",
                "ratio_vs_baseline": 1.0,
                "display_ci": [0.9, 1.1],
            },
            "ppl": {
                "stats": {"coverage": {"preview": {"used": 12}, "final": {"used": 34}}}
            },
        }
    )
    assert full_line == (
        "VERIFY OK metric=ppl_causal n=12/34 point=1.000000 "
        "ci=[0.900000,1.100000] width=0.200000"
    )

    assert (
        verify_output.build_verify_success_line(
            {"primary_metric": {"display_ci": [_BadFloat(), 1.0]}}
        )
        == "VERIFY OK"
    )
