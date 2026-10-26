"""Measurement invariants for the expanded whole-transaction experiment."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from invarlock_rust_experiment.full_runner import (
    CANDIDATES,
    balanced_order,
    comparison_result,
)


def record(root):
    return {
        "output_directory": root,
        "payload": {
            "ok": True,
            "result": {
                "outputs": [
                    {
                        "status": 0,
                        "payload": {
                            "text": "unchanged evidence text",
                            "report": {
                                "requested_outputs": {"html": root + "/report.html"},
                                "written_outputs": {"html": root + "/report.html"},
                                "pack_manifest_digest": "sha256:abc",
                            },
                        },
                    }
                ],
                "files": {"html": "sha256:def"},
            },
        },
    }


def test_four_candidate_order_balances_every_position():
    orders = balanced_order(4, 4127)
    for candidate in CANDIDATES:
        assert [
            sum(order[position] == candidate for order in orders)
            for position in range(4)
        ] == [1, 1, 1, 1]
    assert orders == balanced_order(4, 4127)


def test_only_declared_report_destinations_are_normalized():
    left, right = record("/reference"), record("/native")
    assert comparison_result(left) == comparison_result(right)
    for change in ("digest", "text", "file"):
        altered = copy.deepcopy(right)
        if change == "digest":
            altered["payload"]["result"]["outputs"][0]["payload"]["report"][
                "pack_manifest_digest"
            ] = "sha256:changed"
        if change == "text":
            altered["payload"]["result"]["outputs"][0]["payload"]["text"] = "changed"
        if change == "file":
            altered["payload"]["result"]["files"]["html"] = "sha256:changed"
        assert comparison_result(left) != comparison_result(altered)
    assert right == record("/native")


def test_invalid_output_never_becomes_evaluable():
    for payload in (
        None,
        {},
        {"result": {}},
        {"result": {"outputs": "invalid"}},
        {"result": {"outputs": [], "files": []}},
    ):
        assert (
            comparison_result({"payload": payload, "output_directory": "/native"})
            is None
        )
    malformed = record("/native")
    malformed["payload"]["result"]["outputs"][0]["payload"]["report"][
        "written_outputs"
    ]["html"] = "/outside/report.html"
    assert comparison_result(malformed) is None
