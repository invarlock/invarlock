from __future__ import annotations

import copy

import pytest

from invarlock.policy_pack import build_policy_pack
from invarlock.reporting import report_make_assembly
from invarlock.reporting.report_make_assembly import (
    _resolve_policy_edit_and_telemetry_context,
)
from invarlock.reporting.report_types import create_empty_report
from invarlock.reporting.runtime_policy_receipt import (
    RUNTIME_POLICY_RECEIPT_FORMAT,
    build_runtime_policy_receipt,
    runtime_policy_from_report,
)


def _runtime_policy() -> dict:
    return {
        "spectral": {"correction_enabled": True, "max_caps": 2},
        "rmt": {"epsilon_default": 0.01},
        "variance": {"max_adjusted_modules": 1},
        "metrics": {"accuracy": {"min_examples": 400}},
    }


def _guards() -> list[dict]:
    return [
        {
            "name": "spectral",
            "policy": {
                "correction_enabled": True,
                "max_caps": 2,
                "measurement_contract": {"estimator": "power_iter"},
            },
        },
        {"name": "rmt", "policy": {"epsilon_default": 0.01}},
        {"name": "variance", "policy": {"max_adjusted_modules": 1}},
    ]


def test_runtime_policy_receipt_binds_exact_applied_guard_policies() -> None:
    source = _runtime_policy()
    resolved, receipt = build_runtime_policy_receipt(
        source,
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )

    assert resolved is not source
    assert resolved["spectral"] == _guards()[0]["policy"]
    assert resolved["metrics"] == source["metrics"]
    assert resolved["guard_authority"] == {
        "spectral": "enforce",
        "rmt": "enforce",
        "variance": "enforce",
    }
    assert receipt["format_version"] == RUNTIME_POLICY_RECEIPT_FORMAT
    assert receipt["source"] == "runtime"
    assert receipt["guard_policies"] == ["rmt", "spectral", "variance"]

    report = {
        "guards": _guards(),
        "resolved_policy": resolved,
        "policy_resolution": receipt,
    }
    replayed, errors = runtime_policy_from_report(report)
    assert errors == []
    assert replayed == resolved


def test_runtime_policy_receipt_rejects_tamper_and_duplicate_disagreement() -> None:
    resolved, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="release",
        edit_name="quant_rtn",
    )
    report = {
        "guards": _guards(),
        "resolved_policy": copy.deepcopy(resolved),
        "policy_resolution": copy.deepcopy(receipt),
    }
    report["resolved_policy"]["spectral"]["max_caps"] = 99

    _replayed, errors = runtime_policy_from_report(report)

    assert any("digest does not match" in error for error in errors)
    assert any("disagrees with applied" in error for error in errors)

    duplicate = [*_guards(), {"name": "rmt", "policy": {"epsilon_default": 1.0}}]
    with pytest.raises(ValueError, match="inconsistent applied policies"):
        build_runtime_policy_receipt(
            _runtime_policy(),
            duplicate,
            tier="balanced",
            profile="ci",
            edit_name="quant_rtn",
        )


def test_evaluation_policy_context_prefers_runtime_receipt() -> None:
    resolved, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )
    report = create_empty_report()
    report["edit"]["name"] = "quant_rtn"
    report["guards"] = _guards()  # type: ignore[typeddict-item]
    report["metrics"]["primary_metric"] = {"kind": "ppl_causal", "final": 1.0}
    report["resolved_policy"] = resolved
    report["policy_resolution"] = receipt
    context = _resolve_policy_edit_and_telemetry_context(
        report,
        report,
        {},
        {"tier": "balanced"},
        {},
        {},
        {},
        {},
        "",
        [],
        non_fatal_exceptions=(AttributeError, KeyError, TypeError, ValueError),
    )

    assert context["resolved_policy"] == resolved
    assert context["policy_provenance"]["source"] == "runtime"
    policy_pack = build_policy_pack(
        tier="balanced",
        resolved_policy=context["resolved_policy"],
    )
    assert policy_pack["resolved_policy"] == resolved


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"runtime_policies": {}}, "resolved policy is empty"),
        (
            {
                "runtime_policies": {
                    **_runtime_policy(),
                    "guard_authority": {"spectral": "observe"},
                }
            },
            "guard_authority",
        ),
        ({"tier": "unknown"}, "tier is unsupported"),
        ({"profile": "unknown"}, "profile is unsupported"),
        ({"edit_name": "  "}, "edit_name is required"),
    ],
)
def test_runtime_policy_receipt_rejects_incomplete_runtime_context(
    overrides: dict, message: str
) -> None:
    arguments = {
        "runtime_policies": _runtime_policy(),
        "guard_entries": _guards(),
        "tier": "balanced",
        "profile": "ci",
        "edit_name": "quant_rtn",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_runtime_policy_receipt(**arguments)


def test_runtime_policy_receipt_rejects_missing_and_empty_guard_policies() -> None:
    for policy in (None, {}):
        guards = [{"name": "spectral", "policy": policy}]
        with pytest.raises(ValueError, match="did not retain its applied policy"):
            build_runtime_policy_receipt(
                _runtime_policy(),
                guards,
                tier="balanced",
                profile="ci",
                edit_name="quant_rtn",
            )


def test_runtime_policy_receipt_reports_every_malformed_binding() -> None:
    resolved, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )
    malformed = {
        "guards": "not-a-list",
        "resolved_policy": resolved,
        "policy_resolution": {
            **receipt,
            "extra": True,
            "format_version": "unknown",
            "source": "report",
            "profile": "unknown",
            "tier": "unknown",
            "edit_name": " ",
            "guard_policies": ["unknown"],
            "resolved_policy_sha256": "sha256:invalid",
        },
    }

    replayed, errors = runtime_policy_from_report(malformed)

    assert replayed == resolved
    assert errors == [
        "policy_resolution fields do not match the current schema",
        "policy_resolution has an unknown format_version",
        "policy_resolution source must be runtime",
        "policy_resolution profile is unsupported",
        "policy_resolution tier is unsupported",
        "policy_resolution edit_name must be a non-empty string",
        "policy_resolution guard_policies is malformed",
        "runtime policy receipt digest does not match resolved_policy",
        "runtime policy receipt requires canonical guard entries",
    ]


def test_runtime_policy_receipt_handles_absent_and_non_object_receipts() -> None:
    assert runtime_policy_from_report({}) == (None, [])
    assert runtime_policy_from_report({"policy_resolution": "bad"}) == (
        None,
        ["policy_resolution must be an object"],
    )
    _, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )
    assert runtime_policy_from_report({"policy_resolution": receipt}) == (
        None,
        ["runtime policy receipt requires resolved_policy"],
    )


def test_runtime_policy_receipt_rejects_guard_inventory_and_policy_omission() -> None:
    resolved, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )
    report = {
        "guards": [*_guards(), {"name": "diagnostic", "policy": {}}],
        "resolved_policy": resolved,
        "policy_resolution": {**receipt, "guard_policies": []},
    }
    _, errors = runtime_policy_from_report(report)
    assert "runtime policy receipt guard inventory does not match report" in errors

    report["policy_resolution"] = receipt
    report["guards"] = [{"name": "spectral", "policy": {"new": 1}}]
    _, errors = runtime_policy_from_report(report)
    assert any("disagrees with applied 'spectral' policy" in error for error in errors)


def _resolve_assembly_context(report: dict, *, auto_tier: str = "balanced") -> dict:
    return _resolve_policy_edit_and_telemetry_context(
        report,
        report,
        {},
        {"tier": auto_tier},
        {},
        {},
        {},
        {},
        "",
        [],
        non_fatal_exceptions=(AttributeError, KeyError, TypeError, ValueError),
    )


def _valid_assembly_report() -> dict:
    resolved, receipt = build_runtime_policy_receipt(
        _runtime_policy(),
        _guards(),
        tier="balanced",
        profile="ci",
        edit_name="quant_rtn",
    )
    return {
        "context": {"profile": "ci"},
        "edit": {"name": "quant_rtn"},
        "guards": _guards(),
        "resolved_policy": resolved,
        "policy_resolution": receipt,
    }


def test_report_assembly_rejects_missing_and_invalid_runtime_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report_make_assembly,
        "runtime_policy_from_report",
        lambda _report: (None, []),
    )
    with pytest.raises(ValueError, match="missing its runtime policy receipt"):
        _resolve_assembly_context({})

    monkeypatch.setattr(
        report_make_assembly,
        "runtime_policy_from_report",
        lambda _report: ({"spectral": {}}, ["tampered"]),
    )
    with pytest.raises(ValueError, match="receipt is invalid: tampered"):
        _resolve_assembly_context({})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("profile", "profile does not match"),
        ("tier", "tier does not match"),
        ("edit", "edit does not match"),
    ],
)
def test_report_assembly_rejects_runtime_context_disagreement(
    mutation: str,
    message: str,
) -> None:
    report = _valid_assembly_report()
    auto_tier = "balanced"
    if mutation == "profile":
        report["context"]["profile"] = "release"
    elif mutation == "tier":
        auto_tier = "aggressive"
    else:
        report["edit"]["name"] = "noop"

    with pytest.raises(ValueError, match=message):
        _resolve_assembly_context(report, auto_tier=auto_tier)


def test_report_assembly_accepts_receipt_without_plugin_metadata() -> None:
    context = _resolve_assembly_context(_valid_assembly_report())
    assert context["plugin_provenance"] == {}
    assert context["edit_name"] == "quant_rtn"
