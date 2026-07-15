from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path

import pytest

from invarlock.core.assurance_spectral_replay import (
    _validate_external_baseline,
    _validated_policy,
    replay_spectral_guard,
)
from tests.core._support_spectral_replay import (
    _bounded_report,
    _over_budget_report,
)


def _replay(payload: dict) -> list[str]:
    return replay_spectral_guard(payload, payload["guards"][0], "guards[0]")


def test_replay_accepts_independently_recomputed_bounded_cap() -> None:
    assert _replay(_bounded_report()) == []


def test_replay_observe_waives_only_complete_spectral_cap_budget_outcome() -> None:
    payload = _over_budget_report()
    enforced = _replay(payload)
    assert enforced == ["replayed spectral evidence exceeds the selected-cap budget."]
    assert (
        replay_spectral_guard(
            payload,
            payload["guards"][0],
            "guards[0]",
            enforce_outcome=False,
        )
        == []
    )


@pytest.mark.parametrize("mutation", ["missing-evidence", "fatal-finding"])
def test_replay_observe_rejects_incomplete_or_fatal_spectral_evidence(
    mutation: str,
) -> None:
    payload = _over_budget_report()
    if mutation == "missing-evidence":
        payload["guards"][0].pop("final_z_scores")
    else:
        payload["guards"][0]["policy"]["max_spectral_norm"] = 1.1

    errors = replay_spectral_guard(
        payload,
        payload["guards"][0],
        "guards[0]",
        enforce_outcome=False,
    )

    assert errors
    if mutation == "fatal-finding":
        assert "fatal" in "\n".join(errors)


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("huge-final", "final_z_scores"),
        ("z-score", "final_z_scores"),
        ("family", "module inventories"),
        ("selection", "families_selected"),
        ("budget", "caps_applied"),
        ("absolute-max", "max_spectral_norm"),
        ("over-budget", "max_caps"),
        ("summary", "spectral.summary.modules_checked"),
        ("summary-caps", "spectral.summary.caps_applied"),
    ],
)
def test_replay_rejects_tampered_raw_or_mirrored_evidence(
    mutation: str, fragment: str
) -> None:
    payload = _bounded_report()
    raw = payload["guards"][0]
    if mutation == "huge-final":
        raw["final_metrics"]["layer.0"] = 1e12
    elif mutation == "z-score":
        raw["final_z_scores"]["layer.0"] = 1.5
    elif mutation == "family":
        raw["module_family_map"].pop("layer.1")
    elif mutation == "selection":
        raw["metrics"]["multiple_testing_selection"]["families_selected"] = []
    elif mutation == "budget":
        raw["metrics"]["caps_applied"] = 0
    elif mutation == "absolute-max":
        raw["policy"]["max_spectral_norm"] = 1.1
    elif mutation == "over-budget":
        raw["policy"]["max_caps"] = 0
    elif mutation == "summary":
        payload["spectral"]["summary"]["modules_checked"] = 1
    elif mutation == "summary-caps":
        payload["spectral"]["summary"].pop("caps_applied")

    assert fragment in "\n".join(_replay(payload))


def test_replay_recomputes_enabled_degeneracy_thresholds() -> None:
    payload = _bounded_report()
    raw = payload["guards"][0]
    raw["policy"]["degeneracy"]["enabled"] = True
    raw["baseline_metrics"]["baseline_degeneracy"] = {
        "layer.0": {"stable_rank": 2.0, "norm_collapse": 1.0},
        "layer.1": {"stable_rank": 2.0, "norm_collapse": 1.0},
    }
    raw["final_degeneracy"] = {
        "layer.0": {"stable_rank": 0.4, "norm_collapse": 1.0},
        "layer.1": {"stable_rank": 2.0, "norm_collapse": 1.0},
    }

    errors = _replay(payload)

    assert "degeneracy_stable_rank_drop" in "\n".join(errors)


def test_replay_requires_external_baseline_binding_when_report_requests_it() -> None:
    payload = _bounded_report()
    payload["context"] = {"baseline_guard_evidence_required": True}

    errors = _replay(payload)

    assert "externally required spectral baseline" in "\n".join(errors)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("max_caps", True, "max_caps must be a non-negative integer"),
        ("max_spectral_norm", -1.0, "max_spectral_norm must be null"),
        ("family_caps", {}, "family_caps must be a non-empty object"),
        ("family_caps", {"ffn": {"kappa": -1.0}}, "lacks a finite non-negative"),
        (
            "multiple_testing",
            {"method": "holm", "alpha": 0.05, "m": 1},
            "method must be",
        ),
        ("multiple_testing", {"method": "bh", "alpha": 0.0, "m": 1}, "alpha must be"),
        ("multiple_testing", {"method": "bh", "alpha": 0.05, "m": 0}, "m must be"),
        ("degeneracy", {"enabled": "yes"}, "enabled must be a boolean"),
        (
            "degeneracy",
            {
                "enabled": False,
                "stable_rank": {"warn_ratio": -1.0, "fatal_ratio": 0.1},
                "norm_collapse": {"warn_ratio": 0.2, "fatal_ratio": 0.1},
            },
            "thresholds must be finite and non-negative",
        ),
        ("correction_enabled", 1, "correction_enabled must be a boolean"),
        ("correction_cap_ratio", 0.0, "greater than zero"),
    ],
)
def test_replay_policy_rejects_untyped_or_open_configuration(
    field: str, value: object, fragment: str
) -> None:
    policy = copy.deepcopy(_bounded_report()["guards"][0]["policy"])
    policy[field] = value
    errors: list[str] = []

    assert _validated_policy(errors, policy, {"layer.0": "ffn"}, "guard") is None
    assert fragment in "\n".join(errors)


@pytest.mark.parametrize(
    ("field", "fragment"),
    [
        ("metrics", ".metrics must be a non-empty object"),
        ("policy", ".policy must be a non-empty object"),
        ("baseline_metrics", ".baseline_metrics must be a non-empty object"),
    ],
)
def test_replay_requires_each_raw_evidence_section(field: str, fragment: str) -> None:
    payload = _bounded_report()
    payload["guards"][0][field] = {}

    assert fragment in "\n".join(_replay(payload))


def test_external_baseline_fields_are_typed_ready_and_source_bound() -> None:
    errors: list[str] = []
    _validate_external_baseline(
        errors,
        report={"context": {"baseline_guard_evidence_required": True}},
        metrics={
            "external_baseline_required": "yes",
            "external_baseline_ready": "no",
            "baseline_source": "embedded",
        },
        source="guard",
    )
    text = "\n".join(errors)
    assert "external_baseline_required must be a boolean" in text
    assert "external_baseline_ready must be a boolean" in text
    assert "externally required spectral baseline" in text

    errors = []
    _validate_external_baseline(
        errors,
        report={},
        metrics={
            "external_baseline_required": True,
            "external_baseline_ready": False,
            "baseline_source": "embedded",
        },
        source="guard",
    )
    assert "required external spectral baseline is not ready" in "\n".join(errors)
    assert "baseline_source must be external_run" in "\n".join(errors)


def test_replay_requires_public_summary_and_retained_family_statistics() -> None:
    payload = _bounded_report()
    payload.pop("spectral")
    assert "spectral summary is required" in "\n".join(_replay(payload))

    payload = _bounded_report()
    payload["guards"][0]["baseline_metrics"].pop("family_stats")
    assert "family_stats must be an object" in "\n".join(_replay(payload))


def test_spectral_replay_modules_keep_verifier_boundary_and_acyclic_direction() -> None:
    module_names = (
        "invarlock.core.assurance_spectral_replay",
        "invarlock.core.assurance_spectral_replay_common",
        "invarlock.core.assurance_spectral_replay_inventory",
        "invarlock.core.assurance_spectral_replay_decision",
        "invarlock.core.assurance_spectral_replay_correction",
    )
    public = importlib.import_module(module_names[0])
    assert public.__all__ == ["replay_spectral_guard"]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        source_path = Path(module.__file__ or "")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.startswith("invarlock.guards") for name in imported)
        if module_name != module_names[0]:
            assert "assurance_spectral_replay" not in imported
