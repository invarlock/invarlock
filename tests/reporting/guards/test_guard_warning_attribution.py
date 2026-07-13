from __future__ import annotations

from invarlock.reporting.guard_warnings import build_guard_warnings


def _spectral_report(*, module: str = "layers.0.mlp.up_proj", z: float = 4.2) -> dict:
    return {
        "caps_applied": 1,
        "max_caps": 5,
        "deadband": 0.1,
        "family_caps": {"ffn": {"kappa": 3.0}},
        "top_violations": [
            {
                "module": module,
                "family": "ffn",
                "z_score": z,
                "kappa": 3.0,
                "severity": "warn",
            }
        ],
        "top_z_scores": {"ffn": [{"module": module, "z": z}]},
    }


def test_spectral_same_baseline_cap_does_not_warn() -> None:
    baseline = {"spectral": _spectral_report()}
    subject = {"spectral": _spectral_report(z=4.24)}

    guard_warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"spectral_stable": True},
    )

    assert guard_warnings == {"present": False, "warning_count": 0, "warnings": []}


def test_spectral_same_raw_baseline_guard_cap_does_not_warn() -> None:
    baseline = {
        "guards": [
            {
                "name": "spectral",
                "metrics": {"caps_applied": 1, "max_caps": 5},
                "violations": [
                    {
                        "type": "family_z_cap",
                        "module": "layers.0.mlp.up_proj",
                        "family": "ffn",
                        "z_score": 4.2,
                        "kappa": 3.0,
                    }
                ],
            }
        ]
    }
    subject = {"spectral": _spectral_report(z=4.24)}

    guard_warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"spectral_stable": True},
    )

    assert guard_warnings == {"present": False, "warning_count": 0, "warnings": []}


def test_spectral_same_raw_baseline_guard_cap_survives_normalized_section() -> None:
    baseline = {
        "spectral": {"caps_applied": 1, "max_caps": 5},
        "guards": [
            {
                "name": "spectral",
                "metrics": {"caps_applied": 1, "max_caps": 5},
                "violations": [
                    {
                        "type": "family_z_cap",
                        "module": "model.layers.0.mlp.gate",
                        "family": "router",
                        "z_score": 3.8,
                        "kappa": 5.0,
                    }
                ],
            }
        ],
    }
    subject = {
        "spectral": _spectral_report(
            module="model.layers.0.mlp.gate",
            z=3.82,
        )
    }
    subject["spectral"]["top_violations"][0]["family"] = "router"
    subject["spectral"]["top_violations"][0]["kappa"] = 5.0
    subject["spectral"]["top_z_scores"] = {
        "router": [{"module": "model.layers.0.mlp.gate", "z": 3.82}]
    }

    guard_warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"spectral_stable": True},
    )

    assert guard_warnings == {"present": False, "warning_count": 0, "warnings": []}


def test_spectral_new_capped_module_warns_without_policy_failure() -> None:
    baseline = {"spectral": _spectral_report(module="layers.0.mlp.up_proj")}
    subject = {
        "spectral": {
            **_spectral_report(module="layers.0.mlp.up_proj"),
            "caps_applied": 2,
            "top_violations": [
                {
                    "module": "layers.0.mlp.up_proj",
                    "family": "ffn",
                    "z_score": 4.2,
                    "kappa": 3.0,
                },
                {
                    "module": "layers.31.mlp.up_proj",
                    "family": "ffn",
                    "z_score": 9.7,
                    "kappa": 3.0,
                },
            ],
            "top_z_scores": {
                "ffn": [
                    {"module": "layers.0.mlp.up_proj", "z": 4.2},
                    {"module": "layers.31.mlp.up_proj", "z": 9.7},
                ]
            },
        }
    }

    guard_warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"spectral_stable": True},
    )

    assert guard_warnings["present"] is True
    assert guard_warnings["warning_count"] == 1
    warning = guard_warnings["warnings"][0]
    assert warning["guard"] == "spectral"
    assert warning["kind"] == "new_capped_module"
    assert warning["module"] == "layers.31.mlp.up_proj"
    assert warning["policy_gate"] == "pass"


def test_invariant_warning_count_accepts_raw_and_assembled_shapes_once() -> None:
    raw = {
        "guards": [
            {
                "name": "invariants",
                "metrics": {"warning_violations": 1},
            }
        ]
    }
    assembled = {
        "invariants": {
            "summary": {"warning_violations": 1},
            "warning_violations": 1,
        }
    }

    assert build_guard_warnings(
        subject=assembled,
        baseline=raw,
        validation={"invariants_pass": True},
    ) == {"present": False, "warning_count": 0, "warnings": []}


def test_invariant_warning_delta_reads_raw_guard_metrics() -> None:
    baseline = {
        "guards": [{"name": "invariants", "metrics": {"warning_violations": 1}}]
    }
    subject = {"guards": [{"name": "invariants", "metrics": {"warning_violations": 2}}]}

    warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"invariants_pass": True},
    )

    assert warnings["warning_count"] == 1
    assert warnings["warnings"][0]["baseline"] == {"warning_violations": 1}
    assert warnings["warnings"][0]["subject"] == {"warning_violations": 2}


def test_invariant_warning_delta_prefers_staged_post_without_double_count() -> None:
    baseline = {
        "guards": [
            {
                "name": "invariants",
                "stage": "pre",
                "metrics": {"warning_violations": 1},
            },
            {
                "name": "invariants_post",
                "stage": "post",
                "metrics": {"warning_violations": 1},
            },
        ]
    }
    subject = {
        "guards": [
            {
                "name": "invariants",
                "stage": "pre",
                "metrics": {"warning_violations": 1},
            },
            {
                "name": "invariants_post",
                "stage": "post",
                "metrics": {"warning_violations": 3},
            },
        ]
    }

    warnings = build_guard_warnings(
        subject=subject,
        baseline=baseline,
        validation={"invariants_pass": True},
    )

    assert warnings["warning_count"] == 1
    assert warnings["warnings"][0]["baseline"] == {"warning_violations": 1}
    assert warnings["warnings"][0]["subject"] == {"warning_violations": 3}
