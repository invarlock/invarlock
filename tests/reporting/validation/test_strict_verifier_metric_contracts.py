from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from invarlock.reporting import verify_bootstrap as bootstrap_mod
from invarlock.reporting import verify_contract as contract_mod
from invarlock.reporting import verify_strict_accuracy as accuracy_mod
from invarlock.reporting import verify_strict_ppl as ppl_mod
from invarlock.runtime_provenance import RuntimeProvenanceResult
from tests.cli._support_verify_runtime_provenance import (
    _matching_strict_ppl_baseline,
    _strict_accuracy_cert,
    _strict_provenance_gate_cert,
)
from tests.reporting.validation._support_strict_verifier_branch_contracts import (
    _accuracy_errors,
    _bootstrap_errors,
)


def test_bootstrap_replay_rejects_kind_resolution_schedule_work_and_compute_forks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _strict_provenance_gate_cert()
    original = bootstrap_mod.normalize_metric_kind
    monkeypatch.setattr(
        bootstrap_mod,
        "normalize_metric_kind",
        lambda _value: (_ for _ in ()).throw(RuntimeError("resolver refused")),
    )
    assert bootstrap_mod._is_ppl_report(report) is False
    monkeypatch.setattr(bootstrap_mod, "normalize_metric_kind", original)

    baseline = _matching_strict_ppl_baseline(report)
    baseline["evaluation_windows"]["final"]["window_ids"][0] = 999
    errors = _bootstrap_errors(report, baseline)
    assert any(
        "final window IDs in the exact subject order" in error for error in errors
    )

    report = _strict_provenance_gate_cert()
    baseline = _matching_strict_ppl_baseline(report)
    baseline["metrics"]["primary_metric"]["final"] = 99.0
    report["baseline_ref"]["primary_metric"]["final"] = 88.0
    errors = _bootstrap_errors(report, baseline)
    assert any("baseline_ref/raw-window mismatch" in error for error in errors)

    monkeypatch.setattr(bootstrap_mod, "MAX_STRICT_BOOTSTRAP_WORK_ITEMS", 1)
    errors = _bootstrap_errors(
        _strict_provenance_gate_cert(), _matching_strict_ppl_baseline()
    )
    assert any("exceeds the verifier work limit" in error for error in errors)

    monkeypatch.setattr(bootstrap_mod, "MAX_STRICT_BOOTSTRAP_WORK_ITEMS", 10_000_000)
    monkeypatch.setattr(
        bootstrap_mod,
        "compute_paired_delta_log_ci",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forged weights")),
    )
    errors = _bootstrap_errors(
        _strict_provenance_gate_cert(), _matching_strict_ppl_baseline()
    )
    assert any("bootstrap replay failed: forged weights" in error for error in errors)


def test_accuracy_primitives_reject_missing_malformed_and_mismatched_counts() -> None:
    assert accuracy_mod._correct_count_from_values([]) is None
    assert accuracy_mod._correct_count_from_records([]) is None
    errors: list[str] = []
    accuracy_mod._append_count_pair_mismatch(
        errors,
        source="acceptance.raw",
        observed=None,
        expected=(1, 2),
    )
    assert errors == []

    for block, expected in (
        (None, "as an object"),
        ({"correct_total": "1", "total": 2}, "with total > 0"),
        ({"correct_total": 3, "total": 2}, "exceeds total"),
    ):
        errors = []
        result = accuracy_mod._classification_arm_pair(
            errors,
            classification={"final": block},
            arm="final",
        )
        if block is None or block["correct_total"] == "1":
            assert result is None
        assert any(expected in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda p: p["primary_metric"].pop("n_preview"), "primary_metric.n_preview"),
        (
            lambda p: p["metrics"]["classification"].pop("n_total"),
            "classification.n_correct/n_total",
        ),
        (
            lambda p: p["primary_metric"].update(counts_source="estimated"),
            "primary_metric counts_source=measured",
        ),
        (
            lambda p: p["metrics"]["classification"].update(counts_source="estimated"),
            "metrics.classification counts_source=measured",
        ),
        (
            lambda p: p["primary_metric"].update(final=None),
            "primary_metric.final must be finite",
        ),
        (
            lambda p: p["primary_metric"].update(preview=None),
            "primary_metric.preview must be finite",
        ),
        (
            lambda p: p["baseline_ref"].update(
                primary_metric={"kind": "accuracy", "final": 2.0}
            ),
            "finite final in [0,1]",
        ),
        (
            lambda p: p["primary_metric"].pop("delta_vs_baseline_pp"),
            "delta_vs_baseline_pp",
        ),
        (
            lambda p: p["evaluation_windows"].update(preview=[]),
            "evaluation_windows.preview must be an object",
        ),
        (
            lambda p: p["evaluation_windows"]["final"].pop("records"),
            "evaluation_windows.final.records as a non-empty list",
        ),
        (
            lambda p: p["evaluation_windows"]["final"]["records"][0].update(
                id="different-example"
            ),
            "ID must match example_ids",
        ),
        (
            lambda p: p["evaluation_windows"]["final"]["records"][0].update(correct=1),
            ".correct must be a boolean",
        ),
        (
            lambda p: p["provenance"]["provider_digest"].update(
                ids_sha256="different-schedule"
            ),
            "example IDs disagree with provenance.provider_digest.ids_sha256",
        ),
        (
            lambda p: p["dataset"].update(windows=[]),
            "dataset.windows must be an object",
        ),
        (
            lambda p: p["dataset"]["windows"].pop("preview"),
            "requires dataset.windows.preview",
        ),
        (
            lambda p: p["dataset"]["windows"].update(preview="200"),
            "dataset.windows.preview must be a non-negative integer",
        ),
        (
            lambda p: p["dataset"]["windows"].update(stats=[]),
            "dataset.windows.stats must be an object",
        ),
        (
            lambda p: p["dataset"]["windows"]["stats"].pop("actual_preview"),
            "requires dataset.windows.stats.actual_preview",
        ),
        (
            lambda p: p["dataset"]["windows"]["stats"].update(actual_preview="200"),
            "actual_preview must be a non-negative integer",
        ),
        (
            lambda p: p["dataset"]["windows"]["stats"].update(coverage=[]),
            "stats.coverage must be an object",
        ),
        (
            lambda p: p["dataset"]["windows"]["stats"]["coverage"].pop("preview"),
            "coverage.preview as an object",
        ),
    ],
)
def test_accuracy_recompute_rejects_each_forged_contract_surface(
    mutation, expected: str
) -> None:
    payload = _strict_accuracy_cert()
    mutation(payload)
    _usable, errors = _accuracy_errors(payload)
    assert any(expected in error for error in errors), errors


def test_accuracy_recompute_rejects_preview_count_fork_and_missing_preview_pair() -> (
    None
):
    payload = _strict_accuracy_cert()
    payload["primary_metric"]["n_preview"] += 1
    _usable, errors = _accuracy_errors(payload)
    assert any("primary_metric.n_preview" in error for error in errors)

    payload = _strict_accuracy_cert()
    payload["metrics"]["classification"].pop("preview")
    _usable, errors = _accuracy_errors(payload)
    assert any("classification.preview as an object" in error for error in errors)
    assert not any(
        "evaluation_windows.preview.example_correct" in error for error in errors
    )


def test_ppl_arm_recompute_rejects_malformed_numeric_and_schedule_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    assert ppl_mod._strict_nonnegative_int(True) is None
    assert (
        ppl_mod._append_ppl_arm_recompute_errors(
            errors,
            arm="final",
            section={"logloss": [-1.0], "token_counts": [1], "window_ids": []},
            primary_metric={},
            tolerance=1e-9,
            require_analysis_point=True,
            require_window_ids=True,
        )
        is None
    )
    assert any("logloss[0] must be non-negative" in error for error in errors)

    errors = []
    mean = ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1], "window_ids": "bad"},
        primary_metric={"final": math.e},
        tolerance=1e-9,
        require_analysis_point=True,
    )
    assert mean == 1.0
    assert any("window_ids must be a list" in error for error in errors)
    assert any("analysis_point_final" in error for error in errors)

    errors = []
    ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1], "window_ids": []},
        primary_metric={"analysis_point_final": 1.0, "final": math.e},
        tolerance=1e-9,
        require_analysis_point=True,
        require_window_ids=True,
    )
    assert any("window_ids as a non-empty list" in error for error in errors)

    monkeypatch.setattr("builtins.sum", lambda _values: 0)
    errors = []
    ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1]},
        primary_metric={},
        tolerance=1e-9,
        require_analysis_point=False,
    )
    assert any("no positive token weight" in error for error in errors)


def test_ppl_arm_recompute_handles_a_window_id_surface_that_changes_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    real_isinstance = isinstance
    checks = 0

    def _changing_isinstance(value: object, classinfo: object) -> bool:
        nonlocal checks
        if value is sentinel and classinfo is list:
            checks += 1
            return checks == 1
        return real_isinstance(value, classinfo)

    monkeypatch.setattr(ppl_mod, "isinstance", _changing_isinstance, raising=False)
    errors: list[str] = []
    mean = ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1], "window_ids": sentinel},
        primary_metric={"final": math.e},
        tolerance=1e-9,
        require_analysis_point=False,
    )
    assert mean == 1.0
    assert checks == 2


def test_ppl_schedule_ignores_non_object_windows_and_non_list_logloss() -> None:
    errors: list[str] = []
    ppl_mod._append_strict_ppl_schedule_errors(
        errors, cert_obj={"evaluation_windows": []}
    )
    assert errors == []

    payload = {
        "evaluation_windows": {"preview": {"window_ids": [1]}},
        "dataset": {"windows": {"stats": {"coverage": {}}}},
    }
    errors = []
    ppl_mod._append_strict_ppl_schedule_errors(errors, cert_obj=payload)
    assert errors == []


def test_ppl_arm_recompute_rejects_nonfinite_and_overflowed_recomputations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ppl_mod.math, "fsum", lambda _values: math.inf)
    errors: list[str] = []
    ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1]},
        primary_metric={},
        tolerance=1e-9,
        require_analysis_point=False,
    )
    assert any("mean log-loss is non-finite" in error for error in errors)

    monkeypatch.setattr(ppl_mod.math, "fsum", lambda _values: 1.0)
    monkeypatch.setattr(
        ppl_mod.math, "exp", lambda _value: (_ for _ in ()).throw(OverflowError())
    )
    errors = []
    ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1]},
        primary_metric={},
        tolerance=1e-9,
        require_analysis_point=False,
    )
    assert any("recomputed perplexity overflows" in error for error in errors)

    monkeypatch.setattr(ppl_mod.math, "exp", lambda _value: math.inf)
    errors = []
    ppl_mod._append_ppl_arm_recompute_errors(
        errors,
        arm="final",
        section={"logloss": [1.0], "token_counts": [1]},
        primary_metric={},
        tolerance=1e-9,
        require_analysis_point=False,
    )
    assert any("outside the finite positive range" in error for error in errors)


def test_ppl_schedule_rejects_missing_counts_stats_coverage_and_nested_mirrors() -> (
    None
):
    errors: list[str] = []
    ppl_mod._append_declared_count_mismatch(
        errors,
        container={},
        key="used",
        source="acceptance.used",
        expected=1,
    )
    ppl_mod._append_declared_count_mismatch(
        errors,
        container={"used": "1"},
        key="used",
        source="acceptance.used",
        expected=1,
    )
    assert errors == [
        "Strict PPL evidence requires acceptance.used.",
        "acceptance.used must be a non-negative JSON integer.",
    ]

    for payload, expected in (
        ({"evaluation_windows": {}}, "dataset.windows as an object"),
        (
            {"evaluation_windows": {}, "dataset": {"windows": {}}},
            "dataset.windows.stats as an object",
        ),
    ):
        errors = []
        ppl_mod._append_strict_ppl_schedule_errors(errors, cert_obj=payload)
        assert any(expected in error for error in errors)

    payload = _strict_provenance_gate_cert()
    stats = payload["dataset"]["windows"]["stats"]
    stats["coverage"] = None
    stats["bootstrap"]["coverage"] = {"preview": {}, "final": {}}
    errors = []
    ppl_mod._append_strict_ppl_schedule_errors(errors, cert_obj=payload)
    assert any("stats.coverage as an object" in error for error in errors)
    assert any("bootstrap.coverage.preview.used" in error for error in errors)


def test_ppl_slice_summary_rejects_malformed_metadata_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _strict_provenance_gate_cert()
    stats = payload["dataset"]["windows"]["stats"]
    stats["paired_delta_summary"] = {}
    summary = stats["preview_final_slice_delta_summary"]
    summary.update(
        basis="wrong",
        paired=True,
        ci_method="wrong",
        ci_reason="fallback",
        mean=99.0,
        preview_windows=0,
        final_windows=0,
        ci=[None, 0.0],
    )
    bootstrap = stats["bootstrap"]
    bootstrap.update(
        preview_final_delta_basis="wrong",
        preview_final_delta_method="wrong",
        preview_final_delta_seed=-1,
    )
    errors: list[str] = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors,
        cert_obj=payload,
        preview_mean=1.0,
        final_mean=1.0,
        tolerance=1e-9,
    )
    for expected in (
        "rejects legacy paired_delta_summary",
        ".basis must be independent",
        ".paired must be false",
        ".ci_method must be",
        "must not record a fallback reason",
        ".mean does not match",
        ".preview_windows must equal",
        "preview_final_delta_basis",
        "preview_final_delta_method",
        "preview_final_delta_seed",
        "ci bounds must be finite",
    ):
        assert any(expected in error for error in errors), errors

    summary["ci"] = [0.0, 0.0]
    monkeypatch.setattr(
        ppl_mod,
        "compute_independent_delta_log_ci",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad replay")),
    )
    errors = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors,
        cert_obj=payload,
        preview_mean=1.0,
        final_mean=1.0,
        tolerance=1e-9,
    )
    assert any("CI replay failed: bad replay" in error for error in errors)


def test_ppl_slice_summary_returns_on_missing_stats_raw_bootstrap_and_ci() -> None:
    payload = _strict_provenance_gate_cert()
    payload["dataset"]["windows"].pop("stats")
    errors: list[str] = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors, cert_obj=payload, preview_mean=1.0, final_mean=1.0, tolerance=1e-9
    )
    assert errors == []

    payload = _strict_provenance_gate_cert()
    payload["evaluation_windows"]["preview"]["logloss"] = []
    errors = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors, cert_obj=payload, preview_mean=1.0, final_mean=1.0, tolerance=1e-9
    )
    assert not any("CI replay failed" in error for error in errors)

    payload = _strict_provenance_gate_cert()
    payload["dataset"]["windows"]["stats"].pop("bootstrap")
    errors = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors, cert_obj=payload, preview_mean=1.0, final_mean=1.0, tolerance=1e-9
    )
    assert not any("preview_final_delta_seed" in error for error in errors)

    payload = _strict_provenance_gate_cert()
    payload["dataset"]["windows"]["stats"]["bootstrap"]["replicates"] = 0
    errors = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors, cert_obj=payload, preview_mean=1.0, final_mean=1.0, tolerance=1e-9
    )
    assert not any("preview_final_delta_seed" in error for error in errors)

    payload = _strict_provenance_gate_cert()
    payload["dataset"]["windows"]["stats"]["preview_final_slice_delta_summary"][
        "ci"
    ] = [0.0]
    errors = []
    ppl_mod._append_strict_preview_final_slice_summary_errors(
        errors, cert_obj=payload, preview_mean=1.0, final_mean=1.0, tolerance=1e-9
    )
    assert any("ci must contain two bounds" in error for error in errors)


@pytest.mark.parametrize(
    ("primary_metric", "expected"),
    [
        (
            {"analysis_basis": "wrong", "ci": [], "display_ci": []},
            "analysis_basis=mean_logloss",
        ),
        (
            {
                "analysis_basis": "mean_logloss",
                "ratio_vs_baseline": 0.0,
                "ci": [],
                "display_ci": [],
            },
            "ratio_vs_baseline must be finite and > 0",
        ),
        (
            {
                "analysis_basis": "mean_logloss",
                "ratio_vs_baseline": 1.0,
                "ci": [None, 0.0],
                "display_ci": [1.0, 1.0],
            },
            "bounds must be finite",
        ),
        (
            {
                "analysis_basis": "mean_logloss",
                "ratio_vs_baseline": 1.0,
                "ci": [1.0, 0.0],
                "display_ci": [math.e, 1.0],
            },
            "bounds must be ordered",
        ),
        (
            {
                "analysis_basis": "mean_logloss",
                "ratio_vs_baseline": 1.0,
                "ci": [0.0, 0.0],
                "display_ci": [0.0, 1.0],
            },
            "transforms must remain finite",
        ),
    ],
)
def test_ppl_coherence_rejects_forged_metric_surfaces(
    primary_metric: dict, expected: str
) -> None:
    payload = _strict_provenance_gate_cert()
    errors: list[str] = []
    ppl_mod._append_strict_ppl_coherence_errors(
        errors,
        cert_obj=payload,
        primary_metric=primary_metric,
        preview_mean=None,
        final_mean=1.0,
        tolerance=1e-9,
    )
    assert any(expected in error for error in errors), errors


def test_contract_helpers_reject_non_object_json_and_surface_warning_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        contract_mod._load_json_object_snapshot(path, object_name="acceptance report")

    assert (
        contract_mod._guard_warning_diagnostics(
            {"guard_warnings": {"warning_count": 0, "warnings": []}}
        )
        == ()
    )
    request = contract_mod.VerifyRequest(
        reports=(),
        tolerance="invalid",
        json_mode=False,
    )
    result = contract_mod._run_verify_request(request)
    assert result.outcome is contract_mod.VerifyOutcome.POLICY_FAIL
    assert result.diagnostics[0].message.startswith("Verification failed:")


def test_contract_recompute_rejects_window_id_length_and_duplicate_forks() -> None:
    cert = {
        "primary_metric": {"kind": "ppl_causal", "final": math.e},
        "evaluation_windows": {
            "final": {
                "logloss": [1.0, 1.0],
                "token_counts": [1, 1],
                "window_ids": [1, 1, 1],
            }
        },
    }
    errors: list[str] = []
    contract_mod._append_recompute_errors(
        errors,
        cert_obj=cert,
        prof="dev",
        tol=1e-9,
        json_mode=False,
    )
    assert any("window_ids length differs" in error for error in errors)
    assert any("window_ids contains duplicates" in error for error in errors)


def test_contract_reported_policy_digest_uses_each_supported_surface() -> None:
    assert (
        contract_mod._reported_policy_digest(
            {"policy_digest": {"thresholds_hash": "", "policy_digest": "second"}}
        )
        == "second"
    )
    assert (
        contract_mod._reported_policy_digest(
            {
                "policy_digest": {
                    "thresholds_hash": "",
                    "policy_digest": "",
                    "digest": "",
                },
                "provenance": {"policy": {"policy_digest": "provenance-digest"}},
            }
        )
        == "provenance-digest"
    )
    assert (
        contract_mod._reported_policy_digest(
            {"provenance": {"policy": {"policy_digest": ""}}}
        )
        is None
    )


def test_contract_single_report_loads_snapshot_and_enforces_guard_warning_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cert = {
        "primary_metric": {"kind": "ppl_causal", "final": math.e},
        "evaluation_windows": {"final": {"logloss": [1.0], "token_counts": [1]}},
        "guard_warnings": {
            "warning_count": 1,
            "warnings": [
                {
                    "guard": "spectral",
                    "kind": "policy_warning",
                    "policy_gate": "advisory",
                }
            ],
        },
    }
    cert_path = tmp_path / "subject.json"
    cert_path.write_text(json.dumps(cert), encoding="utf-8")
    monkeypatch.setattr(
        contract_mod,
        "_validate_evaluation_report_payload",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        contract_mod,
        "verify_runtime_provenance",
        lambda *_args, **_kwargs: RuntimeProvenanceResult(verified=True, skipped=False),
    )
    result = contract_mod._verify_single_report(
        cert_path,
        cert_snapshot=None,
        baseline=None,
        baseline_snapshot=None,
        baseline_payload=None,
        baseline_digest=None,
        policy_snapshot=None,
        policy_payload=None,
        tolerance=1e-9,
        profile="dev",
        allow_unverified_provenance=False,
        assurance_mode="report",
        warning_policy="fail",
        json_mode=False,
        expected_runtime_image_digest=None,
    )
    assert any("Guard warning policy failed: 1" in error for error in result.errors)
