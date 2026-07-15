from __future__ import annotations

import copy
import math

import pytest

from invarlock.reporting import verify_baseline as baseline_mod
from invarlock.reporting import verify_bootstrap as bootstrap_mod
from invarlock.reporting import verify_strict_schedule as schedule_mod
from tests.cli.verify._support_runtime_provenance import (
    _matching_strict_accuracy_baseline,
    _matching_strict_ppl_baseline,
    _strict_accuracy_cert,
    _strict_provenance_gate_cert,
)
from tests.reporting.validation._support_strict_verifier_branch_contracts import (
    _baseline_errors,
    _ExplodingFloatInt,
    _ExplodingIntString,
)


def test_numeric_helpers_fail_closed_on_non_json_and_refused_conversion() -> None:
    assert baseline_mod._finite_number(object()) is None
    assert baseline_mod._finite_number(_ExplodingFloatInt(1)) is None
    assert bootstrap_mod._finite_number(object()) is None
    assert bootstrap_mod._finite_number(_ExplodingFloatInt(1)) is None
    assert schedule_mod._strict_finite_number(_ExplodingFloatInt(1)) is None
    assert schedule_mod._schedule_window_id_key(_ExplodingIntString("x")) == (
        "text",
        "x",
    )
    digest = schedule_mod._schedule_digest([_ExplodingIntString("x")])
    assert isinstance(digest, str) and len(digest) == 32


def test_baseline_rejects_missing_and_mismatched_provenance_surfaces() -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["provenance"]["provider_digest"]["ids_sha256"] = ""
    report["dataset"]["seq_len"] = 0
    report["dataset"]["revision"] = "subject-rev"
    supplied["data"]["seq_len"] += 1
    supplied["data"]["revision"] = "baseline-rev"
    supplied["provenance"]["provider_digest"]["tokenizer_sha256"] = "other"
    supplied["meta"]["tokenizer_hash"] = "other"
    report["baseline_ref"].pop("tokenizer_hash")

    errors = _baseline_errors(report, supplied)

    assert any(
        "non-empty report.provenance.provider_digest.ids_sha256" in e for e in errors
    )
    assert any("positive report.dataset.seq_len" in e for e in errors)
    assert any("tokenizer digest mismatch" in e for e in errors)
    assert any("report.baseline_ref.tokenizer_hash" in e for e in errors)
    assert any("dataset revision mismatch" in e for e in errors)


@pytest.mark.parametrize(
    ("subject_revision", "baseline_revision", "message"),
    [
        ("subject", None, "dataset revision parity requires"),
        (None, "baseline", "dataset revision parity requires"),
    ],
)
def test_baseline_rejects_one_sided_optional_revision(
    subject_revision: str | None,
    baseline_revision: str | None,
    message: str,
) -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["dataset"].pop("revision", None)
    supplied["data"].pop("revision", None)
    if subject_revision is not None:
        report["dataset"]["revision"] = subject_revision
    if baseline_revision is not None:
        supplied["data"]["revision"] = baseline_revision

    assert any(message in error for error in _baseline_errors(report, supplied))


def test_baseline_raw_ppl_arms_require_complete_records_and_points() -> None:
    supplied = _matching_strict_ppl_baseline()
    supplied["evaluation_windows"].pop("preview")
    supplied["evaluation_windows"]["final"]["logloss"] = []
    supplied["metrics"]["primary_metric"].pop("final")
    errors: list[str] = []

    baseline_mod._append_raw_arm_completeness_errors(
        errors,
        baseline=supplied,
        metric_kind="ppl_causal",
    )

    assert any(
        "complete raw supplied_baseline.evaluation_windows.preview" in e for e in errors
    )
    assert any(
        "equal-length raw supplied_baseline.evaluation_windows.final" in e
        for e in errors
    )
    assert any("finite metrics.primary_metric.final" in e for e in errors)


def test_baseline_binary_and_identifier_parsers_reject_forged_values() -> None:
    assert baseline_mod._correctness_vector([]) is None
    assert baseline_mod._correctness_vector([2]) is None
    assert baseline_mod._correctness_vector(["yes"]) is None
    assert baseline_mod._record_correctness([]) is None
    assert baseline_mod._record_correctness([{"label": 1}]) is None

    for bad_ids, expected in (
        (None, "non-empty"),
        ([True], "JSON integer"),
        ([""], "JSON integer"),
        ([1, 1], "duplicates"),
    ):
        errors: list[str] = []
        assert (
            baseline_mod._canonical_sample_ids(
                errors,
                value=bad_ids,
                source="acceptance.example_ids",
            )
            is None
        )
        assert any(expected in error for error in errors)


def test_accuracy_baseline_rejects_invalid_aggregate_and_missing_raw_evidence() -> None:
    report = _strict_accuracy_cert()
    supplied = _matching_strict_accuracy_baseline(report)
    supplied["metrics"]["classification"]["n_total"] = 0
    errors: list[str] = []

    baseline_mod._append_accuracy_baseline_errors(
        errors,
        subject=report,
        baseline=supplied,
        tolerance=1e-9,
    )

    assert any("valid measured" in error for error in errors)

    supplied = _matching_strict_accuracy_baseline(report)
    classification = supplied["metrics"]["classification"]
    classification["counts_source"] = "estimated"
    classification["final"].pop("example_correct", None)
    supplied["evaluation_windows"]["final"].pop("example_correct", None)
    supplied["evaluation_windows"]["final"].pop("records", None)
    errors = []
    baseline_mod._append_accuracy_baseline_errors(
        errors,
        subject=report,
        baseline=supplied,
        tolerance=1e-9,
    )

    assert any("measured, non-estimated" in error for error in errors)
    assert any("raw per-example correctness evidence" in error for error in errors)


def test_accuracy_baseline_rejects_malformed_raw_sources_and_missing_window_block() -> (
    None
):
    report = _strict_accuracy_cert()
    supplied = _matching_strict_accuracy_baseline(report)
    supplied["metrics"]["classification"]["final"]["example_correct"] = [2]
    supplied["evaluation_windows"]["final"]["records"] = [{"label": 1}]
    errors: list[str] = []
    baseline_mod._append_accuracy_baseline_errors(
        errors,
        subject=report,
        baseline=supplied,
        tolerance=1e-9,
    )
    assert any("valid binary" in error for error in errors)
    assert any("records with boolean or 0/1" in error for error in errors)

    supplied["evaluation_windows"].pop("final")
    errors = []
    baseline_mod._append_accuracy_baseline_errors(
        errors,
        subject=report,
        baseline=supplied,
        tolerance=1e-9,
    )
    assert any(
        "requires supplied_baseline.evaluation_windows.final" in e for e in errors
    )


def test_baseline_reference_requires_both_binding_objects() -> None:
    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report.pop("baseline_ref")
    errors: list[str] = []
    baseline_mod._append_baseline_reference_binding_errors(
        errors,
        subject=report,
        baseline=supplied,
    )
    assert errors[-1] == "Strict baseline binding requires report.baseline_ref."

    report = _strict_provenance_gate_cert()
    report["baseline_ref"]["model_id"] = "wrong/model"
    report["provenance"].pop("baseline")
    errors = []
    baseline_mod._append_baseline_reference_binding_errors(
        errors,
        subject=report,
        baseline=supplied,
    )
    assert any("baseline_ref model_id mismatch" in error for error in errors)
    assert errors[-1] == "Strict baseline binding requires report.provenance.baseline."


def test_strict_baseline_contract_rejects_unloadable_and_metric_kind_forks() -> None:
    report = _strict_provenance_gate_cert()
    errors = _baseline_errors(report, None)
    assert errors == [
        "Strict assurance verification could not load the independently supplied baseline as a JSON object."
    ]

    supplied = _matching_strict_ppl_baseline(report)
    report["primary_metric"]["kind"] = object()
    errors = _baseline_errors(report, supplied)
    assert any(
        "supported subject and baseline primary metric kinds" in e for e in errors
    )

    report = _strict_accuracy_cert()
    supplied = _matching_strict_ppl_baseline()
    errors = _baseline_errors(report, supplied)
    assert any("primary metric kind differs" in e for e in errors)


def test_baseline_remaining_integer_raw_and_metric_kind_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    baseline_mod._append_required_integer_parity(
        errors,
        subject={"dataset": {"seq_len": 8}},
        baseline={"data": {"seq_len": 16}},
        path="dataset.seq_len",
        label="sequence length",
    )
    assert errors == [
        "Strict baseline sequence length mismatch: report.dataset.seq_len=8 supplied_baseline.data.seq_len=16."
    ]

    supplied = _matching_strict_accuracy_baseline(_strict_accuracy_cert())
    supplied["evaluation_windows"]["final"].pop("example_correct", None)
    supplied["evaluation_windows"]["final"]["records"] = [
        {"correct": value}
        for value in supplied["metrics"]["classification"]["final"]["example_correct"]
    ]
    errors = []
    baseline_mod._append_raw_arm_completeness_errors(
        errors,
        baseline=supplied,
        metric_kind="accuracy",
    )
    assert not any("one raw correctness record" in error for error in errors)

    supplied["evaluation_windows"]["final"].pop("records")
    errors = []
    baseline_mod._append_raw_arm_completeness_errors(
        errors,
        baseline=supplied,
        metric_kind="accuracy",
    )
    assert any("one raw correctness record" in error for error in errors)

    report = _strict_accuracy_cert()
    supplied = _matching_strict_accuracy_baseline(report)
    supplied["primary_metric"] = copy.deepcopy(supplied["metrics"]["primary_metric"])
    errors = []
    baseline_mod._append_accuracy_baseline_errors(
        errors,
        subject=report,
        baseline=supplied,
        tolerance=1e-9,
    )
    assert not any("accuracy metric/count mismatch" in error for error in errors)

    original = baseline_mod.normalize_metric_kind
    calls = 0

    def _raise_once(value: object) -> str | None:
        nonlocal calls
        calls += 1
        if calls in {1, 2}:
            raise RuntimeError("untrusted metric resolver")
        return original(value)

    monkeypatch.setattr(baseline_mod, "normalize_metric_kind", _raise_once)
    errors = _baseline_errors(
        _strict_provenance_gate_cert(), _matching_strict_ppl_baseline()
    )
    assert any(
        "supported subject and baseline primary metric kinds" in e for e in errors
    )

    report = _strict_provenance_gate_cert()
    supplied = _matching_strict_ppl_baseline(report)
    report["dataset"]["revision"] = "same"
    supplied["data"]["revision"] = "same"
    errors = []
    baseline_mod._append_provenance_parity_errors(
        errors,
        subject=report,
        baseline=supplied,
    )
    assert not any("dataset revision" in error for error in errors)

    supplied = _matching_strict_accuracy_baseline(_strict_accuracy_cert())
    for arm in ("preview", "final"):
        section = supplied["evaluation_windows"][arm]
        section["example_correct"] = [True] * len(section["example_ids"])
    errors = []
    baseline_mod._append_raw_arm_completeness_errors(
        errors,
        baseline=supplied,
        metric_kind="accuracy",
    )
    assert not any("one raw correctness record" in error for error in errors)


def test_schedule_metric_parser_and_baseline_candidates_fail_closed() -> None:
    for metric, expected in (
        (None, "as an object"),
        ({"kind": object(), "final": 1.0}, "supported"),
        ({"kind": "ppl_causal", "final": None}, "finite"),
        ({"kind": "ppl_causal", "final": 0.5}, ">= 1"),
        ({"kind": "accuracy", "final": 2.0}, "in [0,1]"),
    ):
        errors: list[str] = []
        assert (
            schedule_mod._metric_kind_and_final(
                errors,
                metric=metric,
                source="acceptance.metric",
            )
            is None
        )
        assert any(expected in error for error in errors)

    errors = []
    assert (
        schedule_mod._supplied_baseline_metric(
            errors,
            baseline_payload={},
            tolerance=1e-9,
        )
        is None
    )
    assert any("independently supplied baseline primary metric" in e for e in errors)

    errors = []
    assert schedule_mod._supplied_baseline_metric(
        errors,
        baseline_payload={
            "primary_metric": {"kind": "accuracy", "final": 0.5},
            "metrics": {"primary_metric": {"kind": "ppl_causal", "final": 2.0}},
        },
        tolerance=1e-9,
    ) == ("accuracy", 0.5)
    assert any("metric kind mismatch between" in e for e in errors)
    assert any("final mismatch between" in e for e in errors)


def test_schedule_metric_normalization_and_kind_forks_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = schedule_mod.normalize_metric_kind
    monkeypatch.setattr(
        schedule_mod,
        "normalize_metric_kind",
        lambda _value: (_ for _ in ()).throw(RuntimeError("resolver refused")),
    )
    errors: list[str] = []
    assert (
        schedule_mod._metric_kind_and_final(
            errors,
            metric={"kind": "ppl", "final": 2.0},
            source="acceptance.metric",
        )
        is None
    )
    assert any("supported acceptance.metric.kind" in error for error in errors)
    monkeypatch.setattr(schedule_mod, "normalize_metric_kind", original)

    for second_final, mismatch_expected in ((0.5, False), (0.7, True)):
        errors = []
        result = schedule_mod._supplied_baseline_metric(
            errors,
            baseline_payload={
                "primary_metric": {"kind": "accuracy", "final": 0.5},
                "metrics": {
                    "primary_metric": {"kind": "accuracy", "final": second_final}
                },
            },
            tolerance=1e-9,
        )
        assert result == ("accuracy", 0.5)
        assert any("final mismatch between" in e for e in errors) is mismatch_expected


def test_schedule_binding_rejects_reference_and_subject_metric_kind_forks() -> None:
    report = _strict_provenance_gate_cert()
    report["baseline_ref"]["primary_metric"] = {"kind": "accuracy", "final": 0.5}
    supplied = {"primary_metric": {"kind": "ppl_causal", "final": 9.0}}
    errors: list[str] = []
    schedule_mod._append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=supplied,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    assert any("report.baseline_ref=accuracy supplied=ppl_causal" in e for e in errors)

    report["baseline_ref"]["primary_metric"] = {"kind": "ppl_causal", "final": 9.0}
    report["primary_metric"] = {"kind": "accuracy", "final": 0.5}
    errors = []
    schedule_mod._append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=supplied,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    assert any(
        "report.primary_metric=accuracy supplied=ppl_causal" in e for e in errors
    )


@pytest.mark.parametrize("bad_ids", [None, [True], [""], [1, "1"]])
def test_schedule_ids_reject_missing_unstable_and_duplicate_identifiers(
    bad_ids: object,
) -> None:
    errors: list[str] = []
    assert (
        schedule_mod._canonical_schedule_ids(
            errors,
            value=bad_ids,
            source="acceptance.window_ids",
        )
        is None
    )
    assert errors


def test_declared_schedule_digest_rejects_non_hex_values() -> None:
    errors: list[str] = []
    digests = schedule_mod._declared_schedule_digests(
        errors,
        payload={
            "provenance": {"window_ids_digest": "not-a-digest"},
            "guard_metric_impact": {"schedule_digest": 123},
        },
        source_prefix="acceptance",
    )
    assert digests == []
    assert len(errors) == 2
    assert all("BLAKE2s-16 hex digest" in error for error in errors)


def test_supplied_baseline_binding_rejects_missing_payload_and_digest_only_strings() -> (
    None
):
    errors: list[str] = []
    schedule_mod._append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj={},
        baseline_payload=None,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    assert errors == [
        "Strict baseline binding could not load the independently supplied baseline."
    ]

    report = _strict_provenance_gate_cert()
    report["evaluation_windows"]["final"]["window_ids"] = ["opaque"]
    digest = schedule_mod._schedule_digest(["opaque"])
    supplied = {
        "primary_metric": {"kind": "ppl_causal", "final": 9.0},
        "provenance": {"window_ids_digest": digest},
    }
    errors = []
    schedule_mod._append_strict_supplied_baseline_binding_errors(
        errors,
        cert_obj=report,
        baseline_payload=supplied,
        baseline_supplied=True,
        tolerance=1e-9,
    )
    assert any(
        "digest-only baseline schedule binding requires signed 64-bit" in e
        for e in errors
    )


def test_bootstrap_raw_windows_reject_each_malformed_surface() -> None:
    cases = [
        ({}, "requires acceptance.final"),
        ({"evaluation_windows": {"final": {}}}, "non-empty raw"),
        (
            {
                "evaluation_windows": {
                    "final": {
                        "window_ids": [1, 2],
                        "logloss": [1.0],
                        "token_counts": [1, 1],
                    }
                }
            },
            "equal-length raw",
        ),
        (
            {
                "evaluation_windows": {
                    "final": {"window_ids": [1], "logloss": [1.0], "token_counts": [1]}
                }
            },
            "at least two paired final windows",
        ),
        (
            {
                "evaluation_windows": {
                    "final": {
                        "window_ids": [True, 2**64],
                        "logloss": [-1.0, math.nan],
                        "token_counts": [0, 2**54],
                    }
                }
            },
            "signed 64-bit JSON integer",
        ),
        (
            {
                "evaluation_windows": {
                    "final": {
                        "window_ids": [1, 1],
                        "logloss": [1.0, 1.0],
                        "token_counts": [1, 1],
                    }
                }
            },
            "contains duplicates",
        ),
    ]
    for payload, expected in cases:
        errors: list[str] = []
        assert (
            bootstrap_mod._parse_raw_final_windows(
                errors,
                payload=payload,
                source="acceptance.final",
            )
            is None
        )
        assert any(expected in error for error in errors)
    malformed_errors = errors
    assert malformed_errors


def test_bootstrap_provenance_rejects_missing_and_invalid_fields() -> None:
    errors: list[str] = []
    assert bootstrap_mod._parse_bootstrap_provenance(errors, {}) is None
    assert any("canonical dataset.windows.stats.bootstrap" in error for error in errors)

    report = _strict_provenance_gate_cert()
    bootstrap = report["dataset"]["windows"]["stats"]["bootstrap"]
    bootstrap.update(
        enabled=False,
        method="percentile",
        replicates=True,
        alpha=2.0,
        seed=-1,
    )
    errors = []
    assert bootstrap_mod._parse_bootstrap_provenance(errors, report) is None
    assert any("enabled=true" in error for error in errors)
    assert any("canonical method" in error for error in errors)
    assert any("positive JSON integer" in error for error in errors)
    assert any("alpha in (0,1)" in error for error in errors)
    assert any("seed in the supported range" in error for error in errors)


def test_bootstrap_coverage_mirror_rejects_forged_values() -> None:
    floors = {"preview": 2, "final": 2, "replicates": 4}
    used = {"preview": 1, "final": 2, "replicates": 3}
    errors: list[str] = []
    bootstrap_mod._validate_coverage_mirror(
        errors,
        coverage=None,
        source="coverage",
        tier="balanced",
        floors=floors,
        used=used,
        required=True,
    )
    assert errors == ["Strict assurance requires coverage coverage evidence."]

    errors = []
    bootstrap_mod._validate_coverage_mirror(
        errors,
        coverage={
            "tier": "wrong",
            "preview": {"used": "1", "required": 0, "ok": "yes"},
            "final": {"used": 99, "required": "2", "ok": False},
        },
        source="coverage",
        tier="balanced",
        floors=floors,
        used=used,
        required=True,
    )
    assert any("tier must match" in error for error in errors)
    assert any("preview.used must be" in error for error in errors)
    assert any("preview.required must equal" in error for error in errors)
    assert any("preview.ok must be a boolean" in error for error in errors)
    assert any("final.used disagrees" in error for error in errors)
    assert any("final.required must be" in error for error in errors)
    assert any("coverage.replicates evidence" in error for error in errors)
    assert any("below the canonical strict" in error for error in errors)

    errors = []
    bootstrap_mod._validate_coverage_mirror(
        errors,
        coverage=None,
        source="optional",
        tier="balanced",
        floors=floors,
        used=used,
        required=False,
    )
    assert errors == []


def test_bootstrap_evidence_volume_rejects_forged_mirrors_and_unknown_profile() -> None:
    report = _strict_provenance_gate_cert()
    report["assurance"]["profile"] = "dev"
    report["dataset"]["windows"] = {"stats": {"coverage": {}}}
    preview = bootstrap_mod._RawFinalWindows((1, 2), (1.0, 1.0), (1, 1))
    final = bootstrap_mod._RawFinalWindows((2, 3), (1.0, 1.0), (1, 1))
    baseline = bootstrap_mod._RawFinalWindows((2,), (1.0,), (1,))
    provenance = bootstrap_mod._BootstrapProvenance(1, 0.5, 0)
    errors: list[str] = []

    bootstrap_mod._append_strict_evidence_volume_errors(
        errors,
        report=report,
        subject_preview=preview,
        subject_final=final,
        baseline_final=baseline,
        provenance=provenance,
    )

    assert any(
        "raw preview/final window IDs must be disjoint" in error for error in errors
    )
    assert any("bootstrap alpha must equal" in error for error in errors)
    assert any("dataset.windows.preview must be" in error for error in errors)
    assert any("profile ci or release" in error for error in errors)


def test_bootstrap_evidence_volume_rejects_count_and_nested_coverage_forks() -> None:
    report = _strict_provenance_gate_cert()
    report["dataset"]["windows"]["stats"]["bootstrap"]["coverage"] = None
    preview = bootstrap_mod._RawFinalWindows((1, 2), (1.0, 1.0), (1, 1))
    final = bootstrap_mod._RawFinalWindows((3, 4, 5), (1.0, 1.0, 1.0), (1, 1, 1))
    baseline = bootstrap_mod._RawFinalWindows((3,), (1.0,), (1,))
    provenance = bootstrap_mod._BootstrapProvenance(1200, 0.05, 0)
    errors: list[str] = []
    bootstrap_mod._append_strict_evidence_volume_errors(
        errors,
        report=report,
        subject_preview=preview,
        subject_final=final,
        baseline_final=baseline,
        provenance=provenance,
    )
    assert any("equal raw preview and final window counts" in error for error in errors)

    report["dataset"]["windows"].pop("stats")
    errors = []
    bootstrap_mod._append_strict_evidence_volume_errors(
        errors,
        report=report,
        subject_preview=preview,
        subject_final=final,
        baseline_final=baseline,
        provenance=provenance,
    )
    assert errors[-1] == "strict assurance requires dataset.windows.stats evidence."

    report = _strict_provenance_gate_cert()
    nested = copy.deepcopy(report["dataset"]["windows"]["stats"]["coverage"])
    report["dataset"]["windows"]["stats"]["bootstrap"]["coverage"] = nested
    nested["preview"]["used"] = 0
    errors = []
    bootstrap_mod._append_strict_evidence_volume_errors(
        errors,
        report=report,
        subject_preview=bootstrap_mod._RawFinalWindows(
            tuple(range(180)), (1.0,) * 180, (1,) * 180
        ),
        subject_final=bootstrap_mod._RawFinalWindows(
            tuple(range(180, 360)), (1.0,) * 180, (1,) * 180
        ),
        baseline_final=bootstrap_mod._RawFinalWindows(
            tuple(range(180, 360)), (1.0,) * 180, (1,) * 180
        ),
        provenance=bootstrap_mod._BootstrapProvenance(1200, 0.05, 0),
    )
    assert any("bootstrap.coverage.preview.used disagrees" in error for error in errors)
